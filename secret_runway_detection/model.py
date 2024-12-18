# model.py

import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# Suppress specific warnings (optional)
import warnings

import transformers
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the path to the cloned GFM repository to sys.path
sys.path.append(os.path.abspath('../GFM'))  # Adjust the path as necessary

# Import build_model and get_config from GFM
from GFM.models import build_model
from GFM.config import get_config

GFM_OUTPUT_DIM = 1000  # Adjust based on the backbone's output dimension

# Define a helper class to simulate argparse.Namespace
class Args:
    def __init__(self, cfg, opts=None, batch_size=None, data_path=None, pretrained=None,
                 resume=None, accumulation_steps=None, use_checkpoint=False, amp_opt_level=None,
                 output=None, tag=None, eval=False, throughput=False, train_frac=None,
                 no_val=False, alpha=None, local_rank=0):
        self.cfg = cfg
        self.opts = opts
        self.batch_size = batch_size
        self.data_path = data_path
        self.pretrained = pretrained
        self.resume = resume
        self.accumulation_steps = accumulation_steps
        self.use_checkpoint = use_checkpoint
        self.amp_opt_level = amp_opt_level
        self.output = output
        self.tag = tag
        self.eval = eval
        self.throughput = throughput
        self.train_frac = train_frac
        self.no_val = no_val
        self.alpha = alpha
        self.local_rank = local_rank

def create_args(cfg_path, pretrained_path):
    """
    Creates an Args instance with necessary attributes.
    """
    return Args(
        cfg=cfg_path,                # Path to the configuration YAML
        pretrained=pretrained_path   # Path to the pretrained weights (gfm.pth)
    )

def _load_pretrained_weights(backbone, pretrained_weights_path) -> None:
        print("\nLoading pretrained weights...")
        state_dict = torch.load(pretrained_weights_path, map_location='cpu')
        
        # Adjust based on how the state_dict is stored
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Optionally, remove 'encoder.' prefix if necessary
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        
        # Filter out keys related to segmentation head if present
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('head.', 'decoder.', 'classifier.',
                                                                                        'teacher.', 'projector.'))}
        
        # Load the state dict into the backbone
        missing_keys, unexpected_keys = backbone.load_state_dict(filtered_state_dict, strict=False)
        
        # Debugging output
        if missing_keys:
            print("\nMissing keys when loading pretrained weights:")
            for key in missing_keys:
                print(f"  - {key}")
        if unexpected_keys:
            print("\nUnexpected keys when loading pretrained weights:")
            for key in unexpected_keys:
                print(f"  - {key}")
        
        if not missing_keys and not unexpected_keys:
            print("\nAll pretrained weights loaded successfully!")
        else:
            print("\nPretrained weights loaded with some missing/unexpected keys.")


# Define the Simple Segmentation Head (Transposed Convolutional Head)
class SimpleSegmentationHead(nn.Module):
    def __init__(self, embedding_dim=1000, output_channels=1, output_size=192):
        super(SimpleSegmentationHead, self).__init__()
        self.output_size = output_size
        self.initial_grid_size = 6  # Adjust based on the backbone's output resolution
        self.num_channels = 256

        self.fc = nn.Linear(embedding_dim, self.num_channels * self.initial_grid_size * self.initial_grid_size)
        self.decoder = nn.Sequential(
            # Upsample from (6x6) to (12x12)
            nn.ConvTranspose2d(self.num_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Additional layers for increased complexity
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Upsample from (12x12) to (24x24)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Upsample from (24x24) to (48x48)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Upsample from (48x48) to (96x96)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Additional layers to maintain resolution
            nn.Conv2d(16, 16, kernel_size=7, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=7, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=7, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Upsample from (96x96) to (192x192)
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),
            # # Final layer to match output channels
            # nn.ConvTranspose2d(8, output_channels, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()  # Use Softmax for multi-class segmentation
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_channels, self.initial_grid_size, self.initial_grid_size)
        x = self.decoder(x)
        # print(f"Output shape before interpolation: {x.shape}")
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bicubic', align_corners=False)
        return x

from types import SimpleNamespace
from transformers.models.upernet.modeling_upernet import UperNetHead

# Define the configuration parameters required by UperNetHead
upernet_config = SimpleNamespace(
    pool_scales=(1, 2, 3, 6),        # Example pool scales
    hidden_size=512,                 # Number of hidden channels in UPerNet
    num_labels=1,                    # Number of segmentation classes
    initializer_range=0.02,          # Weight initialization range
    use_auxiliary_head=False,        # Whether to use an auxiliary head
    # auxiliary_in_channels=[256, 512, 1024, 1024],  # Channels for auxiliary head
    # auxiliary_channels=256,          # Channels in auxiliary head
    # auxiliary_num_convs=1,           # Number of convolutions in auxiliary head
    # auxiliary_concat_input=False,    # Whether to concatenate input in auxiliary head
    loss_ignore_index=255            # Ignore index for loss computation
)


class UPerNetSegmentationModel(nn.Module):
    """https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/upernet/modeling_upernet.py#L356
    """
    def __init__(self, config, pretrained_weights_path, num_classes=1, output_size=192):
        super(UPerNetSegmentationModel, self).__init__()
        
        # Initialize the backbone
        self.backbone = MultiscaleOutputBackbone(config, pretrained_weights_path)
        
        # Define the input channels for UperNetHead based on backbone's feature channels
        # Your backbone outputs feature maps with channels: [256, 512, 1024, 1024]
        self.decode_head = UperNetHead(
            config=upernet_config,
            in_channels=[3, 
                        #  128, 
                         256, 
                        #  512, 
                        #  1024, 1024
                         ]  # Align with your backbone's output channels
        )
        
        # # make transposed convolutions to upsample the patch_embeds to the desired resolutions
        # self.patch_embed_upsample_96 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.patch_embed_upsample_192 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        # )

        self.layer_space_dims = {
            # 'patch_embed': 2304,
            'layer0': 576,
            # 'layer1': 144,
            # 'layer2': 36,
            # 'layer3': 36
        }

        self.fcs = nn.ModuleDict({
            layer: nn.Linear(space_dim, space_dim) 
            for layer, space_dim in self.layer_space_dims.items()
        })

        # # Initialize the auxiliary head if required by config
        # if config.use_auxiliary_head:
        #     self.auxiliary_head = transformers.UperNetFCNHead(config)
        # else:
        #     self.auxiliary_head = None
        
        # Define the desired output size
        self.output_size = output_size

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)  # Should return a dict with keys like 'layer0', 'layer1', etc.
        
        for key in self.layer_space_dims.keys():
            # Replace transpose with permute for correct dimension ordering
            features[key] = features[key].permute(0, 2, 1)  # Adjust the order based on desired dimensions
            features[key] = self.fcs[key](features[key])
            features[key] = features[key].reshape(x.size(0), features[key].size(1), 
                np.sqrt(features[key].size(2)), np.sqrt(features[key].size(2)))

        # # Upsample patch_embed twice to double resp. quadruble resolution
        # patch_embed_96 = self.patch_embed_upsample_96(features['patch_embed'])
        # patch_embed_192 = self.patch_embed_upsample_192(features['patch_embed'])

        # Convert the feature dictionary to a list ordered by resolution (low to high)
        # UperNetHead expects features ordered from low to high resolution
        feature_list = [
            x,
            # features['patch_embed'].reshape(x.size(0), 128, 48, 48),
            features['layer0'].reshape(x.size(0), x.size(1), 24, 24),  # layer0: [B, 256, 24, 24]
            # features['layer1'].reshape(x.size(0), x.size(1), 12, 12),  # layer1: [B, 512, 12, 12]
            # features['layer2'].reshape(x.size(0), x.size(1), 6, 6),   # layer2: [B, 1024, 6, 6]
            # features['layer3'].reshape(x.size(0), x.size(1), 6, 6)    # layer3: [B, 1024, 6, 6]
        ]
        
        # Pass the ordered feature list to the decode head
        logits = self.decode_head(feature_list)
        
        # Interpolate to match the desired output size
        logits = nn.functional.interpolate(
            logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        
        # Handle auxiliary head if present
        # if self.auxiliary_head is not None:
        #     auxiliary_logits = self.auxiliary_head(feature_list)
        #     auxiliary_logits = nn.functional.interpolate(
        #         auxiliary_logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        #     )
        #     return logits, auxiliary_logits
        
        return logits
    

class MidFeatSegmentationModel(nn.Module):
    """https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/upernet/modeling_upernet.py#L356
    """
    def __init__(self, config, pretrained_weights_path, num_classes=1, output_size=192):
        super(MidFeatSegmentationModel, self).__init__()
        
        # Initialize the backbone
        self.backbone = MultiscaleOutputBackbone(config, pretrained_weights_path)
        
        # Define the input channels for UperNetHead based on backbone's feature channels
        # Your backbone outputs feature maps with channels: [256, 512, 1024, 1024]
        self.decode_head = DeepSegmentationHead(256)
        
        self.layer_space_dims = {
            # 'patch_embed': 2304,
            'layer0': 576,
            'layer1': 144,
            # 'layer2': 36,
            # 'layer3': 36
        }

        self.fcs = nn.ModuleDict({
            layer: nn.Linear(space_dim, space_dim) 
            for layer, space_dim in self.layer_space_dims.items()
        })

        self.conv_double = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.output_size = output_size

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)  # Should return a dict with keys like 'layer0', 'layer1', etc.
        
        for key in ['layer0']:
            # Replace transpose with permute for correct dimension ordering
            features[key] = features[key].permute(0, 2, 1)  # Adjust the order based on desired dimensions
            features[key] = self.fcs[key](features[key])
            sqrt_dim = int(np.sqrt(features[key].size(2)))
            features[key] = features[key].reshape(x.size(0), features[key].size(1), 
                sqrt_dim, sqrt_dim)

        # features['layer1'] = self.conv_double(features['layer1'])

        # Pass the ordered feature list to the decode head  
        logits = self.decode_head(features['layer0'])
        

        return logits


class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class DeepSegmentationHead(nn.Module):
    def __init__(self, input_channels, output_channels=1, output_size=192, num_resblocks=3):
        super(DeepSegmentationHead, self).__init__()
        self.output_size = output_size

        # Upsample to ~96x96
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, input_channels,
                kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        # Initial Residual Block with large kernel size
        self.resblock1 = ResidualBlock(
            input_channels, kernel_size=9, padding=4)

        # Upsample to ~192x192
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, input_channels // 2,
                kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Create a ModuleList of Residual Blocks
        self.resblocks = nn.ModuleList([
            ResidualBlock(
                input_channels // 2, kernel_size=9, padding=4)
            for _ in range(num_resblocks)
        ])

        # Convolutional layers to reduce channels
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(
                input_channels // 2, input_channels // 4,
                kernel_size=9, padding=4),
            nn.BatchNorm2d(input_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channels // 4, input_channels // 8,
                kernel_size=9, padding=4),
            nn.BatchNorm2d(input_channels // 8),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.final_conv = nn.Conv2d(
            input_channels // 8, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.resblock1(x)

        x = self.up2(x)

        # Apply the residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        x = self.conv_reduce(x)
        x = self.final_conv(x)

        # Ensure output size matches desired dimensions
        if x.shape[-1] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode='bicubic',
                align_corners=False
            )
        return x


# Define the combined SimpleSegmentationModel
class SimpleSegmentationModel(nn.Module):
    def __init__(self, config, pretrained_weights_path, output_channels=1, output_size=128):
        super(SimpleSegmentationModel, self).__init__()
        # Build the backbone model using GFM's build_model
        self.backbone = build_model(config, is_pretrain=False)
        
        # Initialize the segmentation head
        self.segmentation_head = SimpleSegmentationHead(
            embedding_dim=GFM_OUTPUT_DIM, 
            output_channels=output_channels,
            output_size=output_size
        )
        
        # Load pretrained weights
        _load_pretrained_weights(self.backbone, pretrained_weights_path)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Forward pass through the segmentation head
        x = self.segmentation_head(x)
        return x

class MultiscaleOutputBackbone(nn.Module):
    """
    Outputs dict with keys:
        - 'patch_embed': the tensor of patch embeddings
        - 'pos_drop': the tensor after position embedding and dropout
        - 'layer0': the tensor after the 1st Swin Transformer layer
        - 'layer1': the tensor after the 2nd Swin Transformer layer
        - ...
        - 'norm': the tensor after the final layer norm
        - 'final_feature': the final feature tensor
        - 'output': the final output tensor
    """
    def __init__(self, config, pretrained_weights_path):
        super(MultiscaleOutputBackbone, self).__init__()
        # Build the backbone model using GFM's build_model
        self.model = build_model(config, is_pretrain=False)
        
        # Load pretrained weights
        _load_pretrained_weights(self.model, pretrained_weights_path)

    def forward(self, x):
        # Forward pass through the backbone
        return self.model.forward_features(x)

class CNNSegmentationModel(nn.Module):
    """
    A simple CNN-based segmentation model that maps 192x192 input images to 192x192 output maps.
    """
    def __init__(self, config, pretrained_weights_path, output_channels=1, output_size=192, debug=False):
        super(CNNSegmentationModel, self).__init__()
        self.debug = debug  # Debug flag to control print statements
        
        # Downsampling layers with Batch Normalization
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 192x192 -> 96x96
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers with Batch Normalization
        self.upsample = nn.Sequential(
            # Adjusted ConvTranspose2d layers: Removed output_padding=1 for stride=1
            nn.ConvTranspose2d(32, 16, kernel_size=7, stride=1, padding=3),  # 96x96 -> 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=1, padding=3),  # 96x96 -> 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=1, padding=3),  # 96x96 -> 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=1, padding=3),  # 96x96 -> 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=1, padding=3),  # 96x96 -> 96x96
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final ConvTranspose2d to upsample back to 192x192
            nn.ConvTranspose2d(16, output_channels, kernel_size=7, stride=2, padding=3, output_padding=1),  # 96x96 -> 192x192
            nn.ReLU(inplace=True),
            
            nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1),
        )
        
        self._initialize_weights()  # Initialize weights

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")  # Optional debug print
        
        x = self.downsample(x)
        if self.debug:
            print(f"After downsample: {x.shape}")  # Optional debug print
        
        x = self.upsample(x)
        if self.debug:
            print(f"After upsample: {x.shape}")  # Optional debug print
        
        return x

class CustomSegmentationModel(nn.Module):
    """
    Custom segmentation model optimized for 192x192 images with class imbalance.
    Uses a mix of efficient convolutions and attention mechanisms.
    """
    def __init__(self, input_channels=3, output_channels=1, base_channels=32):
        super(CustomSegmentationModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 192x192 -> 96x96
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 96x96 -> 48x48
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 48x48 -> 24x24 with dilated convolutions
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Channel Attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            # Block 1: 24x24 -> 48x48
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True)
            ),
            # Block 2: 48x48 -> 96x96
            nn.Sequential(
                nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            # Block 3: 96x96 -> 192x192
            nn.Sequential(
                nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True)  # Keep positive activations for better class imbalance handling
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Apply channel attention
        att = self.attention(features)
        features = features * att
        
        # Decoder
        for decoder_block in self.decoder:
            features = decoder_block(features)
            
        # Final convolution
        return self.final_conv(features)

class SimpleCustomSegmentationModel(nn.Module):
    """
    A simpler custom segmentation model optimized for 192x192 images.
    Uses basic conv-bn-relu blocks with skip connections.
    """
    def __init__(self, input_channels=3, output_channels=1):
        super(SimpleCustomSegmentationModel, self).__init__()
        
        # First conv block (192x192 -> 96x96)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Second conv block (96x96 -> 48x48)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Third conv block with dilation (48x48 -> 48x48)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # First deconv block (48x48 -> 96x96)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Second deconv block (96x96 -> 192x192)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)      # 192->96
        x2 = self.conv2(x1)     # 96->48
        x3 = self.conv3(x2)     # 48->48 (dilated)
        
        # Decoder with skip connections
        x = self.deconv1(x3)    # 48->96
        x = x + x1              # Skip connection
        x = self.deconv2(x)     # 96->192
        x = self.final(x)       # Final conv
        
        return x

# Factory methods to instantiate models
def get_model(model_type, cfg_path, pretrained_weights_path, num_classes=1, output_size=192):
    """
    Factory function to get the desired segmentation model.

    Parameters:
    - model_type (str): 'simple', 'upernet', or 'cnn'
    - cfg_path (str): Path to the configuration YAML
    - pretrained_weights_path (str): Path to the pretrained weights
    - num_classes (int): Number of segmentation classes
    - output_size (int): Desired output image size

    Returns:
    - model (nn.Module): The instantiated segmentation model
    """
    args = create_args(cfg_path, pretrained_weights_path)
    config = get_config(args)
    
    if model_type == 'simple':
        model = SimpleSegmentationModel(
            config=config,
            pretrained_weights_path=pretrained_weights_path,
            output_channels=num_classes,
            output_size=output_size
        )
    elif model_type == 'upernet':
        model = UPerNetSegmentationModel(
            config=config,
            pretrained_weights_path=pretrained_weights_path,
            num_classes=num_classes,
            output_size=output_size
        )
    elif model_type == 'cnn':
        model = CNNSegmentationModel(
            config=config,
            pretrained_weights_path=pretrained_weights_path,
            output_channels=num_classes,
            output_size=output_size
        )
    elif model_type == 'midfeat':
        model = MidFeatSegmentationModel(
            config=config,
            pretrained_weights_path=pretrained_weights_path,
            num_classes=num_classes,
            output_size=output_size
        )
    elif model_type == 'custom':
        model = CustomSegmentationModel(
            input_channels=3,
            output_channels=num_classes,
            base_channels=32
        )
    elif model_type == 'simple_custom':
        model = SimpleCustomSegmentationModel(
            input_channels=3,
            output_channels=num_classes
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'simple', 'upernet', or 'cnn'.")
    
    return model

# Example usage (this part can be removed or commented out in the actual model.py file)
if __name__ == "__main__":
    # Define paths
    CONFIG_PATH = '../configs/gfm_config.yaml'  # Adjust as necessary
    PRETRAINED_WEIGHTS_PATH = '../simmim_pretrain/gfm.pth'  # Adjust as necessary
    
    # Instantiate a simple segmentation model
    simple_model = get_model(
        model_type='simple',
        cfg_path=CONFIG_PATH,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,  # Binary segmentation
        output_size=128
    )
    print("\nSimpleSegmentationModel instantiated successfully.")

    bb = simple_model.backbone
    
    # # Perform a dummy forward pass with SimpleSegmentationModel
    dummy_input = torch.randn(5, 3, 192, 192)  # Adjust IMG_SIZE as per your config
    
    bb_feats = bb.forward_features(dummy_input)

    for key, value in bb_feats.items():
        print(f"Key: {key}, Shape: {value.shape}")

    # Instantiate a UPerNet segmentation model
    upernet_model = get_model(
        model_type='upernet',
        cfg_path=CONFIG_PATH,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,          # For binary segmentation
        output_size=192         # Desired output resolution
    )
    print("\nUPerNetSegmentationModel instantiated successfully.")

    print(upernet_model(dummy_input))