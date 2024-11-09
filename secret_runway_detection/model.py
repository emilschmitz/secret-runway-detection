# model.py

import os
import sys
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from types import SimpleNamespace

# Suppress specific warnings (optional)
import warnings
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
            
            # Optional: Add a prebuilt block (e.g., BasicBlock from torchvision)
            BasicBlock(128, 128),
            
            # Upsample from (12x12) to (24x24)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            BasicBlock(64, 64),
            
            # Upsample from (24x24) to (48x48)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            BasicBlock(32, 32),
            
            # Upsample from (48x48) to (96x96)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            BasicBlock(16, 16),
            
            # Upsample from (96x96) to (192x192)
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            
            # Refinement layers
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_channels, self.initial_grid_size, self.initial_grid_size)
        x = self.decoder(x)
        return x

# Define the Multiscale Segmentation Head (UPerNet Style)
class MultiscaleSegmentationHead(nn.Module):
    def __init__(self, in_channels, output_channels=1, output_size=128):
        super(MultiscaleSegmentationHead, self).__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=2, stride=2),
            # nn.Sigmoid()  # Use Softmax for multi-class segmentation
        )

    def forward(self, x):
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bicubic', align_corners=False)
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
        self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, pretrained_weights_path):
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
        missing_keys, unexpected_keys = self.backbone.load_state_dict(filtered_state_dict, strict=False)
        
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

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
        
        # Assuming the backbone returns a single feature map; adjust if multiple
        if isinstance(features, list) or isinstance(features, tuple):
            features = features[-1]  # Use the last feature map
        x = features
        
        # Forward pass through the segmentation head
        x = self.segmentation_head(x)
        return x

# Define the combined UPerNetSegmentationModel
class UPerNetSegmentationModel(nn.Module):
    def __init__(self, config, pretrained_weights_path, num_classes=1, output_size=128):
        super(UPerNetSegmentationModel, self).__init__()
        from transformers import (
            SwinConfig,
            UperNetConfig,
            UperNetForSemanticSegmentation
        )
        
        # Define the Swin Transformer configuration
        backbone_config = SwinConfig(
            embed_dim=GFM_OUTPUT_DIM,  # Adjust based on the backbone's output dimension
            depths=config.MODEL.DEPTHS,
            num_heads=config.MODEL.NUM_HEADS,
            window_size=config.MODEL.WINDOW_SIZE,
            drop_path_rate=config.TRAIN.DROP_PATH_RATE,
            out_features=config.MODEL.OUT_FEATURES,
        )
        
        # Define the UPerNet configuration
        upernet_config = UperNetConfig(
            backbone_config=backbone_config,
            num_labels=num_classes,
            pool_scales=(1, 2, 3, 6),
            aux_channels=256,
            use_auxiliary_head=False,  # Set to True if you want to use an auxiliary head
        )
        
        # Initialize the UPerNet model
        self.upernet = UperNetForSemanticSegmentation(upernet_config)
        
        # Load pretrained weights into the backbone
        self.load_pretrained_weights(pretrained_weights_path)
        
        # Initialize the segmentation head (optional, depending on UPerNet's design)
        # If UPerNet already includes a head, you might not need an additional one.
        # Adjust accordingly based on your specific needs.
        
        # Example: Overriding the head to match the desired output size
        self.upernet.decoder.segmentation_head = MultiscaleSegmentationHead(
            in_channels=upernet_config.backbone_config.embed_dim * 4,  # Adjust based on backbone output
            output_channels=num_classes,
            output_size=output_size
        )

    def load_pretrained_weights(self, pretrained_weights_path):
        print("\nLoading pretrained weights into UPerNet backbone...")
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        
        # Extract the state_dict
        state_dict = checkpoint.get('model', checkpoint)
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Remove 'encoder.' prefix if necessary
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        
        # Load the state dict into the UPerNet's backbone
        missing_keys, unexpected_keys = self.upernet.backbone.load_state_dict(state_dict, strict=False)
        
        # Debugging output
        if missing_keys:
            print("\nMissing keys when loading pretrained weights into UPerNet backbone:")
            for key in missing_keys:
                print(f"  - {key}")
        if unexpected_keys:
            print("\nUnexpected keys when loading pretrained weights into UPerNet backbone:")
            for key in unexpected_keys:
                print(f"  - {key}")
        
        if not missing_keys and not unexpected_keys:
            print("\nAll pretrained weights loaded into UPerNet backbone successfully!")
        else:
            print("\nPretrained weights loaded into UPerNet backbone with some missing/unexpected keys.")

    def forward(self, x):
        # Forward pass through UPerNet
        outputs = self.upernet(x)
        return outputs.logits  # Assuming you want the segmentation logits

# Factory methods to instantiate models
def get_model(model_type, cfg_path, pretrained_weights_path, num_classes=1, output_size=128):
    """
    Factory function to get the desired segmentation model.

    Parameters:
    - model_type (str): 'simple' or 'upernet'
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
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'simple' or 'upernet'.")
    
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

    # Instantiate a UPerNet segmentation model
    upernet_model = get_model(
        model_type='upernet',
        cfg_path=CONFIG_PATH,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,  # Binary segmentation
        output_size=128
    )
    print("\nUPerNetSegmentationModel instantiated successfully.")

    # Perform a dummy forward pass with SimpleSegmentationModel
    dummy_input = torch.randn(1, 3, 192, 192)  # Adjust IMG_SIZE as per your config
    simple_model.eval()
    with torch.no_grad():
        simple_output = simple_model(dummy_input)
    print(f"\nSimpleSegmentationModel Dummy Output Shape: {simple_output.shape}")  # Expected: [1, 1, 128, 128]

    # Perform a dummy forward pass with UPerNetSegmentationModel
    upernet_model.eval()
    with torch.no_grad():
        upernet_output = upernet_model(dummy_input)
    print(f"\nUPerNetSegmentationModel Dummy Output Shape: {upernet_output.shape}")  # Expected: [1, 1, 128, 128]
