import timm
import torch
import torch.nn as nn
from transformers import (
    SwinConfig,
    UperNetConfig,
    UperNetForSemanticSegmentation,
    AutoImageProcessor
)
from transformers.models.swin.convert_swin_timm_to_pytorch import convert_state_dict
from secret_runway_detection.train_utils import TILES_PER_AREA_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import timm
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gfm_model(model_name, model_path):
    """
    Loads the Swin Transformer model with pretrained GFM weights.

    Parameters:
    - model_name (str): The Swin model architecture name.
    - model_path (str): Path to the model checkpoint.

    Returns:
    - model (torch.nn.Module): Loaded model with weights.
    """
    # Initialize the timm model without pretrained weights
    model = timm.create_model(model_name, pretrained=False, num_classes=0).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the state_dict; adjust if the checkpoint has a different structure
    state_dict = checkpoint.get('model', checkpoint)
    
    # Remove 'module.' prefix if present (common with DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Filter out only the encoder's parameters
    encoder_prefix = 'encoder.'
    filtered_state_dict = {k.replace(encoder_prefix, ''): v for k, v in state_dict.items() if k.startswith(encoder_prefix)}
    
    # Optional: Rename keys if necessary to match timm's naming conventions
    # For most timm models, removing 'encoder.' should suffice
    # However, verify by printing some keys
    print("Sample keys from filtered_state_dict:")
    for k in list(filtered_state_dict.keys())[:5]:
        print(k)
    
    # Load the filtered state_dict into the timm model
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print("\nMissing keys when loading pretrained weights:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print("\nUnexpected keys when loading pretrained weights:")
        for key in unexpected_keys:
            print(f"  - {key}")
    
    return model


# Simple Segmentation Head (Single Embedding Approach)
class SimpleSegmentationHead(nn.Module):
    def __init__(self, embedding_dim=1024, output_size=TILES_PER_AREA_LEN):
        super(SimpleSegmentationHead, self).__init__()
        self.output_size = output_size
        self.initial_grid_size = 5
        self.num_channels = 256

        self.fc = nn.Linear(embedding_dim, self.num_channels * self.initial_grid_size * self.initial_grid_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.num_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_channels, self.initial_grid_size, self.initial_grid_size)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bicubic', align_corners=False)
        return x

# Multiscale Segmentation Head (U-Net Style with Skip Connections)
class MultiscaleSegmentationHead(nn.Module):
    def __init__(self, in_channels=256, output_size=TILES_PER_AREA_LEN):
        super(MultiscaleSegmentationHead, self).__init__()
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bicubic', align_corners=False)
        return x

# Backbone classes for both models
class SwinBackboneWithIntermediateOutputs(nn.Module):
    def __init__(self, model_path, model_name='swin_base_patch4_window7_224'):
        super(SwinBackboneWithIntermediateOutputs, self).__init__()
        self.backbone = load_gfm_model(model_name, model_path)
        self.feature_channels = [128, 256, 512, 1024]

    def forward(self, x):
        features = []
        x = self.backbone.patch_embed(x)
        x = self.backbone.layers[0](x)
        features.append(x)
        x = self.backbone.layers[1](x)
        features.append(x)
        x = self.backbone.layers[2](x)
        features.append(x)
        x = self.backbone.layers[3](x)
        features.append(x)
        return features

class SwinBackbone(nn.Module):
    def __init__(self, model_path, model_name='swin_base_patch4_window7_224'):
        super(SwinBackbone, self).__init__()
        self.backbone = load_gfm_model(model_name, model_path)

    def forward(self, x):
        return self.backbone(x)

# Combined model classes for both variants
class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, segmentation_head):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.segmentation_head = segmentation_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.segmentation_head(x)
        return x

class MultiscaleSegmentationModel(nn.Module):
    def __init__(self, backbone, segmentation_head):
        super(MultiscaleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.conv1x1_layers = nn.ModuleList([
            nn.Conv2d(in_ch, 256, kernel_size=1) for in_ch in self.backbone.feature_channels
        ])

    def forward(self, x):
        features = self.backbone(x)
        features = [self.conv1x1_layers[i](feat) for i, feat in enumerate(features)]
        combined_feature = features[0]
        for i in range(1, len(features)):
            upsampled_feat = nn.functional.interpolate(features[i], scale_factor=2 ** i, mode='bilinear', align_corners=False)
            combined_feature += upsampled_feat
        x = self.segmentation_head(combined_feature)
        return x

def load_gfm_weights_into_swin(model, gfm_checkpoint_path):
    """
    Loads the GFM pretrained weights into the Swin backbone of the UPerNet model.

    Parameters:
    - model (UperNetForSemanticSegmentation): The UPerNet model.
    - gfm_checkpoint_path (str): Path to the GFM pretrained weights.

    Returns:
    - model (UperNetForSemanticSegmentation): The UPerNet model with loaded weights.
    """
    # Load the GFM checkpoint
    checkpoint = torch.load(gfm_checkpoint_path, map_location='cpu')

    # Get the state_dict from the checkpoint
    state_dict = checkpoint.get('model', checkpoint)

    # Remove 'module.' prefix if present (if the model was trained using DataParallel or DistributedDataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Convert the timm state dict to Hugging Face format
    # converted_state_dict = convert_state_dict(state_dict, model.backbone)

    # Load the converted state dict into the Swin backbone
    missing_keys, unexpected_keys = model.backbone.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing keys when loading pretrained weights:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys when loading pretrained weights:", unexpected_keys)

    return model


def get_upernet_segmentation_model(model_path, num_classes=1):
    """
    Creates a UPerNet segmentation model with a Swin Transformer backbone,
    and loads the pretrained GFM weights into the backbone.

    Parameters:
    - model_path (str): Path to the GFM pretrained weights.
    - num_classes (int): Number of classes for segmentation.

    Returns:
    - model (UperNetForSemanticSegmentation): UPerNet segmentation model with loaded weights.
    """
    # Define the Swin Transformer configuration
    backbone_config = SwinConfig(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.3,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )

    # Define the UPerNet configuration
    config = UperNetConfig(
        backbone_config=backbone_config,
        num_labels=num_classes,
        pool_scales=(1, 2, 3, 6),
        aux_channels=256,
        use_auxiliary_head=False,  # Set to True if you want to use an auxiliary head
    )

    # Initialize the UPerNet model
    model = UperNetForSemanticSegmentation(config)

    # Load GFM pretrained weights into the Swin backbone
    model = load_gfm_weights_into_swin(model, model_path)

    # Move the model to the appropriate device
    model.to(device)

    return model

# Factory methods
def get_simple_segmentation_model(model_path):
    backbone = SwinBackbone(model_path=model_path)
    segmentation_head = SimpleSegmentationHead()
    return SimpleSegmentationModel(backbone, segmentation_head)

def get_multiscale_segmentation_model(model_path):
    backbone = SwinBackboneWithIntermediateOutputs(model_path=model_path)
    segmentation_head = MultiscaleSegmentationHead(in_channels=256)
    return MultiscaleSegmentationModel(backbone, segmentation_head)
