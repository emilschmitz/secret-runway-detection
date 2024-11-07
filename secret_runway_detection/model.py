import torch
import torch.nn as nn
import timm
from secret_runway_detection.train_utils import TILES_PER_AREA_LEN

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
    model = timm.create_model(model_name, pretrained=False, num_classes=0).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Extract and clean the state dictionary
    state_dict = checkpoint.get('model', checkpoint)
    new_state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=False)
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


# Factory methods
def get_simple_segmentation_model(model_path):
    backbone = SwinBackbone(model_path=model_path)
    segmentation_head = SimpleSegmentationHead()
    return SimpleSegmentationModel(backbone, segmentation_head)


def get_multiscale_segmentation_model(model_path):
    backbone = SwinBackboneWithIntermediateOutputs(model_path=model_path)
    segmentation_head = MultiscaleSegmentationHead(in_channels=256)
    return MultiscaleSegmentationModel(backbone, segmentation_head)
