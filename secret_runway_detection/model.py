import torch.nn as nn
from secret_runway_detection.train_utils import (
    TILES_PER_AREA_LEN,
)


class SegmentationHead(nn.Module):
    def __init__(self, embedding_dim=1024, output_size=TILES_PER_AREA_LEN):
        super(SegmentationHead, self).__init__()
        self.output_size = output_size
        self.initial_grid_size = 5  # Starting grid size (adjustable)
        self.num_channels = 256     # Number of feature channels (adjustable)

        # Fully connected layer to map embedding to initial grid
        self.fc = nn.Linear(embedding_dim, self.num_channels * self.initial_grid_size * self.initial_grid_size)

        # Decoder layers to upsample to output_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.num_channels, 128, kernel_size=4, stride=2, padding=1),  # Upsample x2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                # Upsample x2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),                 # Upsample x2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),                 # Upsample x2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),                  # Upsample x2
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),                   # Upsample x1
        )

    def forward(self, x):
        # x: (batch_size, embedding_dim)
        x = self.fc(x)  # (batch_size, num_channels * initial_grid_size * initial_grid_size)
        x = x.view(-1, self.num_channels, self.initial_grid_size, self.initial_grid_size)
        x = self.decoder(x)  # (batch_size, 1, H, W), where H and W should be close to output_size

        # If the output size is not exactly 200x200, interpolate
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bicubic', align_corners=False)
        return x  # (batch_size, 1, output_size, output_size)
    
class CombinedModel(nn.Module):
    def __init__(self, backbone, segmentation_head):
        super(CombinedModel, self).__init__()
        self.backbone = backbone
        self.segmentation_head = segmentation_head

    def forward(self, x):
        # Pass input through backbone to get embedding
        x = self.backbone(x)  # Output shape: (batch_size, embedding_dim)
        x = self.segmentation_head(x)
        return x
    