from torch.utils.data import Dataset
import torch
from pathlib import Path
import logging
import numpy as np


logger = logging.getLogger(__name__)


class LandingStripDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        # Get list of image files
        self.image_files = sorted(self.images_dir.glob('*.npy'))
        # Map image files to corresponding label files
        self.samples = []
        for image_file in self.image_files:
            label_file = self.labels_dir / image_file.name
            if label_file.exists():
                self.samples.append((image_file, label_file))
            else:
                logger.warning(f"Label file {label_file} does not exist.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, label_file = self.samples[idx]
        # Load image and label
        image = np.load(image_file)
        label = np.load(label_file)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float()  # Shape (tiles_per_area_len, tiles_per_area_len)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor