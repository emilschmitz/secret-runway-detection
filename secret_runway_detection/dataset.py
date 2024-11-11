from torch.utils.data import Dataset
import torch
from pathlib import Path
import logging
import numpy as np
import torchvision.transforms as T
import random


logger = logging.getLogger(__name__)


class LandingStripDataset(Dataset):
    def __init__(self, images_dir, labels_dir, file_list=None, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        if file_list is not None:
            # Use the provided list of filenames
            self.image_files = [self.images_dir / filename for filename in file_list]
        else:
            # Get list of all image files in the directory
            self.image_files = sorted(self.images_dir.glob('*.npy'))

        # Map image files to corresponding label files
        self.samples = []
        for image_file in self.image_files:
            label_file = self.labels_dir / image_file.name
            if label_file.exists():
                self.samples.append((image_file, label_file))
            else:
                logger.warning(f"Label file {label_file} does not exist.")

        # Calculate class weights based on pixel distribution
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self, samples_to_use=1000):
        """
        Calculate class weights based on pixel distribution.
        Uses a subset of samples for efficiency.
        """
        pos_pixels = 0
        total_pixels = 0
        
        # Use a subset of samples for calculation
        sample_indices = random.sample(
            range(len(self.samples)), 
            min(samples_to_use, len(self.samples))
        )
        
        for idx in sample_indices:
            _, label_file = self.samples[idx]
            label = np.load(label_file)
            pos_pixels += np.sum(label)
            total_pixels += label.size
        
        # Calculate weights
        pos_ratio = pos_pixels / total_pixels
        neg_ratio = 1 - pos_ratio
        
        # Create balanced weights
        pos_weight = 1 / (2 * pos_ratio) if pos_ratio > 0 else 1
        neg_weight = 1 / (2 * neg_ratio) if neg_ratio > 0 else 1
        
        # Return as tensor
        weights = torch.FloatTensor([neg_weight, pos_weight])
        
        print(f"Class distribution - Negative: {neg_ratio:.3%}, Positive: {pos_ratio:.3%}")
        print(f"Class weights - Negative: {neg_weight:.3f}, Positive: {pos_weight:.3f}")
        
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, label_file = self.samples[idx]

        # Load image and label
        image = np.load(image_file)
        label = np.load(label_file)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).float()

        if self.transform:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)

        return image_tensor, label_tensor


class RandomHorizontalFlip:
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])  # Flip image horizontally
            label = torch.flip(label, dims=[1])  # Flip label horizontally
        return image, label

class RandomVerticalFlip:
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])  # Flip image vertically
            label = torch.flip(label, dims=[0])  # Flip label vertically
        return image, label

class RandomRotate90:
    def __call__(self, image, label):
        k = random.randint(0, 3)  # Randomly choose rotation
        image = torch.rot90(image, k, [1, 2])  # Rotate image
        label = torch.rot90(label, k, [0, 1])  # Rotate label
        return image, label

class RandomColorJitter:
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image, label):
        # ColorJitter expects (C,H,W) input
        image = self.color_jitter(image)
        return image, label

class RandomNoise:
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, image, label):
        noise = torch.randn_like(image) * self.std
        image = image + noise
        image = torch.clamp(image, 0, 1)
        return image, label

class SegmentationTransform:
    def __init__(self, p_noise=0.3):
        self.transforms = [
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate90(),
            RandomColorJitter(
                brightness=0.1,
                contrast=0.1, 
                saturation=0.1,
                hue=0.05
            ),
        ]
        self.noise = RandomNoise(std=0.01)
        self.p_noise = p_noise

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
            
        # Apply noise with probability p_noise
        if random.random() < self.p_noise:
            image, label = self.noise(image, label)
        
        return image, label