import pytest
import torch
import numpy as np
from pathlib import Path
from secret_runway_detection.dataset import LandingStripDataset, RandomRotate90, SegmentationTransform


# Create temporary directories and files for testing
@pytest.fixture
def setup_test_data(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create dummy image and label files
    for i in range(3):
        image = np.random.rand(3, 256, 256).astype(np.float32)
        label = np.random.randint(0, 2, (256, 256)).astype(np.float32)
        np.save(images_dir / f"image_{i}.npy", image)
        np.save(labels_dir / f"image_{i}.npy", label)

    return images_dir, labels_dir

def test_random_rotate_90():
    transform = RandomRotate90()
    image = torch.rand(3, 256, 256)
    label = torch.rand(256, 256)

    rotated_image, rotated_label = transform(image, label)

    assert rotated_image.shape == image.shape
    assert rotated_label.shape == label.shape

def test_segmentation_transform():
    transform = SegmentationTransform()
    image = torch.rand(3, 256, 256)
    label = torch.rand(256, 256)

    transformed_image, transformed_label = transform(image, label)

    assert transformed_image.shape == image.shape
    assert transformed_label.shape == label.shape

def test_landing_strip_dataset(setup_test_data):
    images_dir, labels_dir = setup_test_data
    transform = SegmentationTransform()
    dataset = LandingStripDataset(images_dir, labels_dir, transform=transform)

    assert len(dataset) == 3

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.shape == torch.Size([3, 256, 256])
        assert label.shape == torch.Size([256, 256])

if __name__ == "__main__":
    pytest.main()