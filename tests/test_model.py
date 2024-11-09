import pytest
import torch
import os
import sys

# sys.path.append(os.path.abspath('../GFM'))  # Adjust the path as necessary
sys.path.append(os.path.abspath('GFM'))  # Adjust the path as necessary

from secret_runway_detection.model import (
    SimpleSegmentationHead,
    MultiscaleSegmentationHead,
    SimpleSegmentationModel,
    UPerNetSegmentationModel,
    get_model,
    create_args,
    get_config
)

# Define paths for configuration and pretrained weights
CONFIG_PATH = '../configs/gfm_config.yaml'  # Adjust as necessary
PRETRAINED_WEIGHTS_PATH = '../simmim_pretrain/gfm.pth'  # Adjust as necessary

# Mock configuration for testing
class MockConfig:
    class MODEL:
        DEPTHS = [2, 2, 18, 2]
        NUM_HEADS = [4, 8, 16, 32]
        WINDOW_SIZE = 7
        OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
    class TRAIN:
        DROP_PATH_RATE = 0.1

@pytest.fixture
def mock_config():
    return MockConfig()

@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 192, 192)  # Adjust IMG_SIZE as per your config

def test_simple_segmentation_head():
    model = SimpleSegmentationHead(embedding_dim=1000, output_channels=1, output_size=192)
    dummy_input = torch.randn(1, 1000)
    output = model(dummy_input)
    assert output.shape == (1, 1, 192, 192), f"Unexpected output shape: {output.shape}"

def test_multiscale_segmentation_head():
    model = MultiscaleSegmentationHead(in_channels=256, output_channels=1, output_size=128)
    dummy_input = torch.randn(1, 256, 8, 8)
    output = model(dummy_input)
    assert output.shape == (1, 1, 128, 128), f"Unexpected output shape: {output.shape}"

def test_simple_segmentation_model(mock_config, dummy_input):
    model = SimpleSegmentationModel(config=mock_config, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, output_channels=1, output_size=128)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 1, 128, 128), f"Unexpected output shape: {output.shape}"

def test_upernet_segmentation_model(mock_config, dummy_input):
    model = UPerNetSegmentationModel(config=mock_config, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, num_classes=1, output_size=128)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 1, 128, 128), f"Unexpected output shape: {output.shape}"

def test_get_model_simple():
    model = get_model(model_type='simple', cfg_path=CONFIG_PATH, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, num_classes=1, output_size=128)
    assert isinstance(model, SimpleSegmentationModel), "Model type mismatch for 'simple'"

def test_get_model_upernet():
    model = get_model(model_type='upernet', cfg_path=CONFIG_PATH, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, num_classes=1, output_size=128)
    assert isinstance(model, UPerNetSegmentationModel), "Model type mismatch for 'upernet'"