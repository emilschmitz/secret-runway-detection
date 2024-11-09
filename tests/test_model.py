import pytest
import torch
import os
import sys

# sys.path.append(os.path.abspath('../GFM'))  # Adjust the path as necessary
sys.path.append(os.path.abspath('GFM'))  # Adjust the path as necessary

from secret_runway_detection.model import (
    SimpleSegmentationHead,
    # MultiscaleSegmentationHead,
    SimpleSegmentationModel,
    UPerNetSegmentationModel,
    get_model,
    create_args,
    get_config
)

# Define paths for configuration and pretrained weights
CONFIG_PATH = 'configs/gfm_config.yaml'  # Adjust as necessary
PRETRAINED_WEIGHTS_PATH = 'simmim_pretrain/gfm.pth'  # Adjust as necessary

# Mock configuration for testing
class MockConfig:
    class MODEL:
        TYPE = 'swin'
        NAME = 'simmim_finetune'
        DROP_PATH_RATE = 0.1

        class SWIN:
            EMBED_DIM = 128
            DEPTHS = [2, 2, 18, 2]
            NUM_HEADS = [4, 8, 16, 32]
            WINDOW_SIZE = 6
            IN_CHANS = 3

    class DATA:
        IMG_SIZE = 192

    class TRAIN:
        EPOCHS = 100
        WARMUP_EPOCHS = 20
        BASE_LR = 1.0e-4
        WARMUP_LR = 1.0e-7
        MIN_LR = 1.0e-5
        WEIGHT_DECAY = 0.05
        LAYER_DECAY = 0.9
        DROP_PATH_RATE = 0.1

    PRINT_FREQ = 100
    SAVE_FREQ = 5
    TAG = 'simmim_finetune__swin_base__img128_window4__100ep'

@pytest.fixture
def mock_config():
    return MockConfig()

@pytest.fixture
def dummy_input():
    # Update dummy input dimensions
    return torch.randn(2, 3, 192, 192)  # Increased batch size and spatial dimensions

def test_simple_segmentation_head():
    model = SimpleSegmentationHead(embedding_dim=1000, output_channels=1, output_size=192)
    dummy_input = torch.randn(1, 1000)
    output = model(dummy_input)
    assert output.shape == (1, 1, 192, 192), f"Unexpected output shape: {output.shape}"

def test_upernet_segmentation_model(dummy_input):
    # Modify test to use more appropriate input dimensions
    model = UPerNetSegmentationModel(
        config=MockConfig(),  # Use the mock config
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,
        output_size=192
    )
    model.eval()  # Set to evaluation mode to avoid batch norm issues
    
    # Use larger spatial dimensions for input
    with torch.no_grad():  # Use no_grad for testing
        output = model(dummy_input)
    
    assert output.shape == (2, 1, 192, 192), f"Unexpected output shape: {output.shape}"

def test_get_model_simple(dummy_input):
    model = get_model(
        model_type='simple',
        cfg_path=CONFIG_PATH,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,
        output_size=192
    )
    model.eval()  # Set to evaluation mode
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1, 192, 192), f"Unexpected output shape: {output.shape}"

def test_get_model_upernet(dummy_input):
    model = get_model(
        model_type='upernet',
        cfg_path=CONFIG_PATH,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
        num_classes=1,          # For binary segmentation
        output_size=192         # Desired output resolution
    )    
    assert isinstance(model, UPerNetSegmentationModel), "Model type mismatch for 'upernet'"

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1, 192, 192), f"Unexpected output shape: {output.shape}"