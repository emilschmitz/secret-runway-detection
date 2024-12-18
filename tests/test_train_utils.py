# tests/test_train_utils.py

import os
import sys
import numpy as np
import pytest
from shapely.geometry import Polygon
import geopandas as gpd
import torch

# # Add the src directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'secret_runway_detection')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(os.getcwd())

from secret_runway_detection.train_utils import (
    add_buffer_to_label,
    landing_strips_to_enclosing_input_areas,
    input_area_to_has_strip_tensor,
    input_area_to_input_image,
    make_input_image_tensor,
    LandingStripDataset,
    make_train_set
)

# Constants (Ensure these match the values in train_utils.py)
TILE_SIDE_LEN = 10.0  # meters per pixel
# NB: CHECKED WITH ORGANIZERS (this height-row and width-column doesn't seem to make any sense)
# TRANSPOSING FOR NOW
AOI_HEIGHT = 15270.0  # in meters
AOI_WIDTH = 15410.0   # in meters
# Cols and rows of tiles in an AOI
# ROWS_COUNT = 1541  # counts zero-indexed
# COLUMNS_COUNT = 1527  
ROWS_COUNT = 1527  # counts zero-indexed
COLUMNS_COUNT = 1541
assert (TILE_SIDE_LEN == AOI_HEIGHT / ROWS_COUNT) and (TILE_SIDE_LEN == AOI_WIDTH / COLUMNS_COUNT)
INPUT_IMAGE_HEIGHT = 224  # in pixels
INPUT_IMAGE_WIDTH = 224
INPUT_AREAS_VERTICALLY = 10
INPUT_AREAS_HORIZONTALLY = 10

# Fixtures

@pytest.fixture
def sample_aoi():
    """Fixture for a sample AOI Polygon using AOI_HEIGHT and AOI_WIDTH."""
    minx = 0
    miny = 0
    maxx = minx + AOI_WIDTH
    maxy = miny + AOI_HEIGHT
    return Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny)
    ])

@pytest.fixture
def sample_tiles_gdf(sample_aoi):
    """Fixture for a sample tiles GeoDataFrame covering the AOI."""
    tiles_gdf = landing_strips_to_enclosing_input_areas(sample_aoi, num_tiles_per_area_side_len=10)
    return tiles_gdf

@pytest.fixture
def sample_input_areas_gdf(sample_aoi):
    """Fixture for a sample input areas GeoDataFrame."""
    input_areas_gdf = landing_strips_to_enclosing_input_areas(sample_aoi, num_tiles_per_area_side_len=10)
    return input_areas_gdf

@pytest.fixture
def dummy_model():
    """Fixture for a dummy PyTorch model."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Return random predictions
            return torch.rand(x.size(0), 1, x.size(2), x.size(3))
    return DummyModel()

# Test Cases

def test_landing_strips_to_input_areas(sample_aoi):
    """Test the landing_strips_to_input_areas function."""
    # Create sample landing strips within the AOI
    landing_strip_polygon = Polygon([
        (-70.08928656863503, -13.129844039931504), 
        (-70.08052787039317, -13.128251134052485), 
        (-70.08052787039317, -13.128251134052485)
    ])
    landing_strips = gpd.GeoDataFrame({'geometry': [landing_strip_polygon]}, crs='EPSG:3263')
    
    input_areas = landing_strips_to_enclosing_input_areas(landing_strips, num_tiles_per_area_side_len=224)
    
    # Check if the number of input areas matches the number of landing strips
    assert len(input_areas) == len(landing_strips), "Number of input areas does not match number of landing strips."
    
    # Check if input areas contain the landing strips
    for idx, area in input_areas.iterrows():
        assert area.geometry.contains(landing_strip_polygon), f"Input area at index {idx} does not contain the landing strip."

def test_input_area_to_input_image(sample_aoi):
    """Test the input_area_to_input_image function."""
    input_area_polygon = sample_aoi
    input_image = input_area_to_input_image(input_area_polygon, input_area_crs=None)
    
    # Check the shape of the input image
    assert input_image.shape == (3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), \
        f"Expected input image shape (3, {INPUT_IMAGE_HEIGHT}, {INPUT_IMAGE_WIDTH}), got {input_image.shape}."
    
    # Check the data type
    assert input_image.dtype == np.float32, "Input image dtype is not float32."

def test_make_input_tensor():
    """Test the make_input_tensor function."""
    # Create a dummy input image
    input_image = np.zeros((3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=np.float32)
    
    input_tensor = make_input_image_tensor(input_image)
    
    # Expected shape: (1, 3, 224, 224)
    expected_shape = (1, 3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert input_tensor.shape == expected_shape, f"Expected input tensor shape {expected_shape}, got {input_tensor.shape}."
    
    # Check the data type
    assert input_tensor.dtype == torch.float32, "Input tensor dtype is not float32."

def test_input_area_to_has_strip_tensor(sample_aoi):
    """Test the make_label_tensor function."""
    input_image_polygon = sample_aoi
    # Create a simple landing strip within the AOI
    landing_strip_polygon = Polygon([
        (5000, 5000),
        (7000, 5000),
        (7000, 7000),
        (5000, 7000),
        (5000, 5000)
    ])
    landing_strips = gpd.GeoDataFrame({'geometry': [landing_strip_polygon]}, crs='EPSG:32633')
    
    label_tensor = input_area_to_has_strip_tensor(input_image_polygon, landing_strips, input_area_crs=landing_strips.crs, buffer_type='cross')
    
    expected_shape = (1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert label_tensor.shape == expected_shape, f"Expected label tensor shape {expected_shape}, got {label_tensor.shape}."
    assert isinstance(label_tensor, torch.Tensor), "Label tensor is not a torch.Tensor."
    
    # Check that the label tensor has at least one positive label
    assert torch.sum(label_tensor) >= 1, "Label tensor should contain at least one positive label."

def test_add_buffer_to_label_cross():
    """
    Test the add_buffer_to_label function.
    """
    # Input label array
    label = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.uint8)
    
    # Expected output with num_buffer_tiles == 1
    expected_output = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ], dtype=np.uint8)
    
    # Call the function with num_buffer_tiles == 1
    buffered_label = add_buffer_to_label(label, num_buffer_tiles=1, buffer_type='cross')
    
    # Assert that the output matches the expected output
    assert np.array_equal(buffered_label, expected_output), "Buffered label does not match expected output with num_buffer_tiles=1."
    
    # Test with num_buffer_tiles == 0 (should be the same as the input)
    buffered_label_zero = add_buffer_to_label(label, num_buffer_tiles=0, buffer_type='cross')
    assert np.array_equal(buffered_label_zero, label), "Buffered label does not match input when num_buffer_tiles=0."
    
    # Test with num_buffer_tiles == 2
    expected_output_2 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    buffered_label_2 = add_buffer_to_label(label, num_buffer_tiles=2, buffer_type='cross')
    assert np.array_equal(buffered_label_2, expected_output_2), "Buffered label does not match expected output with num_buffer_tiles=2."

def test_add_buffer_to_label_empty():
    """
    Test add_buffer_to_label with an empty label array.
    """
    label = np.zeros((5, 5), dtype=np.uint8)
    expected_output = np.zeros((5, 5), dtype=np.uint8)
    
    buffered_label = add_buffer_to_label(label, num_buffer_tiles=1, buffer_type='ball')
    assert np.array_equal(buffered_label, expected_output), "Buffered label should be empty when input label is empty."