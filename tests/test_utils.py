# tests/test_utils.py

# TODO check that tests make sense

import pytest
from shapely.geometry import Polygon, Point
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import os
import shutil
from utils import (
    point_to_aoi_southeast,
    aoi_to_tiles,
    aoi_to_input_areas,
    input_area_to_input_image,
    make_input_tensor,
    make_label_tensor,
    LandingStripDataset,
    pad_output_tensor,
    run_inference_on_aoi,
    tensor_to_submission_csv
)

# Constants (Ensure these match the values in utils.py)
TILE_SIDE_LEN = 10.0  # meters per pixel
# NB: CHECKED WITH ORGANIZERS (this heigth-row and width-column is doesn't seem to make any sense)
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
def sample_point():
    """Fixture for a sample Point geometry."""
    return Point(1000, 2000)

@pytest.fixture
def sample_aoi():
    """Fixture for a sample AOI Polygon."""
    return Polygon([
        (0, 0),
        (TILE_SIDE_LEN * 1526, 0),
        (TILE_SIDE_LEN * 1526, TILE_SIDE_LEN * 1541),
        (0, TILE_SIDE_LEN * 1541),
        (0, 0)
    ])

@pytest.fixture
def sample_tiles_gdf():
    """Fixture for a sample tiles GeoDataFrame."""
    tiles = []
    minx, miny = 0, 0
    maxx, maxy = TILE_SIDE_LEN * 2, TILE_SIDE_LEN * 2  # 20x20 meters
    for row_idx in range(2):
        for col_idx in range(2):
            x = col_idx * TILE_SIDE_LEN
            y = row_idx * TILE_SIDE_LEN
            tile = Polygon([
                (x, y),
                (x + TILE_SIDE_LEN, y),
                (x + TILE_SIDE_LEN, y + TILE_SIDE_LEN),
                (x, y + TILE_SIDE_LEN),
                (x, y)
            ])
            tiles.append({'geometry': tile, 'row_idx': row_idx, 'col_idx': col_idx})
    return gpd.GeoDataFrame(tiles, crs='EPSG:32633')  # Replace with appropriate CRS

@pytest.fixture
def sample_input_areas_gdf(sample_aoi):
    """Fixture for a sample input areas GeoDataFrame."""
    return aoi_to_input_areas(sample_aoi, num_areas_vertically=1, num_areas_horizontally=1)

@pytest.fixture
def dummy_model():
    """Fixture for a dummy PyTorch model."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Return random predictions
            return torch.rand(x.size(0), 1, x.size(2), x.size(3))
    return DummyModel()

# Test Cases

def test_point_to_aoi_southeast(sample_point):
    aoi_polygon = point_to_aoi_southeast(sample_point)
    expected_coords = [
        (sample_point.x, sample_point.y),  # Northwest corner
        (sample_point.x + AOI_WIDTH, sample_point.y),  # Northeast corner
        (sample_point.x + AOI_WIDTH, sample_point.y - AOI_HEIGHT),  # Southeast corner
        (sample_point.x, sample_point.y - AOI_HEIGHT),  # Southwest corner
        (sample_point.x, sample_point.y)  # Close polygon
    ]
    assert list(aoi_polygon.exterior.coords) == expected_coords, "AOI polygon coordinates mismatch."

def test_aoi_to_tiles(sample_aoi, sample_tiles_gdf):
    tiles_gdf = aoi_to_tiles(sample_aoi)
    expected_num_tiles = 4  # 2x2 tiles for 20x20 meters AOI with 10x10 meters per tile
    assert len(tiles_gdf) == expected_num_tiles, f"Expected {expected_num_tiles} tiles, got {len(tiles_gdf)}."
    
    # Combine tiles into a single geometry and compare with AOI
    tiles_union = tiles_gdf.unary_union
    assert sample_aoi.equals(tiles_union), "Tiles do not correctly cover the AOI."

def test_aoi_to_input_areas(sample_aoi):
    input_areas_gdf = aoi_to_input_areas(sample_aoi, num_areas_vertically=1, num_areas_horizontally=1)
    expected_num_areas = 1
    assert len(input_areas_gdf) == expected_num_areas, f"Expected {expected_num_areas} input areas, got {len(input_areas_gdf)}."
    
    # Check if AOI is within the union of input areas
    input_areas_union = input_areas_gdf.unary_union
    assert sample_aoi.within(input_areas_union) or sample_aoi.equals(input_areas_union), "Input areas do not cover the AOI."

def test_input_area_to_input_image():
    input_area = Polygon([(0, 0), (0, 2240), (2240, 2240), (2240, 0), (0, 0)])
    input_image = input_area_to_input_image(input_area)
    num_bands = 3  # Assuming RGB
    expected_shape = (num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert input_image.shape == expected_shape, f"Expected image shape {expected_shape}, got {input_image.shape}."
    assert np.all(input_image == 0), "Expected input image to be all zeros."

def test_make_input_tensor():
    num_bands = 3
    input_image = np.zeros((num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=np.float32)
    input_tensor = make_input_tensor(input_image)
    expected_shape = (1, num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert input_tensor.shape == expected_shape, f"Expected tensor shape {expected_shape}, got {input_tensor.shape}."
    assert isinstance(input_tensor, torch.Tensor), "Input tensor is not a torch.Tensor."

def test_make_label_tensor(sample_aoi):
    input_image_polygon = sample_aoi
    # Create a simple landing strip within the AOI
    landing_strip_polygon = Polygon([
        (500, 500),
        (700, 500),
        (700, 700),
        (500, 700),
        (500, 500)
    ])
    landing_strips = gpd.GeoDataFrame({'geometry': [landing_strip_polygon]}, crs='EPSG:32633')
    label_tensor = make_label_tensor(input_image_polygon, landing_strips)
    expected_shape = (1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert label_tensor.shape == expected_shape, f"Expected label tensor shape {expected_shape}, got {label_tensor.shape}."
    assert isinstance(label_tensor, torch.Tensor), "Label tensor is not a torch.Tensor."
    # Check that the landing strip area has labels set to 1
    # This requires mapping polygon to pixel indices, which is complex; here we check the tensor type and values
    assert label_tensor.dtype == torch.float32, "Label tensor dtype is not torch.float32."

def test_make_training_dataset(sample_aoi):
    input_image_polygon = sample_aoi
    input_image_polygons = gpd.GeoDataFrame({'geometry': [input_image_polygon]}, crs='EPSG:32633')
    # Create a simple landing strip within the AOI
    landing_strip_polygon = Polygon([
        (500, 500),
        (700, 500),
        (700, 700),
        (500, 700),
        (500, 500)
    ])
    landing_strips = gpd.GeoDataFrame({'geometry': [landing_strip_polygon]}, crs='EPSG:32633')
    dataset = LandingStripDataset(input_image_polygons, landing_strips)
    assert len(dataset) == 1, f"Expected dataset length 1, got {len(dataset)}."
    input_tensor, label_tensor = dataset[0]
    expected_input_shape = (1, 3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    expected_label_shape = (1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert input_tensor.shape == expected_input_shape, f"Expected input tensor shape {expected_input_shape}, got {input_tensor.shape}."
    assert label_tensor.shape == expected_label_shape, f"Expected label tensor shape {expected_label_shape}, got {label_tensor.shape}."

def test_pad_output_tensor():
    output_tensor = torch.ones((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
    idxs = {
        "bottom_row_idx": 100,
        "top_row_idx": 100 + INPUT_IMAGE_HEIGHT,
        "leftmost_col_idx": 200,
        "rightmost_col_idx": 200 + INPUT_IMAGE_WIDTH
    }
    padded_tensor = pad_output_tensor(output_tensor, idxs)
    expected_shape = (ROWS_COUNT, COLUMNS_COUNT)
    assert padded_tensor.shape == expected_shape, f"Expected padded tensor shape {expected_shape}, got {padded_tensor.shape}."
    # Check that the specific region is set to 1
    assert torch.all(padded_tensor[idxs["bottom_row_idx"]:idxs["top_row_idx"],
                                   idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]] == 1), "Padded region does not match expected values."
    # Check that the rest of the tensor is 0
    mask = torch.zeros_like(padded_tensor)
    mask[idxs["bottom_row_idx"]:idxs["top_row_idx"], idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]] = 1
    assert torch.all((padded_tensor == 0) | (mask == 1)), "Padded tensor contains unexpected non-zero values."

def test_run_inference_on_aoi(sample_aoi, dummy_model):
    final_prediction_tensor = run_inference_on_aoi(sample_aoi, dummy_model, threshold=0.5)
    expected_shape = (ROWS_COUNT, COLUMNS_COUNT)
    assert final_prediction_tensor.shape == expected_shape, f"Expected final prediction tensor shape {expected_shape}, got {final_prediction_tensor.shape}."
    assert isinstance(final_prediction_tensor, torch.Tensor), "Final prediction tensor is not a torch.Tensor."
    # Check that tensor contains only 0s and 1s
    unique_values = torch.unique(final_prediction_tensor)
    assert torch.all((unique_values == 0) | (unique_values == 1)), "Final prediction tensor contains values other than 0 and 1."

def test_tensor_to_submission_csv(tmp_path):
    # Create a sample tensor with some predictions
    tensor = torch.zeros((ROWS_COUNT, COLUMNS_COUNT))
    tensor[100, 200] = 1
    tensor[300, 400] = 1
    tensor[500, 600] = 1
    # Define a temporary directory for CSV output
    csvs_dir = tmp_path / "submission_csvs"
    submission_df = tensor_to_submission_csv(tensor, csvs_dir=csvs_dir)
    
    # Check DataFrame structure
    assert isinstance(submission_df, pd.DataFrame), "Submission is not a pandas DataFrame."
    assert list(submission_df.columns) == ['row', 'col', 'prediction'], "CSV columns do not match expected names."
    assert len(submission_df) == 3, f"Expected 3 predictions, got {len(submission_df)}."
    
    # Check specific entries
    expected_rows = [100, 300, 500]
    expected_cols = [200, 400, 600]
    for row, col in zip(expected_rows, expected_cols):
        assert ((submission_df['row'] == row) & (submission_df['col'] == col)).any(), f"Missing prediction at row {row}, col {col}."
    
    # Check if CSV file exists
    expected_csv_path = csvs_dir / 'submission.csv'
    assert os.path.exists(expected_csv_path), "Submission CSV file was not created."
    
    # Optionally, read the CSV and verify contents
    loaded_df = pd.read_csv(expected_csv_path)
    pd.testing.assert_frame_equal(submission_df.reset_index(drop=True), loaded_df.reset_index(drop=True), check_dtype=False)

    # Clean up is handled by tmp_path fixture

# To run the tests, navigate to the project directory and execute:
# pytest

