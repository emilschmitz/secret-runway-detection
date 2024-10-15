# tests/test_utils.py

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
    return Point(0, 0)  # Starting from origin for simplicity

@pytest.fixture
def sample_aoi(sample_point):
    """Fixture for a sample AOI Polygon using AOI_HEIGHT and AOI_WIDTH."""
    minx = sample_point.x
    miny = sample_point.y
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
    tiles_gdf = aoi_to_tiles(sample_aoi)
    return tiles_gdf

@pytest.fixture
def sample_input_areas_gdf(sample_aoi):
    """Fixture for a sample input areas GeoDataFrame."""
    input_areas_gdf = aoi_to_input_areas(sample_aoi)
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

def test_point_to_aoi_southeast(sample_point):
    aoi_polygon = point_to_aoi_southeast(sample_point)
    expected_coords = [
        (sample_point.x, sample_point.y),  # Southwest corner
        (sample_point.x + AOI_WIDTH, sample_point.y),  # Southeast corner
        (sample_point.x + AOI_WIDTH, sample_point.y - AOI_HEIGHT),  # Northeast corner
        (sample_point.x, sample_point.y - AOI_HEIGHT),  # Northwest corner
        (sample_point.x, sample_point.y)  # Close polygon
    ]
    assert list(aoi_polygon.exterior.coords) == expected_coords, "AOI polygon coordinates mismatch."

def test_aoi_to_tiles(sample_aoi):
    tiles_gdf = aoi_to_tiles(sample_aoi)
    expected_num_tiles = ROWS_COUNT * COLUMNS_COUNT
    assert len(tiles_gdf) == expected_num_tiles, f"Expected {expected_num_tiles} tiles, got {len(tiles_gdf)}."

    # Combine tiles into a single geometry and compare with AOI
    tiles_union = tiles_gdf.unary_union
    assert sample_aoi.equals(tiles_union), "Tiles do not correctly cover the AOI."

def test_input_areas_same_size(sample_aoi):
    input_areas_gdf = aoi_to_input_areas(sample_aoi)
    areas = input_areas_gdf['geometry'].area

    # Check that all input areas have the same area
    assert areas.nunique() == 1, "Input areas have different sizes."

    # Expected area is input_area_width * input_area_height
    expected_area = (INPUT_IMAGE_WIDTH * TILE_SIDE_LEN) * (INPUT_IMAGE_HEIGHT * TILE_SIDE_LEN)
    actual_area = areas.iloc[0]
    assert actual_area == expected_area, f"Input area has unexpected size. Expected {expected_area}, got {actual_area}."

def test_input_area_tiles_union(sample_aoi):
    tiles_gdf = aoi_to_tiles(sample_aoi)
    input_areas_gdf = aoi_to_input_areas(sample_aoi)

    for idx, input_area_row in input_areas_gdf.iterrows():
        input_area_polygon = input_area_row['geometry']

        # Find tiles that intersect with the input area
        tiles_in_input_area = tiles_gdf[tiles_gdf.intersects(input_area_polygon)]

        # Union of tiles
        tiles_union = tiles_in_input_area.unary_union

        # Check that the union of tiles equals the input area
        # Allowing a small tolerance for floating point errors
        assert input_area_polygon.almost_equals(tiles_union, decimal=6), f"Input area at index {idx} is not exactly covered by its tiles."

def test_input_tensor_size():
    num_bands = 3  # Assuming RGB
    input_image = np.zeros((num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=np.float32)
    input_tensor = make_input_tensor(input_image)
    expected_shape = (1, num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert input_tensor.shape == expected_shape, f"Expected input tensor shape {expected_shape}, got {input_tensor.shape}."

def test_output_tensor_size():
    output_tensor = torch.ones((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
    expected_num_entries = INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH
    actual_num_entries = output_tensor.numel()
    assert actual_num_entries == expected_num_entries, f"Expected output tensor to have {expected_num_entries} entries, got {actual_num_entries}."

def test_make_label_tensor(sample_aoi):
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
    label_tensor = make_label_tensor(input_image_polygon, landing_strips)
    expected_shape = (1, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)
    assert label_tensor.shape == expected_shape, f"Expected label tensor shape {expected_shape}, got {label_tensor.shape}."
    assert isinstance(label_tensor, torch.Tensor), "Label tensor is not a torch.Tensor."
    # Check that the label tensor has at least one positive label
    assert torch.sum(label_tensor) >= 1, "Label tensor should contain at least one positive label."

def test_make_training_dataset(sample_aoi):
    input_image_polygon = sample_aoi
    input_image_polygons = gpd.GeoDataFrame({'geometry': [input_image_polygon]}, crs='EPSG:32633')
    # Create a simple landing strip within the AOI
    landing_strip_polygon = Polygon([
        (5000, 5000),
        (7000, 5000),
        (7000, 7000),
        (5000, 7000),
        (5000, 5000)
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
    region = padded_tensor[idxs["bottom_row_idx"]:idxs["top_row_idx"],
                           idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]]
    assert torch.all(region == 1), "Padded region does not match expected values."
    # Check that the rest of the tensor is 0
    outside_region = torch.zeros_like(padded_tensor)
    outside_region[idxs["bottom_row_idx"]:idxs["top_row_idx"],
                   idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]] = 1
    assert torch.all((padded_tensor == 0) | (outside_region == 1)), "Padded tensor contains unexpected non-zero values."

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
    submission_df = tensor_to_submission_csv(tensor, 'from-top-left', csvs_dir=csvs_dir)

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
