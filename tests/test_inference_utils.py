# tests/test_inference_utils.py

import pandas as pd
import pytest
from shapely.geometry import Polygon, Point
import geopandas as gpd
import torch
import os
from src.inference_utils import (
    point_to_aoi_southeast,
    aoi_to_tiles,
    aoi_to_input_areas,
    pad_output_tensor,
    run_inference_on_aoi,
    tensor_to_submission_csv
)

# Constants (Ensure these match the values in inference_utils.py)
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
TILES_PER_AREA_LEN = 200  # side of one input area there should fit exactly this many tiles
RANDOM_SEED = 42

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
    tiles_gdf = aoi_to_tiles(sample_aoi)
    return tiles_gdf

@pytest.fixture
def sample_input_areas_gdf(sample_aoi):
    """Fixture for a sample input areas GeoDataFrame."""
    input_areas_gdf = aoi_to_input_areas(sample_aoi, crs=None)  # Provide CRS if required
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

def test_point_to_aoi_southeast(sample_aoi):
    """Test the point_to_aoi_southeast function."""
    # Define a point at the northwest corner of the AOI
    point = sample_aoi.exterior.coords[0]
    point = Polygon(sample_aoi.exterior.coords).centroid  # Using centroid for simplicity
    aoi_polygon = point_to_aoi_southeast(point, crs='EPSG:32633')
    
    # Calculate expected AOI size
    expected_aoi_width = AOI_WIDTH
    expected_aoi_height = AOI_HEIGHT
    
    minx = point.x
    maxx = minx + expected_aoi_width
    maxy = point.y
    miny = point.y - expected_aoi_height
    
    expected_polygon = Polygon([
        (minx, maxy),  # Northwest corner
        (maxx, maxy),  # Northeast corner
        (maxx, miny),  # Southeast corner
        (minx, miny),  # Southwest corner
        (minx, maxy)   # Close polygon
    ])
    
    assert aoi_polygon.equals(expected_polygon), "AOI polygon does not match expected coordinates."

def test_aoi_to_tiles(sample_aoi):
    """Test the aoi_to_tiles function."""
    tiles_gdf = aoi_to_tiles(sample_aoi)
    expected_num_tiles = ROWS_COUNT * COLUMNS_COUNT
    assert len(tiles_gdf) == expected_num_tiles, f"Expected {expected_num_tiles} tiles, got {len(tiles_gdf)}."
    
    # Combine tiles into a single geometry and compare with AOI
    tiles_union = tiles_gdf.unary_union
    assert sample_aoi.equals(tiles_union), "Tiles do not correctly cover the AOI."

def test_aoi_to_input_areas(sample_aoi):
    """Test the aoi_to_input_areas function."""
    input_areas_gdf = aoi_to_input_areas(sample_aoi, crs=None)  # Provide CRS if required
    
    # Check the number of input areas
    expected_num_areas = INPUT_AREAS_VERTICALLY * INPUT_AREAS_HORIZONTALLY
    assert len(input_areas_gdf) == expected_num_areas, f"Expected {expected_num_areas} input areas, got {len(input_areas_gdf)}."
    
    # Check if input areas have correct geometries
    for idx, area in input_areas_gdf.iterrows():
        assert isinstance(area.geometry, Polygon), f"Input area at index {idx} is not a Polygon."

def test_pad_output_tensor():
    """Test the pad_output_tensor function."""
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
    """Test the run_inference_on_aoi function."""
    final_prediction_tensor = run_inference_on_aoi(sample_aoi, dummy_model, threshold=0.5)
    
    expected_shape = (ROWS_COUNT, COLUMNS_COUNT)
    assert final_prediction_tensor.shape == expected_shape, f"Expected final prediction tensor shape {expected_shape}, got {final_prediction_tensor.shape}."
    assert isinstance(final_prediction_tensor, torch.Tensor), "Final prediction tensor is not a torch.Tensor."
    
    # Check that tensor contains only 0s and 1s
    unique_values = torch.unique(final_prediction_tensor)
    assert torch.all((unique_values == 0) | (unique_values == 1)), "Final prediction tensor contains values other than 0 and 1."

def test_tensor_to_submission_csv(tmp_path):
    """Test the tensor_to_submission_csv function."""
    # Create a sample tensor with some predictions
    tensor = torch.zeros((ROWS_COUNT, COLUMNS_COUNT))
    tensor[100, 200] = 1
    tensor[300, 400] = 1
    tensor[500, 600] = 1
    
    # Define a temporary directory for CSV output
    csvs_dir = tmp_path / "submission_csvs"
    submission_df = tensor_to_submission_csv(tensor, 'from-top-left', csvs_dir=str(csvs_dir))
    
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
