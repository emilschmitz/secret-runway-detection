import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from shapely.geometry import Polygon, Point

# Important TODO: Make sure that image pixels, and tiles and aoi line up as they should
# One tile is approximately 10x10 meters, so is one pixel in Sentinel-2 imagery
# We will try our model on 224x224 pixel images, which corresponds to roughly 2240x2240 meters

# One image will be approx. 1500x1500 pixels
METERS_PER_PIXEL = 10

AOI_HEIGHT = 15270.0  # both in meters
AOI_WIDTH = 15410.0

ROWS_COUNT = 1541  # counts zero-indexed
COLUMNS_COUNT = 1526  

INPUT_IMAGE_HEIGHT = 224  # in pixels
INPUT_IMAGE_WIDTH = 224

# this is more than needed because 7 areas of 224 pixels will cover the whole with sentinel 2 resolution
# the areas will overlap
INPUT_AREAS_VERTICALLY = 10  
INPUT_AREAS_HORIZONTALLY = 10

# class Tile:
#     def __init__(self, row: int, column: int, geometry: Polygon):
#         self.row = row
#         self.column = column
#         self.geometry = polygon

def point_to_aoi_southeast(point: Point) -> Polygon:
    ...

def aoi_to_tiles(aoi: Polygon) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with both the tile geometries and the tile row and column indices.
    Function only used for testing purposes.
    """
    ...

def aoi_to_input_areas(aoi: Polygon, tiles: gpd.GeoDataFrame, num_areas_vertically: int, num_areas_horizontally: int) -> gpd.GeoDataFrame:
    """
    Returns both input area geometries and the tile row and column index ranges that cover them.
    """
    # Check that every input area is exactly covered by (always the same) number of tiles
    # Check that the aoi is completely covered by the input areas (they can overlap)
    ...

# def input_area_to_tiles(input_area: Polygon, tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     """
#     Gives the row and column indices of the tiles that cover the input area.
#     """
#     # Check that the input area is exactly covered by the tiles
#     ...

def input_area_to_input_image(input_area: Polygon) -> np.ndarray:
    ...

def make_label_tensor(input_image_polygon: Polygon, landing_strips: gpd.GeoDataFrame) -> torch.Tensor:
    ...

def make_input_tensor(input_image_polygon: Polygon) -> torch.Tensor:
    ...

def make_training_dataset(input_image_polygons: gpd.GeoDataFrame, landing_strips: gpd.GeoDataFrame) -> torch.utils.data.Dataset:
    ...

def pad_output_tensor(output_tensor: torch.Tensor, idxs: dict) -> torch.Tensor:
    aoi_sized_tensor = torch.zeros((ROWS_COUNT, COLUMNS_COUNT))
    aoi_sized_tensor[idxs["top_row_idx"]:idxs["bottom_row_idx"], idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]] = output_tensor
    return aoi_sized_tensor

def run_inference_on_aoi(aoi: Polygon, model: torch.nn.Module, threshold: float) -> torch.Tensor:
    """
    NB we are using max confidence from overlapping areas
    """
    input_areas = aoi_to_input_areas(aoi)

    padded_output_tensors = []
    for input_area in input_areas:
        input_image = input_area_to_input_image(input_area)
        input_tensor = make_input_tensor(input_image)
        output_tensor = model(input_tensor)

        output_tensor_padded = pad_output_tensor(output_tensor, input_area['idxs'])
        padded_output_tensors.append(output_tensor_padded)

    aoi_confidence = torch.max(torch.stack(padded_output_tensors, dim=0), dim=0)

    final_prediction_tensor = aoi_confidence > threshold

    return final_prediction_tensor

def tensor_to_submission_csv(tensor: torch.Tensor, csvs_dir='submission_csvs') -> pd.DataFrame:
    ...