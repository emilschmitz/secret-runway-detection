import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import torch
from shapely.geometry import Polygon, Point
import os

import train_utils

# NB
# Our tile indexes go from top to bottom and left to right
# If the competition uses different indexes, we will adjust to it in method tensor_to_submission_csv

# Important TODO: Make sure that image pixels, tiles, and AOI line up as they should
# One tile is approximately 10x10 meters, which is one pixel in Sentinel-2 imagery
# We will try our model on 224x224 pixel images, corresponding to roughly 2240x2240 meters

# One image will be approx. 1500x1500 pixels
TILE_SIDE_LEN = 10.0

# NB: CHECKED WITH ORGANIZERS (this heigth-row and width-column is doesn't seem to make any sense)
# TRANSPOSING FOR NOW
AOI_HEIGHT = 15270.0  # in meters
AOI_WIDTH = 15410.0   # in meters

# ROWS_COUNT = 1541  # counts zero-indexed
# COLUMNS_COUNT = 1527  
ROWS_COUNT = 1527  # counts zero-indexed
COLUMNS_COUNT = 1541

assert (TILE_SIDE_LEN == AOI_HEIGHT / ROWS_COUNT) and (TILE_SIDE_LEN == AOI_WIDTH / COLUMNS_COUNT)

INPUT_IMAGE_HEIGHT = 224  # in pixels
INPUT_IMAGE_WIDTH = 224

# This is more than needed because 7 areas of 224 pixels will cover the whole with Sentinel-2 resolution
# The areas will overlap
INPUT_AREAS_VERTICALLY = 10  
INPUT_AREAS_HORIZONTALLY = 10

TILES_PER_AREA_LEN = 200  # side of one input area there should fit exactly this many tiles

RANDOM_SEED = 42

def point_to_aoi_southeast(point: Point, crs) -> Polygon:
    """
    Creates a rectangular AOI polygon starting from a point (northwest corner),
    extending AOI_WIDTH meters east and AOI_HEIGHT meters south.
    """
    minx = point.x
    maxx = point.x + AOI_WIDTH  # Move east
    maxy = point.y
    miny = point.y - AOI_HEIGHT  # Move south
    return Polygon([
        (minx, maxy),  # Northwest corner
        (maxx, maxy),  # Northeast corner
        (maxx, miny),  # Southeast corner
        (minx, miny),  # Southwest corner
        (minx, maxy)   # Close polygon
    ])

def aoi_to_tiles(aoi: Polygon) -> gpd.GeoDataFrame:
    """
    Divides the AOI into tiles of size METERS_PER_PIXEL x METERS_PER_PIXEL,
    and returns a GeoDataFrame with tile geometries and their row and column indices.
    Function only used for testing purposes.
    """
    tiles = []
    minx, miny, maxx, maxy = aoi.bounds
    x_coords = np.arange(minx, maxx, TILE_SIDE_LEN)
    y_coords = np.arange(miny, maxy, TILE_SIDE_LEN)

    row_idx = 0
    for y in y_coords:
        col_idx = 0
        for x in x_coords:
            tile = Polygon([
                (x, y),
                (x + TILE_SIDE_LEN, y),
                (x + TILE_SIDE_LEN, y + TILE_SIDE_LEN),
                (x, y + TILE_SIDE_LEN),
                (x, y)
            ])
            tiles.append({
                'geometry': tile,
                'row_idx': row_idx,
                'col_idx': col_idx
            })
            col_idx += 1
        row_idx += 1

    tiles_gdf = gpd.GeoDataFrame(tiles, crs='EPSG:32633')  # Replace with appropriate CRS
    return tiles_gdf

def aoi_to_input_areas(aoi: Polygon, crs: pyproj.CRS, num_areas_vertically: int = INPUT_AREAS_VERTICALLY, num_areas_horizontally: int = INPUT_AREAS_HORIZONTALLY) -> gpd.GeoDataFrame:
    """
    Returns both input area geometries and the tile row and column index ranges that cover them.
    """
    input_areas = []
    minx, miny, maxx, maxy = aoi.bounds

    # The input area size in meters
    input_area_width = INPUT_IMAGE_WIDTH * TILE_SIDE_LEN  # 224 * 10 = 2240 meters
    input_area_height = INPUT_IMAGE_HEIGHT * TILE_SIDE_LEN  # 224 * 10 = 2240 meters

    # Calculate the step sizes to cover the AOI with overlapping input areas
    x_total_range = maxx - minx
    y_total_range = maxy - miny

    x_overlap = (num_areas_horizontally * input_area_width - x_total_range) / (num_areas_horizontally - 1)
    y_overlap = (num_areas_vertically * input_area_height - y_total_range) / (num_areas_vertically - 1)

    x_step = input_area_width - x_overlap
    y_step = input_area_height - y_overlap

    for i in range(num_areas_horizontally):
        for j in range(num_areas_vertically):
            left = minx + i * x_step
            bottom = miny + j * y_step
            right = left + input_area_width
            top = bottom + input_area_height

            input_area_polygon = Polygon([
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
                (left, bottom)
            ])

            # Calculate tile indices covering this input area
            leftmost_col_idx = int(round((left - minx) / TILE_SIDE_LEN))
            rightmost_col_idx = int(round((right - minx) / TILE_SIDE_LEN))
            bottom_row_idx = int(round((bottom - miny) / TILE_SIDE_LEN))
            top_row_idx = int(round((top - miny) / TILE_SIDE_LEN))

            input_areas.append({
                'geometry': input_area_polygon,
                'idxs': {
                    'leftmost_col_idx': leftmost_col_idx,
                    'rightmost_col_idx': rightmost_col_idx,
                    'bottom_row_idx': bottom_row_idx,
                    'top_row_idx': top_row_idx
                }
            })

    input_areas_gdf = gpd.GeoDataFrame(input_areas, crs=crs)
    return input_areas_gdf

def pad_output_tensor(output_tensor: torch.Tensor, idxs: dict) -> torch.Tensor:
    """
    Pads the output tensor to the size of the AOI, placing it at the correct location based on indices.
    """
    aoi_sized_tensor = torch.zeros((ROWS_COUNT, COLUMNS_COUNT))
    aoi_sized_tensor[idxs["bottom_row_idx"]:idxs["top_row_idx"],
                     idxs["leftmost_col_idx"]:idxs["rightmost_col_idx"]] = output_tensor.squeeze()
    return aoi_sized_tensor

def run_inference_on_aoi(aoi: Polygon, model: torch.nn.Module, threshold: float) -> torch.Tensor:
    """
    NB we are using max confidence from overlapping areas
    """
    input_areas = aoi_to_input_areas(aoi)

    padded_output_tensors = []
    for _, input_area_row in input_areas.iterrows():
        input_area = input_area_row['geometry']
        idxs = input_area_row['idxs']
        input_image = train_utils.input_area_to_input_image(input_area)
        input_tensor = train_utils.make_input_tensor(input_image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.squeeze(0).squeeze(0)  # Assuming output shape is (1, 1, H, W)

        output_tensor_padded = pad_output_tensor(output_tensor, idxs)
        padded_output_tensors.append(output_tensor_padded)

    # Stack and take the maximum confidence for overlapping areas
    aoi_confidence = torch.stack(padded_output_tensors, dim=0)
    aoi_confidence, _ = torch.max(aoi_confidence, dim=0)

    final_prediction_tensor = (aoi_confidence > threshold).float()

    return final_prediction_tensor

def tensor_to_submission_csv(tensor: torch.Tensor, indexes: str,csvs_dir='submission_csvs') -> pd.DataFrame:
    """
    Converts the final prediction tensor into a submission CSV file.
    """
    if not indexes == 'from-top-left':
        raise NotImplementedError("Only 'from-top-left' indexes are supported at the moment")

    # Flatten the tensor and get the indices of positive predictions
    positive_indices = torch.nonzero(tensor).cpu().numpy()
    rows = positive_indices[:, 0]
    cols = positive_indices[:, 1]

    data = {
        'row': rows,
        'col': cols,
        'prediction': tensor[rows, cols].cpu().numpy()
    }

    submission_df = pd.DataFrame(data)
    # Save to CSV
    os.makedirs(csvs_dir, exist_ok=True)
    submission_csv_path = os.path.join(csvs_dir, 'submission.csv')
    submission_df.to_csv(submission_csv_path, index=False)
    return submission_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    import geopandas as gpd

    # Create a sample point to define the AOI (Adjust coordinates as needed)
    point = Point(500000, 4500000)  # Example coordinates in meters (UTM)

    # Generate the AOI polygon using your function
    aoi_polygon = point_to_aoi_southeast(point)

    # Create a GeoDataFrame for the AOI
    aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_polygon]}, crs='EPSG:32633')  # Replace with your CRS

    # Generate the tiles over the AOI
    tiles_gdf = aoi_to_tiles(aoi_polygon)

    # Generate the input areas (input images)
    input_areas_gdf = aoi_to_input_areas(aoi_polygon)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the AOI boundary
    aoi_gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, label='AOI')

    # Plot the tiles
    tiles_gdf.boundary.plot(ax=ax, edgecolor='blue', linewidth=0.5, alpha=0.5, label='Tiles')

    # Plot the input areas
    input_areas_gdf.boundary.plot(ax=ax, edgecolor='green', linewidth=1, alpha=0.7, label='Input Areas')

    # Set plot title and legend
    ax.set_title('AOI, Tiles, and Input Areas')
    ax.legend()

    # Equal aspect ratio ensures that the scales on x and y axes are equal
    ax.set_aspect('equal')

    # Show the plot
    plt.show()
