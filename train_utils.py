import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import torch
from shapely.geometry import Polygon, Point
import os

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

def point_to_input_area_southeast(point: Point, crs: pyproj.CRS, num_tiles_per_area_side_len: int) -> Polygon:
    """
    Creates a rectangular input area polygon starting from a point (northwest corner),
    extending south and east based on the number of tiles per side length.

    Parameters:
    - point: Shapely Point object representing the northwest corner.
    - crs: Coordinate Reference System.
    - num_tiles_per_area_side_len: Number of tiles per side length of the area.

    Returns:
    - Polygon representing the input area.
    """
    area_size = num_tiles_per_area_side_len * TILE_SIDE_LEN  # in meters

    minx = point.x
    maxx = point.x + area_size  # Move east
    maxy = point.y
    miny = point.y - area_size  # Move south

    return Polygon([
        (minx, maxy),  # Northwest corner
        (maxx, maxy),  # Northeast corner
        (maxx, miny),  # Southeast corner
        (minx, miny),  # Southwest corner
        (minx, maxy)   # Close polygon
    ])


def landing_strips_to_input_areas(landing_strips: gpd.GeoDataFrame, num_tiles_per_area_side_len: int) -> gpd.GeoDataFrame:
    """
    From a list of landing strips, create input areas, each containing one strip.
    Areas may overlap; logs info about overlaps.

    Parameters:
    - landing_strips: GeoDataFrame containing landing strip geometries.
    - num_tiles_per_area_side_len: Number of tiles per side length of each area.

    Returns:
    - GeoDataFrame containing input area polygons.
    """
    import random
    random.seed(RANDOM_SEED)

    area_size = num_tiles_per_area_side_len * TILE_SIDE_LEN  # in meters

    # Initialize an empty GeoDataFrame for input areas
    input_areas = gpd.GeoDataFrame(columns=['geometry'], crs=landing_strips.crs)

    overlap_count = 0  # Counter for overlapping areas

    for idx, strip in landing_strips.iterrows():
        # Get the geometry of the strip
        strip_geom = strip.geometry

        # Generate random point c within the strip's bounds
        minx, miny, maxx, maxy = strip_geom.bounds
        rand_x = random.uniform(minx, maxx)
        rand_y = random.uniform(miny, maxy)
        rand_point = Point(rand_x, rand_y)

        if not strip_geom.contains(rand_point):
            rand_point = strip_geom.centroid

        c = rand_point

        # Generate random offsets
        offset_x = random.uniform(0, area_size)
        offset_y = random.uniform(0, area_size)

        minx_area = c.x - offset_x
        miny_area = c.y - offset_y
        maxx_area = minx_area + area_size
        maxy_area = miny_area + area_size

        area_polygon = Polygon([
            (minx_area, miny_area),
            (maxx_area, miny_area),
            (maxx_area, maxy_area),
            (minx_area, maxy_area),
            (minx_area, miny_area)
        ])

        # Check if area_polygon overlaps with any existing areas
        if not input_areas.empty:
            possible_matches_index = input_areas.sindex.intersection(area_polygon.bounds)
            possible_matches = input_areas.iloc[list(possible_matches_index)]
            overlaps = possible_matches.intersects(area_polygon).any()
            if overlaps:
                overlap_count += 1
        else:
            overlaps = False

        # Append the area regardless of overlaps
        input_areas = input_areas.append({'geometry': area_polygon}, ignore_index=True)

    print(f"Total overlapping areas: {overlap_count}")

    return input_areas

def input_area_to_has_strip_tensor(landing_strips: gpd.GeoDataFrame, input_area: Polygon, input_area_crs: pyproj.crs, tiles_per_area_len=TILES_PER_AREA_LEN) -> torch.Tensor:
    """
    Outputs a tensor of shape TILES_PER_AREA_LEN x TILES_PER_AREA_LEN indicating whether each tile has a landing strip.
    """
    # We need to check that the landing strips have their own crs
    ...

def input_area_to_input_image(input_area: Polygon, input_area_crs: pyproj.crs) -> np.ndarray:
    """
    Reads the satellite imagery corresponding to the input area and returns it as a NumPy array.
    """
    # Placeholder implementation; in reality, you would read the image data from files or a database
    # For this example, we will generate a dummy image array
    # Assume that the image data is stored in a larger NumPy array representing the entire AOI
    # For the purpose of this function, we will return a dummy array
    num_bands = 3  # For example, RGB bands
    input_image = np.zeros((num_bands, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=np.float32)
    return input_image

def make_input_tensor(input_image: np.ndarray) -> torch.Tensor:
    """
    Converts the input image NumPy array into a PyTorch tensor suitable for model input.
    """
    # Assume input_image shape is (C, H, W)
    input_tensor = torch.from_numpy(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

def make_label_tensor(input_image_polygon: Polygon, landing_strips: gpd.GeoDataFrame) -> torch.Tensor:
    """
    Creates a label tensor for the input image based on the landing strips present.
    """
    # Create a binary mask where landing strips are present
    label = np.zeros((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), dtype=np.uint8)

    # Get the bounds of the input image polygon
    minx, miny, maxx, maxy = input_image_polygon.bounds

    # Pixel size in meters
    pixel_size_x = (maxx - minx) / INPUT_IMAGE_WIDTH
    pixel_size_y = (maxy - miny) / INPUT_IMAGE_HEIGHT

    # Iterate over landing strips
    for _, strip in landing_strips.iterrows():
        if strip.geometry.intersects(input_image_polygon):
            # Get the intersection geometry
            intersection = strip.geometry.intersection(input_image_polygon)
            if not intersection.is_empty:
                # Rasterize the intersection geometry onto the label array
                # Map geometry coordinates to pixel indices
                xs, ys = intersection.exterior.coords.xy
                col_indices = ((np.array(xs) - minx) / pixel_size_x).astype(np.int32)
                row_indices = ((maxy - np.array(ys)) / pixel_size_y).astype(np.int32)
                # Ensure indices are within bounds
                col_indices = np.clip(col_indices, 0, INPUT_IMAGE_WIDTH - 1)
                row_indices = np.clip(row_indices, 0, INPUT_IMAGE_HEIGHT - 1)
                # Set the pixels corresponding to the landing strip to 1
                label[row_indices, col_indices] = 1

    label_tensor = torch.from_numpy(label).unsqueeze(0)  # Add channel dimension
    return label_tensor.float()

class LandingStripDataset(torch.utils.data.Dataset):
    def __init__(self, input_images, landing_strips_idxs):
        ...

def make_train_set(input_areas: gpd.GeoDataFrame, landing_strips: gpd.GeoDataFrame) -> LandingStripDataset:
    """
    Creates a PyTorch Dataset for training the model.
    """
    input_image_tensors = []
    has_strip_tensors = []
    for _, input_area_row in input_areas.iterrows():
        input_area = input_area_row['geometry']
        input_image = input_area_to_input_image(input_area)
        input_image_tensor = torch.tensor(input_image)
        input_image_tensors.append(input_image_tensor)

        has_strip = input_area_to_has_strip_tensor(landing_strips, input_area)
        has_strip_tensors.append(has_strip)

    dataset = LandingStripDataset(input_image_tensors, has_strip_tensors)
    return dataset
