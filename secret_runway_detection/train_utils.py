import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import requests
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon, Point
import os
from PIL import Image
import ee
import numpy as np
from shapely.geometry import mapping
import geopandas as gpd
import geemap
from datetime import datetime, timedelta

from secret_runway_detection.dataset import LandingStripDataset

# NB
# Our tile indexes go from top to bottom and left to right
# If the competition uses different indexes, we will adjust to it in method tensor_to_submission_csv

# One image will be approx. 1500x1500 pixels
TILE_SIDE_LEN = 10.0

# ROWS_COUNT = 1541  # counts zero-indexed
# COLUMNS_COUNT = 1527  
ROWS_COUNT = 1527  # counts zero-indexed
COLUMNS_COUNT = 1541

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


def landing_strips_to_enclosing_input_areas(landing_strips: gpd.GeoDataFrame, num_tiles_per_area_side_len: int) -> gpd.GeoDataFrame:
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

    for _, strip in landing_strips.iterrows():
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
        input_areas = pd.concat([input_areas, gpd.GeoDataFrame([{'geometry': area_polygon}], crs=input_areas.crs)], ignore_index=True)

    print(f"Total overlapping areas: {overlap_count}")

    return input_areas

def landing_strips_to_big_area(landing_strips: gpd.GeoDataFrame) -> Polygon:
    """
    From a list of landing strips, create a big area that contains all strips.
    """
    ...

def big_area_to_input_areas(big_area: Polygon, num_tiles_per_area_side_len: int) -> gpd.GeoDataFrame:
    """
    Divides a big area, either arbitrary or an AOI, into smaller input areas.
    """
    ...

import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import torch
from shapely.geometry import Polygon

def input_area_to_has_strip_tensor(
    landing_strips: gpd.GeoDataFrame, 
    input_area: Polygon, 
    input_area_crs: pyproj.crs.CRS, 
    tiles_per_area_len: int = 200
) -> torch.Tensor:
    """
    Converts an input area and its landing strips into a binary tensor indicating the presence of landing strips in each tile.

    Parameters:
    - landing_strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries.
    - input_area (Polygon): Shapely Polygon representing the area of interest.
    - input_area_crs (pyproj.crs.CRS): Coordinate Reference System of the input area.
    - tiles_per_area_len (int): Number of tiles along one side of the area (default is 200).

    Returns:
    - torch.Tensor: A square tensor of shape (tiles_per_area_len, tiles_per_area_len) with binary values.
    """
    # 1. Validate that the input area is square
    minx, miny, maxx, maxy = input_area.bounds
    width = maxx - minx
    height = maxy - miny

    if not np.isclose(width, height, atol=1e-6):
        raise ValueError(f"Input area is not square: width={width}, height={height}")

    # 2. Ensure landing_strips GeoDataFrame is in the same CRS as input_area
    if landing_strips.crs != input_area_crs:
        landing_strips = landing_strips.to_crs(input_area_crs)

    # 3. Filter out landing strips that do not overlap with the input area
    overlapping_strips = landing_strips[landing_strips.intersects(input_area)].copy()

    # If no overlapping strips, return a tensor of zeros
    if overlapping_strips.empty:
        return torch.zeros((tiles_per_area_len, tiles_per_area_len), dtype=torch.float32)

    # 4. Divide the input area into tiles_per_area_len x tiles_per_area_len tiles
    tile_size = width / tiles_per_area_len  # Assuming square tiles

    tiles = []
    for row in range(tiles_per_area_len):
        for col in range(tiles_per_area_len):
            # Calculate the bounds of the current tile
            tile_minx = minx + col * tile_size
            tile_maxx = tile_minx + tile_size
            tile_maxy = maxy - row * tile_size
            tile_miny = tile_maxy - tile_size

            # Create the tile polygon
            tile = Polygon([
                (tile_minx, tile_maxy),  # Northwest
                (tile_maxx, tile_maxy),  # Northeast
                (tile_maxx, tile_miny),  # Southeast
                (tile_minx, tile_miny),  # Southwest
                (tile_minx, tile_maxy)   # Close the polygon
            ])

            tiles.append(tile)

    # 5. Create a GeoDataFrame for all tiles
    tiles_gdf = gpd.GeoDataFrame({'geometry': tiles}, crs=input_area_crs)

    # 6. Spatial join between tiles and overlapping landing strips
    # This will add an index from overlapping_strips to tiles that intersect
    tiles_with_strips = gpd.sjoin(tiles_gdf, overlapping_strips, how='left', predicate='intersects')

    # 7. Determine which tiles have at least one overlapping landing strip
    # If 'index_right' is not NaN, the tile intersects with a landing strip
    # To handle multiple overlaps, group by the tile index and check if any overlap exists
    has_strip = tiles_with_strips.groupby(level=0)['index_right'].apply(lambda x: x.notna().any())

    # 8. Convert the boolean series to a NumPy array and reshape it
    has_strip_matrix = has_strip.values.reshape((tiles_per_area_len, tiles_per_area_len))

    # 9. Convert the NumPy array to a PyTorch tensor of type float
    tensor = torch.tensor(has_strip_matrix, dtype=torch.float32)

    return tensor


def input_area_to_input_image(
    input_area: Polygon, 
    input_area_crs: pyproj.CRS, 
    input_image_width: int = 224, 
    input_image_height: int = 224
) -> np.ndarray:
    """
    Reads the satellite imagery corresponding to the input area and returns it as a NumPy array.
    
    Parameters:
    - input_area (Polygon): The area of interest.
    - input_area_crs (pyproj.CRS): The CRS of the input area.
    - input_image_width (int): Desired image width in pixels.
    - input_image_height (int): Desired image height in pixels.
    
    Returns:
    - np.ndarray: The satellite image array with shape (3, H, W).
    """
    # Ensure EE is initialized
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

    # Reproject input_area to WGS84 (EPSG:4326) for Earth Engine compatibility
    input_area_gdf = gpd.GeoDataFrame({'geometry': [input_area]}, crs=input_area_crs)
    if input_area_gdf.crs.to_string() != 'EPSG:4326':
        input_area_gdf = input_area_gdf.to_crs('EPSG:4326')
    input_area_wgs84 = input_area_gdf.iloc[0].geometry

    # Convert the input_area to GeoJSON-like dict
    input_area_geojson = mapping(input_area_wgs84)

    # Convert to EE Geometry
    aoi = ee.Geometry.Polygon(input_area_geojson['coordinates'])

    # Define the time period (e.g., last 12 months)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Load Sentinel-2 image collection
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR')
        .filterDate(start_date_str, end_date_str)
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )

    # Apply cloud masking function
    def maskS2clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    # Apply cloud masking and scaling to each image
    collection = collection.map(maskS2clouds)

    # Create a mean composite
    try:
        image = collection.mean()
    except Exception as e:
        image = collection.median()
        logging.warning(f"Error creating mean composite: {e}, falling back to median composite")

    # Select bands
    bands = ['B4', 'B3', 'B2']  # RGB bands

    image = image.select(bands)

    # Define visualization parameters
    vis_params = {
        'min': 0,
        'max': 0.3,
        'bands': bands,
    }

    # Get the thumbnail URL
    thumb_url = image.getThumbURL({
        'region': aoi,
        'dimensions': f"{input_image_width}x{input_image_height}",
        'format': 'png',
        'bands': bands,
        'min': vis_params['min'],
        'max': vis_params['max'],
        'gamma': 1.4
    })

    # Fetch the image from the URL
    try:
        response = requests.get(thumb_url)
        if response.status_code != 200:
            print(f"Error fetching image: HTTP {response.status_code}")
            return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)
        
        # Open the image using PIL
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Convert to NumPy array
        img_np = np.array(img)
        
        # Normalize the image (already divided by 10000, adjust if necessary)
        img_np = img_np.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Transpose to (C, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        
    except Exception as e:
        print(f"Error retrieving image data: {e}")
        # Return an array of zeros
        img_np = np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    return img_np


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
