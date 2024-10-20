from io import BytesIO
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import requests
import shapely
import torch
from shapely.geometry import Polygon, Point
from PIL import Image
import ee
import numpy as np
from shapely.geometry import mapping
import geopandas as gpd
import random

from secret_runway_detection.dataset import LandingStripDataset

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

random.seed(RANDOM_SEED)

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


def landing_strips_to_enclosing_input_areas(
    landing_strips: gpd.GeoDataFrame, 
    num_tiles_per_area_side_len: int, 
) -> gpd.GeoDataFrame:
    """
    From a list of landing strips (LINESTRINGs), create input areas (POLYGONs), 
    each containing one strip. Areas may overlap; logs info about overlaps.

    Parameters:
    - landing_strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries.
    - num_tiles_per_area_side_len (int): Number of tiles per side length of each area.
    - TILE_SIDE_LEN (float): Side length of a single tile in meters.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing input area polygons.
    """
    area_side_len = num_tiles_per_area_side_len * TILE_SIDE_LEN  # in meters

    # Initialize an empty GeoDataFrame for input areas
    input_areas = gpd.GeoDataFrame(columns=['geometry'], crs=landing_strips.crs)

    overlap_count = 0  # Counter for overlapping areas

    for idx, strip in landing_strips.iterrows():
        strip_geom = strip.geometry

        # Validate geometry: must be a LineString, valid, not empty, and finite coordinates
        if not isinstance(strip_geom, shapely.geometry.linestring.LineString):
            print(f"Skipping non-LineString geometry at index {idx}.")
            continue

        if not strip_geom.is_valid or strip_geom.is_empty:
            print(f"Skipping invalid or empty LineString at index {idx}.")
            continue

        # Check for finite coordinates
        coords = list(strip_geom.coords)
        if any([not np.isfinite(x) or not np.isfinite(y) for x, y in coords]):
            print(f"Skipping LineString with infinite coordinates at index {idx}.")
            continue

        # Sample a random distance along the LineString's length
        random_distance = random.uniform(0, strip_geom.length)

        # Interpolate the point at the sampled distance
        rand_point = strip_geom.interpolate(random_distance)

        # Assign the sampled point to 'c' for further processing
        c = rand_point

        # Sample a random number 'n' from 0 to area_side_len
        n = random.uniform(0, area_side_len)

        # Calculate min and max coordinates based on 'n'
        minx_area = c.x - (area_side_len - n)
        maxx_area = c.x + n
        miny_area = c.y - (area_side_len - n)
        maxy_area = c.y + n

        # Create the input area polygon
        area_polygon = Polygon([
            (minx_area, miny_area),
            (maxx_area, miny_area),
            (maxx_area, maxy_area),
            (minx_area, maxy_area),
            (minx_area, miny_area)
        ])

        # Validate the created polygon
        if not area_polygon.is_valid:
            print(f"Invalid polygon created for LineString at index {idx}. Attempting to repair.")
            area_polygon = area_polygon.buffer(0)
            if not area_polygon.is_valid:
                print(f"Failed to repair polygon for LineString at index {idx}. Skipping.")
                continue

        # Check for overlaps with existing input areas
        if not input_areas.empty:
            possible_matches_index = list(input_areas.sindex.intersection(area_polygon.bounds))
            possible_matches = input_areas.iloc[possible_matches_index]
            overlaps = possible_matches.intersects(area_polygon).any()
            if overlaps:
                overlap_count += 1
        else:
            overlaps = False

        # Append the new input area
        input_areas = pd.concat([
            input_areas, 
            gpd.GeoDataFrame([{'geometry': area_polygon}], crs=landing_strips.crs)
        ], ignore_index=True)

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
        logging.info(f"Inside area-to-tensor method. Landing Strips on different CRS, reprojected to: {input_area_crs}")

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

def get_time_period_of_strips_on_area(strips: gpd.GeoDataFrame, area: Polygon, area_crs: pyproj.CRS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns the time period of the strips that are on the specified area.
    
    Parameters:
    - strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries with a 'yr' attribute (four-digit year).
    - area (Polygon): The area polygon to check for overlapping strips.
    - area_crs (pyproj.CRS): Coordinate Reference System of the area.
    
    Returns:
    - tuple[pd.Timestamp, pd.Timestamp]: Start and end timestamps defining the smallest enclosing period.
    
    Raises:
    - ValueError: If no strips overlap the area or 'yr' attribute is missing.
    - TypeError: If the 'yr' attribute is not numeric.
    """
    # Ensure the strips are in the same CRS as the area
    if strips.crs != area_crs:
        strips = strips.to_crs(area_crs)
    
    # Filter strips that intersect the area
    overlapping_strips = strips[strips.intersects(area)].copy()
    logging.debug(f"Overlapping strips: {overlapping_strips}")
    
    if overlapping_strips.empty:
        raise ValueError("No landing strips intersect the specified area.")
    
    # Check if 'yr' attribute exists
    if 'yr' not in overlapping_strips.columns:
        raise ValueError("'yr' attribute not found in the strips GeoDataFrame.")
    
    # Extract 'yr' attribute
    yrs = overlapping_strips['yr']
    
    # Ensure 'yr' is numeric (integer)
    if not pd.api.types.is_numeric_dtype(yrs):
        raise TypeError("'yr' attribute must be numeric (four-digit year).")
    
    # Convert 'yr' to integer if not already
    yrs = yrs.astype(int)
    logging.debug(f"Strip years: {yrs}")
    
    # Find the earliest and latest years
    min_year = yrs.min()
    max_year = yrs.max()
    
    # Create pd.Timestamp objects for the start and end of the period
    start_timestamp = pd.Timestamp(year=min_year, month=1, day=1)
    end_timestamp = pd.Timestamp(year=max_year, month=12, day=31)
    
    return (start_timestamp, end_timestamp)

def input_area_to_input_image(
    input_area: Polygon, 
    image_data_start_date: pd.Timestamp,
    image_data_end_date: pd.Timestamp,
    input_area_crs: pyproj.CRS,
    input_image_width: int = 512,  # Example default value
    input_image_height: int = 512,  # Example default value
) -> np.ndarray:
    """
    Reads the satellite imagery corresponding to the input area and returns it as a NumPy array.
    
    Parameters:
    - input_area (Polygon): The area of interest.
    - image_data_start_date (pd.Timestamp): Start date for the satellite image data.
    - image_data_end_date (pd.Timestamp): End date for the satellite image data.
    - input_area_crs (pyproj.CRS): The CRS of the input area.
    - input_image_width (int): Desired image width in pixels.
    - input_image_height (int): Desired image height in pixels.
    
    Returns:
    - np.ndarray: The satellite image array with shape (3, H, W).
    """
    # Ensure EE is initialized
    try:
        ee.Initialize()
        logger.debug("Earth Engine initialized successfully.")
    except Exception as e:
        logger.info("Earth Engine not initialized. Attempting to authenticate.")
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.debug("Earth Engine authenticated and initialized successfully.")
        except Exception as auth_e:
            logger.error(f"Earth Engine authentication failed: {auth_e}")
            return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Reproject input_area to WGS84 (EPSG:4326) for Earth Engine compatibility
    input_area_gdf = gpd.GeoDataFrame({'geometry': [input_area]}, crs=input_area_crs)
    if input_area_gdf.crs.to_string() != 'EPSG:4326':
        input_area_gdf = input_area_gdf.to_crs('EPSG:4326')
        logger.debug("Input area reprojected to EPSG:4326.")
    input_area_wgs84 = input_area_gdf.iloc[0].geometry

    # Convert the input_area to GeoJSON-like dict
    input_area_geojson = mapping(input_area_wgs84)

    # Convert to EE Geometry
    aoi = ee.Geometry.Polygon(input_area_geojson['coordinates'])
    logger.debug("AOI (Area of Interest) created for Earth Engine.")

    # Load Sentinel-2 image collection
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(image_data_start_date, image_data_end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    try:
        collection_size = collection.size().getInfo()
        logger.debug(f"Found {collection_size} images in the Sentinel-2 collection.")
    except Exception as e:
        logger.error(f"Error retrieving collection size: {e}")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    if collection_size == 0:
        logger.warning("No images found in the collection for the specified date range and area.")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Apply cloud masking function
    def maskS2clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    collection = collection.map(maskS2clouds)
    logger.debug("Cloud masking applied to the image collection.")

    # Create a mean composite
    try:
        image = collection.mean()
        logger.debug("Mean composite image created.")
    except Exception as e:
        logger.warning(f"Error creating mean composite: {e}, falling back to median composite.")
        try:
            image = collection.median()
            logger.debug("Median composite image created.")
        except Exception as median_e:
            logger.error(f"Error creating median composite: {median_e}")
            return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Select bands
    bands = ['B4', 'B3', 'B2']  # RGB bands
    image = image.select(bands)
    logger.debug("RGB bands selected for the composite image.")

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
    logger.debug(f"Thumbnail URL generated: {thumb_url}")

    # Fetch the image from the URL with timeout
    try:
        response = requests.get(thumb_url, timeout=30)  # 30 seconds timeout
        response.raise_for_status()
        logger.debug("Thumbnail image fetched successfully.")
    except requests.exceptions.Timeout:
        logger.error("Request timed out while fetching the thumbnail image.")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching the thumbnail image: {http_err}")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)
    except Exception as e:
        logger.error(f"An error occurred while fetching the thumbnail image: {e}")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Open the image using PIL
    try:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.debug("Thumbnail image opened with PIL.")
    except Exception as e:
        logger.error(f"Error opening the thumbnail image with PIL: {e}")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Convert to NumPy array
    try:
        img_np = np.array(img)
        logger.debug("Thumbnail image converted to NumPy array.")
    except Exception as e:
        logger.error(f"Error converting image to NumPy array: {e}")
        return np.zeros((3, input_image_height, input_image_width), dtype=np.float32)

    # Normalize the image (already divided by 10000, adjust if necessary)
    img_np = img_np.astype(np.float32) / 255.0  # Normalize to [0,1]
    logger.debug("Thumbnail image normalized to [0,1].")

    # Transpose to (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))
    logger.debug(f"Thumbnail image transposed to shape {img_np.shape}.")

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
