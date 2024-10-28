from io import BytesIO
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import requests
import torch
from shapely.geometry import Polygon, LineString, mapping
from PIL import Image
import ee
import numpy as np
import geopandas as gpd
import random
from scipy.ndimage.morphology import binary_dilation

from secret_runway_detection.dataset import LandingStripDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

TILES_PER_AREA_LEN = 224  # side of one input area there should fit exactly this many tiles

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def landing_strips_to_enclosing_input_areas(
    landing_strips: gpd.GeoDataFrame, 
    num_tiles_per_area_side_len: int, 
    tile_side_len: float = TILE_SIDE_LEN  # in meters
) -> gpd.GeoDataFrame:
    """
    From a list of landing strips (LINESTRINGs), create input areas (POLYGONs), 
    each containing one strip. Areas may overlap; logs info about overlaps.

    Parameters:
    - landing_strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries.
    - num_tiles_per_area_side_len (int): Number of tiles per side length of each area.
    - tile_side_len (float): Side length of a single tile in meters.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing input area polygons.
    """
    # Determine a suitable projected CRS (e.g., UTM zone) based on the landing strips
    # For simplicity, we'll use UTM zone determined by the centroid of the landing strips
    mean_longitude = landing_strips.geometry.unary_union.centroid.x
    utm_zone = int((mean_longitude + 180) / 6) + 1
    if landing_strips.geometry.unary_union.centroid.y >= 0:
        utm_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
    else:
        utm_crs = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere

    # Reproject landing strips to the projected CRS
    landing_strips_proj = landing_strips.to_crs(utm_crs)

    area_side_len = num_tiles_per_area_side_len * tile_side_len  # in meters

    # Initialize an empty GeoDataFrame for input areas
    input_areas = gpd.GeoDataFrame(columns=['geometry'], crs=utm_crs)

    overlap_count = 0  # Counter for overlapping areas

    for idx, strip in landing_strips_proj.iterrows():
        strip_geom = strip.geometry

        # Validate geometry: must be a LineString, valid, not empty, and finite coordinates
        if not isinstance(strip_geom, LineString):
            logger.error(f"Skipping non-LineString geometry at index {idx}.")
            raise

        if not strip_geom.is_valid or strip_geom.is_empty:
            logger.error(f"Skipping invalid or empty LineString at index {idx}.")
            raise

        # Check for finite coordinates
        coords = list(strip_geom.coords)
        if any([not np.isfinite(x) or not np.isfinite(y) for x, y in coords]):
            logger.error(f"Skipping LineString with infinite coordinates at index {idx}.")
            raise

        # Sample a random distance along the LineString's length
        random_distance = random.uniform(0, strip_geom.length)

        # Interpolate the point at the sampled distance
        c = strip_geom.interpolate(random_distance)

        # Generate random offsets for X and Y axes
        offset_x = random.uniform(0, area_side_len)
        offset_y = random.uniform(0, area_side_len)

        # Calculate min and max coordinates for the input area polygon
        minx_area = c.x - (area_side_len - offset_x)
        maxx_area = c.x + offset_x
        miny_area = c.y - (area_side_len - offset_y)
        maxy_area = c.y + offset_y

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
            logger.info(f"Invalid polygon created for LineString at index {idx}. Attempting to repair.")
            area_polygon = area_polygon.buffer(0)
            if not area_polygon.is_valid:
                logger.error(f"Failed to repair polygon for LineString at index {idx}. Skipping.")
                raise

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
            gpd.GeoDataFrame([{'geometry': area_polygon}], crs=utm_crs)
        ], ignore_index=True)

    logger.info(f"Total overlapping areas: {overlap_count}")

    # Reproject input areas back to the original CRS if needed
    input_areas = input_areas.to_crs(landing_strips.crs)

    return input_areas

def input_area_to_has_strip_tensor(
    landing_strips: gpd.GeoDataFrame, 
    input_area: Polygon, 
    input_area_crs: pyproj.CRS, 
    tiles_per_area_len: int = TILES_PER_AREA_LEN,
    num_buffer_tiles: int = 20,
) -> torch.Tensor:
    """
    Converts an input area and its landing strips into a binary tensor indicating the presence of landing strips in each tile.

    Parameters:
    - landing_strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries.
    - input_area (Polygon): Shapely Polygon representing the area of interest.
    - input_area_crs (pyproj.CRS): Coordinate Reference System of the input area.
    - tiles_per_area_len (int): Number of tiles along one side of the area (default is 200).
    - num_buffer_tiles (int): Number of tiles to buffer around the landing strips 
        (default is 20, see https://zindi.africa/competitions/geoai-amazon-basin-secret-runway-detection-challenge/discussions/22422).

    Returns:
    - torch.Tensor: A square tensor of shape (tiles_per_area_len, tiles_per_area_len) with binary values.
    """
    # 1. Validate that the input area is square
    minx, miny, maxx, maxy = input_area.bounds
    width = maxx - minx
    height = maxy - miny

    if not np.isclose(width, height, atol=1e-6):
        logger.warning(f"Input area is not square: width={width}, height={height}, ratio={width/height}")

    # 2. Ensure landing_strips GeoDataFrame is in the same CRS as input_area
    if landing_strips.crs != input_area_crs:
        landing_strips = landing_strips.to_crs(input_area_crs)
        logger.info(f"Inside area-to-tensor method. Landing Strips on different CRS, reprojected to: {input_area_crs}")

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

    # 9. Add buffer region around the tiles with value 1
    has_strip_matrix = add_buffer_to_label(has_strip_matrix, num_buffer_tiles)

    # 10. Convert the NumPy array to a PyTorch tensor of type float
    tensor = torch.tensor(has_strip_matrix, dtype=torch.float32)

    return tensor

def add_buffer_to_label(label: np.ndarray, num_buffer_tiles: int) -> np.ndarray:
    """
    Adds a buffer region around the tiles with value 1 in the label array.
    For everyone, it basically adds a cross with 'arms' of length num_buffer_tiles around the 1s.

    Parameters:
    - label (np.ndarray): 2D array representing the label mask.
    - num_buffer_tiles (int): Number of pixels to buffer around the tiles with value 1.

    Returns:
    - np.ndarray: The label array with the buffer added.
    """
    # Create a structuring element for dilation
    arr = np.concatenate([np.zeros(num_buffer_tiles), np.ones(1), np.zeros(num_buffer_tiles)])
    structure = np.add.outer(arr, arr) > 0
    structure = structure.astype(np.uint8)

    # Apply binary dilation to add buffer
    buffered_label = binary_dilation(label, structure=structure).astype(np.uint8)

    return buffered_label

def get_time_period_of_strips_on_area(strips: gpd.GeoDataFrame, area: Polygon, area_crs: pyproj.CRS) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Returns the time period of the strips that are on the specified area.
    
    Parameters:
    - strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip geometries with a 'yr' attribute (four-digit year).
    - area (Polygon): The area polygon to check for overlapping strips.
    - area_crs (pyproj.CRS): Coordinate Reference System of the area.
    
    Returns:
    - tuple[pd.Timestamp, pd.Timestamp]: Start and end timestamps defining the smallest enclosing period.
    - None: If no strips overlap the area.
    
    Raises:
    - ValueError: If 'yr' attribute is missing.
    - TypeError: If the 'yr' attribute is not numeric.
    """
    # Ensure the strips are in the same CRS as the area
    if strips.crs != area_crs:
        strips = strips.to_crs(area_crs)
    
    # Filter strips that intersect the area
    overlapping_strips = strips[strips.intersects(area)].copy()
    logging.debug(f"Overlapping strips: {overlapping_strips}")
    
    if overlapping_strips.empty:
        logger.warning("No landing strips intersect the specified area.")
        return None
    
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
    input_image_width: int = INPUT_IMAGE_WIDTH,  # Example default value
    input_image_height: int = INPUT_IMAGE_HEIGHT,  # Example default value
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

    logger.debug(f"Inputted area: {input_area}")

    # Reproject input_area to WGS84 (EPSG:4326) for Earth Engine compatibility
    input_area_gdf = gpd.GeoDataFrame({'geometry': [input_area]}, crs=input_area_crs)
    if input_area_gdf.crs.to_string() != 'EPSG:4326':
        input_area_gdf = input_area_gdf.to_crs('EPSG:4326')
        logger.debug("Input area reprojected to EPSG:4326.")
    input_area_wgs84 = input_area_gdf.iloc[0].geometry
    logger.debug(f"Input area after reprojection: {input_area_wgs84}")

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
        logger.warning(f"No images found in the collection for the specified date range {image_data_start_date}--{image_data_end_date} and area.")
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


def make_input_image_tensor(input_image: np.ndarray) -> torch.Tensor:
    """
    Converts the input image NumPy array into a PyTorch tensor suitable for model input.
    """
    # Assume input_image shape is (C, H, W)
    input_tensor = torch.from_numpy(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

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
