import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import torch
from shapely.geometry import Polygon, Point
import os

import secret_runway_detection.train_utils as train_utils

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

import os
import ee
import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import box
from shapely.ops import split
from shapely.geometry import Polygon
from typing import Optional, List, Tuple

def fetch_and_stitch_aoi_quarters(
    aoi_polygon: Polygon,
    aoi_crs: str,
    image_start_date: str,
    image_end_date: str,
    output_dir: str,
    output_filename: str,
    bands: Optional[List[str]] = None,
    scale: int = 10,
    cloud_coverage: int = 20
) -> str:
    """
    Splits the AOI into four quarters, downloads images for each quarter,
    stitches them back together, and saves the final image.

    Parameters:
    - aoi_polygon (Polygon): The AOI polygon.
    - aoi_crs (str): Coordinate reference system of the AOI (e.g., 'EPSG:4326').
    - image_start_date (str): Start date for image collection (YYYY-MM-DD).
    - image_end_date (str): End date for image collection (YYYY-MM-DD).
    - output_dir (str): Directory to save the images and final mosaic.
    - output_filename (str): Filename for the final stitched image (without extension).
    - bands (List[str], optional): List of band names to include. Defaults to ['B4', 'B3', 'B2'].
    - scale (int, optional): Resolution in meters per pixel. Defaults to 10.
    - cloud_coverage (int, optional): Maximum allowed cloud coverage percentage. Defaults to 20.

    Returns:
    - str: Path to the final stitched image.

    Raises:
    - Exception: If the image download or stitching process fails.
    """
    # Initialize Earth Engine
    ee.Initialize()

    if bands is None:
        bands = ['B4', 'B3', 'B2']  # Default to Red, Green, Blue bands

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the AOI into four quarters
    minx, miny, maxx, maxy = aoi_polygon.bounds
    mid_x = (minx + maxx) / 2
    mid_y = (miny + maxy) / 2

    # Define quarters
    quarters = [
        aoi_polygon.intersection(box(minx, mid_y, mid_x, maxy)),  # Top-left
        aoi_polygon.intersection(box(mid_x, mid_y, maxx, maxy)),  # Top-right
        aoi_polygon.intersection(box(minx, miny, mid_x, mid_y)),  # Bottom-left
        aoi_polygon.intersection(box(mid_x, miny, maxx, mid_y)),  # Bottom-right
    ]

    # Function to download image for a given geometry
    def download_image(geometry: Polygon, filename: str) -> Optional[str]:
        # Convert the shapely polygon to GeoJSON format
        geojson = geometry.__geo_interface__
        # Create an Earth Engine geometry
        ee_geom = ee.Geometry(geojson)
        
        # Define the image collection and filter it
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(image_start_date, image_end_date) \
            .filterBounds(ee_geom) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_coverage))

        # Create a median composite
        image = collection.median().clip(ee_geom)

        # Generate the download URL using getDownloadURL
        try:
            url = image.getDownloadURL({
                'name': filename,
                'bands': bands,
                'region': ee_geom,
                'scale': scale,
                'crs': aoi_crs,
                'filePerBand': False,
                'format': 'GEO_TIFF'
            })
        except Exception as e:
            print(f"Error generating download URL for {filename}: {e}")
            return None

        # Download the image
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f'Error fetching image for {filename}')
            return None
        else:
            output_path = os.path.join(output_dir, f'{filename}.tif')
            with open(output_path, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024):
                    fd.write(chunk)
            print(f"Image saved to {output_path}")
            return output_path

    # Download images for each quarter
    image_paths: List[Tuple[int, str]] = []
    for idx, quarter in enumerate(quarters, 1):
        if not quarter.is_empty:
            filename = f'{output_filename}_quarter_{idx}'
            path = download_image(quarter, filename)
            if path:
                image_paths.append((idx, path))
        else:
            print(f"Quarter {idx} is empty and will be skipped.")

    # Read and stitch the images
    src_files_to_mosaic = []
    for idx, path in image_paths:
        src = rasterio.open(path)
        src_files_to_mosaic.append(src)

    if src_files_to_mosaic:
        mosaic, out_trans = merge(src_files_to_mosaic)
        # Write the mosaic to disk
        mosaic_output = os.path.join(output_dir, f'{output_filename}.tif')
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs
        })
        with rasterio.open(mosaic_output, "w", **out_meta) as dest:
            dest.write(mosaic)
        print(f"Mosaic image saved to {mosaic_output}")

        # Close the dataset files
        for src in src_files_to_mosaic:
            src.close()

        return mosaic_output
    else:
        raise Exception("No images were downloaded to create a mosaic.")

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
        input_tensor = train_utils.make_input_image_tensor(input_image)
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

def has_strip_tensors_to_submission_csv(
    has_strip_tensors: dict[str, torch.Tensor],
    indexes: str,
    reorder: bool = True,
    csvs_dir: str = '../submission_csvs',
    sample_submission_path: str = '../SampleSubmission.csv'
) -> pd.DataFrame:
    """
    Converts the final prediction tensors into a submission CSV file.

    Parameters:
    - has_strip_tensors: Dictionary mapping AOI names to prediction tensors.
    - indexes: Indexing method used ('from-top-left' is currently supported).
    - reorder: Whether to reorder the output to match the sample submission.
    - csvs_dir: Directory to save the submission CSV.
    - sample_submission_path: Path to the sample submission CSV.

    Returns:
    - submission_df: A DataFrame containing the submission data.
    """
    if indexes != 'from-top-left':
        raise NotImplementedError("Only 'from-top-left' indexes are supported at the moment")

    # Read the sample submission
    sample_submission = pd.read_csv(sample_submission_path)

    # Initialize a list to store the labels
    labels = []

    def convert_aoi_name(aoi_name_in_submission: str) -> str:
        """
        Converts AOI names from the sample submission format to the tensor format.
        E.g., 'aoi_21_02' -> 'aoi_2021_02'
        """
        parts = aoi_name_in_submission.split('_')
        if len(parts) != 3:
            raise ValueError(f"Unexpected AOI name format: {aoi_name_in_submission}")
        prefix, year_short, idx = parts
        year_full = '20' + year_short  # Convert '21' to '2021'
        return f"{prefix}_{year_full}_{idx}"

    # Process each row in the sample submission
    for _, row in sample_submission.iterrows():
        tile_id = row['tile_row_column']  # E.g., 'Tileaoi_21_02_332_448'

        # Remove 'Tile' prefix and split the identifier
        tile_id_no_prefix = tile_id[4:]
        parts = tile_id_no_prefix.split('_')

        if len(parts) != 5:
            raise ValueError(f"Unexpected tile_id format: {tile_id}")

        # Extract AOI name, row, and column indices
        aoi_name_submission = '_'.join(parts[:3])  # 'aoi_21_02'
        row_idx = int(parts[3])
        col_idx = int(parts[4])

        # Convert AOI name to match the keys in has_strip_tensors
        aoi_name = convert_aoi_name(aoi_name_submission)  # 'aoi_2021_02'

        # Retrieve the tensor for the AOI
        if aoi_name not in has_strip_tensors:
            raise ValueError(f"AOI {aoi_name} not found in has_strip_tensors")

        tensor = has_strip_tensors[aoi_name]

        # Ensure indices are within tensor bounds
        if row_idx >= tensor.shape[0] or col_idx >= tensor.shape[1]:
            label = 0  # Assign 0 if indices are out of bounds
        else:
            label = int(tensor[row_idx, col_idx].item())

        labels.append(label)

    # Add the labels to the DataFrame
    submission_df = sample_submission.copy()
    submission_df['label'] = labels

    # Reorder columns if necessary
    if reorder:
        submission_df = submission_df[['tile_row_column', 'label']]

    # Save to CSV
    os.makedirs(csvs_dir, exist_ok=True)
    submission_csv_path = os.path.join(csvs_dir, 'submission.csv')
    submission_df.to_csv(submission_csv_path, index=False)

    return submission_df

