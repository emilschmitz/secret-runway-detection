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

def landing_strips_to_input_areas(landing_strips: gpd.GeoDataFrame, num_areas: int) -> gpd.GeoDataFrame:
    """
    From a list of landing strips, create input areas, each containing one or more strips.
    Areas will not overlap.

    Parameters:
    - landing_strips: GeoDataFrame containing landing strip geometries.
    - num_areas: The number of input areas to create.

    Returns:
    - GeoDataFrame containing input area polygons.
    """
    from sklearn.cluster import KMeans

    # Ensure there are landing strips to process
    if landing_strips.empty:
        raise ValueError("The landing_strips GeoDataFrame is empty.")

    # Calculate centroids of landing strips for clustering
    landing_strips['centroid'] = landing_strips.geometry.centroid

    # Extract coordinates for clustering
    coords = np.array([[geom.x, geom.y] for geom in landing_strips['centroid']])

    # Handle cases where num_areas is greater than the number of landing strips
    num_areas = min(num_areas, len(landing_strips))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_areas, random_state=42)
    landing_strips['cluster'] = kmeans.fit_predict(coords)

    # Create input area polygons by aggregating strips in each cluster
    input_areas = []
    for cluster_id in range(num_areas):
        cluster_strips = landing_strips[landing_strips['cluster'] == cluster_id]
        # Union of all strips in the cluster
        union_geom = cluster_strips.unary_union
        # Create a convex hull around the unioned geometry
        convex_hull = union_geom.convex_hull
        input_areas.append({
            'cluster_id': cluster_id,
            'geometry': convex_hull
        })

    # Create GeoDataFrame for input areas
    input_areas_gdf = gpd.GeoDataFrame(input_areas, crs=landing_strips.crs)

    # Clean up temporary columns
    landing_strips.drop(columns=['centroid', 'cluster'], inplace=True)

    return input_areas_gdf


def point_to_input_area_southeast(point: Point, crs: pyproj.crs) -> Polygon:
    ...

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
        input_image = input_area_to_input_image(input_area)
        input_tensor = make_input_tensor(input_image)
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
