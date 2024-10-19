# %%
# Import necessary libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon, Point
import pyproj
import os
import ee
import geemap
import sys

# Add the src directory to the sys.path
sys.path.append(os.path.abspath('..'))

# Import functions from your secret_runway_detection package
from secret_runway_detection import train_utils, dataset

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Set random seed for reproducibility
import random
random.seed(train_utils.RANDOM_SEED)

# %%
# Define the path to the landing strips shapefile
landing_strips_shapefile = '../pac_2024_training/pac_2024_training.shp'

# Check if the shapefile exists
if not os.path.exists(landing_strips_shapefile):
    raise FileNotFoundError(f"Shapefile not found at {landing_strips_shapefile}")

# Read the landing strips shapefile
landing_strips = gpd.read_file(landing_strips_shapefile)

# Print the number of landing strips loaded
print(f"Number of landing strips loaded: {len(landing_strips)}")

# Display the first few rows
landing_strips.head()

# %%
# Define the number of tiles per side length for input areas
num_tiles_per_area_side_len = train_utils.TILES_PER_AREA_LEN  # Assuming this is defined in train_utils

# Generate input areas using the provided function
input_areas = train_utils.landing_strips_to_enclosing_input_areas(
    landing_strips=landing_strips,
    num_tiles_per_area_side_len=num_tiles_per_area_side_len
)

# Print the number of input areas generated
print(f"Number of input areas generated: {len(input_areas)}")

# Display the first few input areas
input_areas.head()

# %%
# Select one input area to work with (e.g., the first one)
selected_input_area = input_areas.iloc[0]['geometry']

# Get the CRS of the input area
input_area_crs = input_areas.crs

# Generate the has_strip tensor for the selected input area
has_strip_tensor = train_utils.input_area_to_has_strip_tensor(
    landing_strips=landing_strips,
    input_area=selected_input_area,
    input_area_crs=input_area_crs,
    tiles_per_area_len=num_tiles_per_area_side_len
)

print(f"has_strip_tensor shape: {has_strip_tensor.shape}")

# Generate the satellite image for the selected input area
satellite_image = train_utils.input_area_to_input_image(
    input_area=selected_input_area,
    input_area_crs=input_area_crs
)

print(f"Satellite image shape: {satellite_image.shape}")

# %%
def plot_satellite_with_strips(satellite_image: np.ndarray, input_area: Polygon, landing_strips: gpd.GeoDataFrame, crs: pyproj.crs.CRS):
    """
    Plots the satellite image with landing strips overlaid.

    Parameters:
    - satellite_image (np.ndarray): Satellite image array with shape (C, H, W).
    - input_area (Polygon): The input area polygon.
    - landing_strips (gpd.GeoDataFrame): GeoDataFrame containing landing strip polygons.
    - crs (pyproj.crs.CRS): Coordinate Reference System of the data.
    """
    # Convert satellite_image from (C, H, W) to (H, W, C) for plotting
    satellite_image_rgb = np.transpose(satellite_image, (1, 2, 0))
    
    # Normalize the image for display (assuming values are between 0 and 1)
    satellite_image_rgb = np.clip(satellite_image_rgb, 0, 1)
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Display the satellite image
    plt.imshow(satellite_image_rgb)
    
    # Create a GeoSeries for the input area
    input_area_gs = gpd.GeoSeries([input_area], crs=crs)
    
    # Reproject landing strips to the same CRS as the input area if necessary
    if landing_strips.crs != crs:
        landing_strips = landing_strips.to_crs(crs)
    
    # Filter landing strips that intersect the input area
    landing_strips_in_area = landing_strips[landing_strips.intersects(input_area)]
    
    # Plot the input area boundary
    input_area_gs.boundary.plot(edgecolor='yellow', linewidth=2, ax=plt.gca(), label='Input Area')
    
    # Plot the landing strips
    landing_strips_in_area.plot(ax=plt.gca(), facecolor='none', edgecolor='red', linewidth=2, label='Landing Strips')
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Remove axes
    plt.axis('off')
    
    # Show the plot
    plt.title('Satellite Image with Landing Strips Overlaid')
    plt.show()

# Plot the satellite image with landing strips overlaid
plot_satellite_with_strips(
    satellite_image=satellite_image,
    input_area=selected_input_area,
    landing_strips=landing_strips,
    crs=input_area_crs
)

# %%
def plot_satellite_with_tensor_grid(satellite_image: np.ndarray, has_strip_tensor: torch.Tensor, tiles_per_area_len: int):
    """
    Plots the satellite image with a grid overlay indicating landing strip presence.

    Parameters:
    - satellite_image (np.ndarray): Satellite image array with shape (C, H, W).
    - has_strip_tensor (torch.Tensor): Tensor indicating landing strip presence in each tile.
    - tiles_per_area_len (int): Number of tiles per side.
    """
    # Convert satellite_image from (C, H, W) to (H, W, C) for plotting
    satellite_image_rgb = np.transpose(satellite_image, (1, 2, 0))
    
    # Normalize the image for display (assuming values are between 0 and 1)
    satellite_image_rgb = np.clip(satellite_image_rgb, 0, 1)
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Display the satellite image
    plt.imshow(satellite_image_rgb)
    
    # Overlay the has_strip tensor
    has_strip_np = has_strip_tensor.numpy()
    
    # Create a color map where 0 is transparent and 1 is red with some transparency
    cmap = plt.cm.Reds
    cmap.set_under(color='none')  # Set color for values below the threshold (0)
    
    # Plot the tensor as a semi-transparent overlay
    plt.imshow(
        has_strip_np, 
        cmap=cmap, 
        alpha=0.3, 
        interpolation='nearest',
        extent=[0, satellite_image_rgb.shape[1], satellite_image_rgb.shape[0], 0],
        vmin=0.1  # Ensures that 0 values are transparent
    )
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title('Satellite Image with Landing Strip Grid Overlay')
    
    # Show the plot
    plt.show()

# Plot the satellite image with has_strip tensor grid overlay
plot_satellite_with_tensor_grid(
    satellite_image=satellite_image,
    has_strip_tensor=has_strip_tensor,
    tiles_per_area_len=num_tiles_per_area_side_len
)
