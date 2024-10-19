# %% [markdown]
# # Landing Strip Detection Training Pipeline
# 
# This notebook implements a training pipeline for detecting landing strips using satellite imagery. The pipeline includes:
# 
# - Loading input landing strip data.
# - Creating input areas around the landing strips.
# - Downloading Sentinel-2 imagery from Google Earth Engine.
# - Preparing a dataset for training.
# - Loading the Geo Foundation Model (GFM) for transfer learning.
# - Setting up a training loop with Weights & Biases (wandb) logging.
# 
# **Note**: Ensure that you have authenticated with Google Earth Engine (GEE) using `ee.Authenticate()` and have initialized it with `ee.Initialize()`. Also, make sure `train_utils.py` is in your working directory or Python path.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import random
import ee
import wandb
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # PyTorch Image Models library
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

# Import functions and constants from train_utils
from secret_runway_detection.train_utils import (
    landing_strips_to_enclosing_input_areas,
    input_area_to_input_image,
    make_label_tensor,
    TILE_SIDE_LEN,
    TILES_PER_AREA_LEN,
    INPUT_IMAGE_HEIGHT,
    INPUT_IMAGE_WIDTH,
    RANDOM_SEED
)

# %% [markdown]
# ## 2. Configuration and Initialization

# %%
# Debug flag: Set to True to run on CPU, False to use GPU if available
DEBUG = True

# Device configuration
device = torch.device('cpu') if DEBUG else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Initialize wandb
wandb.init(project='landing-strip-detection', name='training-run')

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# %% [markdown]
# ## 3. Load Landing Strips Data

# %%
# Path to the landing strips shapefile
landing_strips_shp = 'pac_2024_training/pac_2024_training.shp'  # Update this path as needed

# Load the landing strips shapefile
landing_strips = gpd.read_file(landing_strips_shp)

# Ensure CRS is WGS84
if landing_strips.crs != 'EPSG:4326':
    landing_strips = landing_strips.to_crs('EPSG:4326')

print(f"Loaded {len(landing_strips)} landing strips.")

# %% [markdown]
# ## 4. Create Input Areas Around Landing Strips

# %%
# Use the function from train_utils to create input areas
num_tiles_per_area_side_len = TILES_PER_AREA_LEN  # From train_utils constants
input_areas = landing_strips_to_enclosing_input_areas(landing_strips, num_tiles_per_area_side_len)

print(f"Created {len(input_areas)} input areas.")

# %% [markdown]
# ## 5. Define the Dataset

# %%
class LandingStripDataset(Dataset):
    def __init__(self, input_areas, landing_strips, transform=None):
        self.input_areas = input_areas
        self.landing_strips = landing_strips
        self.transform = transform
        self.images = []
        self.labels = []
        self.prepare_dataset()

    def prepare_dataset(self):
        for idx, area_row in self.input_areas.iterrows():
            input_area = area_row['geometry']
            # Get input image
            input_image = input_area_to_input_image(input_area)
            if input_image is not None:
                # Create label tensor
                label_tensor = make_label_tensor(input_area, self.landing_strips)
                self.images.append(input_image)
                self.labels.append(label_tensor)
            else:
                print(f"Skipping area at index {idx} due to missing image.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(torch.from_numpy(image).float())
        else:
            image = torch.from_numpy(image).float()
        label = label.float()
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = LandingStripDataset(input_areas, landing_strips, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)} samples")

# %% [markdown]
# ## 6. Load the Geo Foundation Model (GFM)

# %%
def load_gfm_model(model_path):
    """
    Loads the Geo Foundation Model (GFM) from a checkpoint.
    
    Parameters:
    - model_path (str): Path to the model checkpoint.
    
    Returns:
    - model (torch.nn.Module): Loaded model.
    """
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=1  # Assuming binary classification
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract the state dictionary
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Clean the state dictionary (remove 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    
    # Load the state dictionary
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    print("Model loaded and moved to device.")
    return model

# Path to the pre-trained GFM model
model_path = 'simmim_pretrain/gfm.pth'  # Replace with your actual model path

# Load the model
model = load_gfm_model(model_path)

# %% [markdown]
# ## 7. Define Loss Function and Optimizer

# %%
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification with logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optionally, define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %% [markdown]
# ## 8. Training Loop with wandb Logging

# %%
num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # Adjust dimensions if necessary
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        if i % 10 == 9 or i == len(dataloader) - 1:  # Log every 10 batches or last batch
            avg_loss = running_loss / 10
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.4f}")
            wandb.log({'epoch': epoch + 1, 'batch': i + 1, 'loss': avg_loss})
            running_loss = 0.0
    
    # Step the scheduler
    scheduler.step()
    
    # Optionally, log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({'learning_rate': current_lr})
    print(f"Epoch {epoch + 1} completed. Learning Rate: {current_lr}")

print("Training complete.")

# %% [markdown]
# ## 9. Save the Trained Model

# %%
# Save the trained model
model_save_path = 'trained_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'.")

# %% [markdown]
# ## 10. Conclusion

# %%
print("""
# Training Summary

- **Model**: Swin Transformer (GFM) loaded from pre-trained checkpoint.
- **Dataset**: Landing strips with Sentinel-2 imagery.
- **Loss Function**: BCEWithLogitsLoss.
- **Optimizer**: Adam with learning rate scheduler.
- **Logging**: Weights & Biases (wandb) for experiment tracking.
- **Device**: {}
- **Epochs**: {}

Training has been completed and the model has been saved.
""".format(device, num_epochs))

# %%
