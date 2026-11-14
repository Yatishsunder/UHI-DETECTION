import numpy as np
import rasterio

raster_path = "data/raw/Bhubaneswar_UHI_2025.tif"

with rasterio.open(raster_path) as src:

    data = src.read()

# Create mask for valid pixels
missing_mask = np.any(np.isnan(data), axis=0)
valid_mask = ~missing_mask

# Move bands to the last dimension
# From:
# (10, 669, 669)
#
# To:
# (669, 669, 10)

data = np.moveaxis(data, 0, -1)

# Keep only valid pixels
pixels = data[valid_mask]

print("Original raster shape:", data.shape)
print("Valid pixel dataset shape:", pixels.shape)

# Separate inputs and target
X = pixels[:, :9]
y = pixels[:, 9]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst sample:")
print("Features:", X[0])
print("LST:", y[0])