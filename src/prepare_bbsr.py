import numpy as np
import rasterio

raster_path = "data/raw/Bhubaneswar_UHI_2025.tif"

# -----------------------------
# 1. Load raster
# -----------------------------

with rasterio.open(raster_path) as src:
    data = src.read()

# Move bands to last dimension
# (10, 669, 669) -> (669, 669, 10)
data = np.moveaxis(data, 0, -1)

# -----------------------------
# 2. Remove missing pixels
# -----------------------------

missing_mask = np.any(np.isnan(data), axis=2)
valid_mask = ~missing_mask

pixels = data[valid_mask]

# First 9 bands = input features
# Band 10 = LST target

X = pixels[:, :9]
y = pixels[:, 9]

print("Total samples:", len(X))
print("Features:", X.shape)
print("Target:", y.shape)

# -----------------------------
# 3. Train / validation / test
# -----------------------------

np.random.seed(42)

indices = np.random.permutation(len(X))

train_end = int(0.70 * len(X))
val_end = int(0.85 * len(X))

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("\nDataset split:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

# -----------------------------
# 4. Min-Max normalization
# -----------------------------

# Calculate min/max ONLY from training data
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

# Avoid division by zero
range_values = train_max - train_min
range_values[range_values == 0] = 1

# Normalize
X_train_scaled = (X_train - train_min) / range_values
X_val_scaled = (X_val - train_min) / range_values
X_test_scaled = (X_test - train_min) / range_values

print("\nNormalization complete.")

print("\nTraining minimum:")
print(X_train_scaled.min(axis=0))

print("\nTraining maximum:")
print(X_train_scaled.max(axis=0))

# -----------------------------
# 5. Inspect one sample
# -----------------------------

print("\nExample normalized sample:")
print(X_train_scaled[0])

print("\nCorresponding LST:")
print(y_train[0])

# -----------------------------
# 6. Save processed datasets
# -----------------------------

import os

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/X_train.npy", X_train_scaled)
np.save(f"{output_dir}/y_train.npy", y_train)

np.save(f"{output_dir}/X_val.npy", X_val_scaled)
np.save(f"{output_dir}/y_val.npy", y_val)

np.save(f"{output_dir}/X_test.npy", X_test_scaled)
np.save(f"{output_dir}/y_test.npy", y_test)

print("\nProcessed datasets saved successfully.")
print("Location:", output_dir)