import rasterio
import numpy as np
import os

# ==========================================
# 1. LOAD BERHAMPUR RASTER
# ==========================================

raster_path = "data/raw/Berhampur_UHI_2025.tif"

with rasterio.open(raster_path) as src:
    data = src.read()

print("Original raster shape:", data.shape)


# ==========================================
# 2. MOVE BANDS TO LAST DIMENSION
# ==========================================

data = np.moveaxis(data, 0, -1)

print("Rearranged shape:", data.shape)


# ==========================================
# 3. FLATTEN PIXELS
# ==========================================

data = data.reshape(-1, 10)

# Remove pixels containing NaN
valid_mask = ~np.isnan(data).any(axis=1)

data = data[valid_mask]

print("Valid pixel dataset shape:", data.shape)


# ==========================================
# 4. SEPARATE INPUTS AND TARGET
# ==========================================

X = data[:, :9]
y = data[:, 9]

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

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


# ==========================================
# 6. NORMALIZATION
# ==========================================

X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

X_train_scaled = (X_train - X_min) / (X_max - X_min)
X_val_scaled = (X_val - X_min) / (X_max - X_min)
X_test_scaled = (X_test - X_min) / (X_max - X_min)

print("\nNormalization complete.")

print("Training minimum:", X_train_scaled.min(axis=0))
print("Training maximum:", X_train_scaled.max(axis=0))


# ==========================================
# 7. SAVE DATASET
# ==========================================

output_dir = "data/processed/berhampur"

os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/X_train.npy", X_train_scaled)
np.save(f"{output_dir}/y_train.npy", y_train)

np.save(f"{output_dir}/X_val.npy", X_val_scaled)
np.save(f"{output_dir}/y_val.npy", y_val)

np.save(f"{output_dir}/X_test.npy", X_test_scaled)
np.save(f"{output_dir}/y_test.npy", y_test)

print("\nBerhampur dataset saved successfully!")
print(output_dir)