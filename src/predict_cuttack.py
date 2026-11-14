
import rasterio
import numpy as np
import torch
import torch.nn as nn

# 1. FILE PATHS
raster_path = "data/raw/Cuttack_UHI_2025.tif"
model_path = "data/processed/cuttack/uhi_model_cuttack.pth"
output_path = "data/processed/cuttack/Cuttack_Predicted_LST_2025.tif"

# 2. LOAD ORIGINAL RASTER
with rasterio.open(raster_path) as src:
    data = src.read()
    profile = src.profile.copy()
    height = src.height
    width = src.width

print("Original raster:", data.shape)

# 3. MOVE BANDS TO LAST DIMENSION
data_last = np.moveaxis(data, 0, -1)
rows, cols, bands = data_last.shape
pixels = data_last.reshape(-1, 10)

print("Flattened pixels:", pixels.shape)

# 4. FIND VALID PIXELS
valid_mask = np.isfinite(pixels).all(axis=1)
valid_pixels = pixels[valid_mask]

print("Valid pixels:", valid_pixels.shape)

# 5. SEPARATE FEATURES
X = valid_pixels[:, :9]
print("Features:", X.shape)

# 6. RECREATE TRAINING SPLIT
np.random.seed(42)
indices = np.random.permutation(len(X))
train_end = int(0.70 * len(X))
train_idx = indices[:train_end]
X_train = X[train_idx]

print("Training samples:", X_train.shape)

# 7. RECOVER TRAINING NORMALIZATION
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
feature_range = X_max - X_min

if np.any(feature_range == 0):
    raise ValueError("A feature has zero range. Check the training data.")

print("Training minimum:", X_min)
print("Training maximum:", X_max)

# 8. NORMALIZE ALL VALID PIXELS
X_scaled = (X - X_min) / feature_range

# 9. DEFINE MODEL ARCHITECTURE
class UHIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# 10. LOAD TRAINED MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = UHIModel().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.eval()

print("Model loaded successfully.")

# 11. PREDICT ALL VALID PIXELS
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
predictions = []
batch_size = 4096

print("\nPredicting Cuttack LST...")

with torch.no_grad():
    for start in range(0, len(X_tensor), batch_size):
        batch = X_tensor[start:start + batch_size].to(device)
        output = model(batch)
        predictions.append(output.cpu().numpy())

predictions = np.concatenate(predictions).flatten()

print("Predictions:", predictions.shape)
print("Predicted LST minimum:", predictions.min())
print("Predicted LST maximum:", predictions.max())

# 12. RECONSTRUCT PREDICTION RASTER
prediction_raster = np.full(rows * cols, np.nan, dtype=np.float32)
prediction_raster[valid_mask] = predictions
prediction_raster = prediction_raster.reshape(rows, cols)

# 13. SAVE PREDICTED LST GEOTIFF
profile.update(
    count=1,
    dtype="float32",
    nodata=np.nan,
    compress="lzw"
)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(prediction_raster, 1)

print("\n================================")
print("CUTTACK PREDICTION COMPLETE")
print("================================")
print("Output:", output_path)
