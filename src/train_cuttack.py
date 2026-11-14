import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. CHECK GPU
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. LOAD CUTTACK DATASET
# ==========================================

data_dir = "data/processed/cuttack"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")

X_val = np.load(f"{data_dir}/X_val.npy")
y_val = np.load(f"{data_dir}/y_val.npy")

X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

print("\nCuttack Dataset:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 3. CONVERT NUMPY → PYTORCH
# ==========================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# ==========================================
# 4. CREATE DATA LOADERS
# ==========================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2048,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2048,
    shuffle=False
)


# ==========================================
# 5. DEFINE NEURAL NETWORK
# ==========================================

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


model = UHIModel().to(device)

print("\nModel:")
print(model)


# ==========================================
# 6. LOSS + OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. TRAINING
# ==========================================

epochs = 30

print("\nStarting Cuttack training...\n")

for epoch in range(epochs):

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


# ==========================================
# 8. TEST MODEL
# ==========================================

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        output = model(X_batch)

        predictions.append(output.cpu().numpy())
        actual.append(y_batch.numpy())


predictions = np.concatenate(predictions).flatten()
actual = np.concatenate(actual).flatten()


# ==========================================
# 9. METRICS
# ==========================================

mae = np.mean(np.abs(predictions - actual))

mse = np.mean((predictions - actual) ** 2)

rmse = np.sqrt(mse)

ss_res = np.sum((actual - predictions) ** 2)

ss_tot = np.sum((actual - np.mean(actual)) ** 2)

r2 = 1 - (ss_res / ss_tot)


print("\n==============================")
print("CUTTACK FINAL TEST RESULTS")
print("==============================")

print(f"MAE  : {mae:.4f} °C")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} °C")
print(f"R²   : {r2:.4f}")


# ==========================================
# 10. SAVE CUTTACK MODEL
# ==========================================

model_path = f"{data_dir}/uhi_model_cuttack.pth"

torch.save(
    model.state_dict(),
    model_path
)

print("\nCuttack model saved:")
print(model_path)