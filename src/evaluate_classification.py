
import os
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# ANN ARCHITECTURE
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


# ==========================================
# CITY FILE PATHS
# ==========================================

cities = {
    "Bhubaneswar": {
        "folder": "data/processed",
        "model": "data/processed/uhi_model.pth"
    },
    "Cuttack": {
        "folder": "data/processed/cuttack",
        "model": "data/processed/cuttack/uhi_model_cuttack.pth"
    },
    "Berhampur": {
        "folder": "data/processed/berhampur",
        "model": "data/processed/berhampur/uhi_model_berhampur.pth"
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_city(city, paths):

    folder = paths["folder"]
    model_path = paths["model"]

    print("\n" + "=" * 55)
    print("CITY:", city)
    print("=" * 55)

    # Load processed datasets
    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy")).reshape(-1)

    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy")).reshape(-1)

    # Check dimensions
    if X_train.shape[1] != 9 or X_test.shape[1] != 9:
        raise ValueError("Expected 9 input features.")

    # Threshold based only on training targets
    threshold = float(np.mean(y_train) + np.std(y_train))

    print("Hotspot threshold:", round(threshold, 4), "°C")

    # Load trained ANN
    model = UHIModel().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Predict test LST
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), 4096):
            batch = X_tensor[start:start + 4096].to(device)
            output = model(batch)
            predictions.append(output.cpu().numpy())

    y_pred = np.concatenate(predictions).reshape(-1)

    # Convert actual and predicted LST to binary classes
    actual_class = (y_test > threshold).astype(int)
    predicted_class = (y_pred > threshold).astype(int)

    # Confusion matrix
    tn = int(np.sum((actual_class == 0) & (predicted_class == 0)))
    fp = int(np.sum((actual_class == 0) & (predicted_class == 1)))
    fn = int(np.sum((actual_class == 1) & (predicted_class == 0)))
    tp = int(np.sum((actual_class == 1) & (predicted_class == 1)))

    confusion = np.array([
        [tn, fp],
        [fn, tp]
    ])

    # Metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0

    results = {}

    for class_id, class_name in [(0, "Non-hotspot"), (1, "Hotspot")]:

        if class_id == 0:
            TP = tn
            FP = fn
            FN = fp
            support = tn + fp
        else:
            TP = tp
            FP = fp
            FN = fn
            support = tp + fn

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        results[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

    # Display confusion matrix
    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("                 Non-hotspot   Hotspot")
    print(f"Actual Non-hotspot   {tn:10d} {fp:10d}")
    print(f"Actual Hotspot      {fn:10d} {tp:10d}")

    # Display classification report
    print("\nCLASSIFICATION REPORT")
    print(f"{'Class':<18}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")

    for name, values in results.items():
        print(
            f"{name:<18}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1']:>12.4f}"
            f"{values['support']:>12d}"
        )

    macro_precision = np.mean([v["precision"] for v in results.values()])
    macro_recall = np.mean([v["recall"] for v in results.values()])
    macro_f1 = np.mean([v["f1"] for v in results.values()])

    print(
        f"\n{'Accuracy:':<18}{accuracy:.4f}"
    )
    print(
        f"{'Macro average:':<18}"
        f"{macro_precision:.4f} precision, "
        f"{macro_recall:.4f} recall, "
        f"{macro_f1:.4f} F1"
    )

    # Save results
    output_folder = "data/processed/classification_results"
    os.makedirs(output_folder, exist_ok=True)

    np.savetxt(
        os.path.join(output_folder, f"{city}_confusion_matrix.csv"),
        confusion,
        delimiter=",",
        fmt="%d"
    )

    with open(
        os.path.join(output_folder, f"{city}_classification_report.txt"),
        "w"
    ) as file:

        file.write(f"City: {city}\n")
        file.write(f"Threshold: {threshold:.4f} °C\n")
        file.write("Confusion Matrix:\n")
        file.write(str(confusion))
        file.write("\n\nClassification Report:\n")

        for name, values in results.items():
            file.write(
                f"{name}: Precision={values['precision']:.4f}, "
                f"Recall={values['recall']:.4f}, "
                f"F1={values['f1']:.4f}, "
                f"Support={values['support']}\n"
            )

        file.write(f"\nAccuracy: {accuracy:.4f}\n")

    print("\nResults saved for", city)


# ==========================================
# RUN ALL THREE CITIES
# ==========================================

for city, paths in cities.items():
    evaluate_city(city, paths)

print("\nALL CLASSIFICATION EVALUATIONS COMPLETED.")
