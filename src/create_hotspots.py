
import rasterio
import numpy as np
import os

cities = {
    "Bhubaneswar": (
        "data/processed/Bhubaneswar_Predicted_LST_2025.tif",
        "data/processed/Bhubaneswar_UHI_Hotspots_2025.tif"
    ),
    "Cuttack": (
        "data/processed/cuttack/Cuttack_Predicted_LST_2025.tif",
        "data/processed/cuttack/Cuttack_UHI_Hotspots_2025.tif"
    ),
    "Berhampur": (
        "data/processed/berhampur/Berhampur_Predicted_LST_2025.tif",
        "data/processed/berhampur/Berhampur_UHI_Hotspots_2025.tif"
    )
}

for city, paths in cities.items():

    input_path, output_path = paths

    print(f"\nProcessing {city}...")

    if not os.path.exists(input_path):
        print("ERROR: Prediction raster not found:", input_path)
        continue

    with rasterio.open(input_path) as src:

        lst = src.read(1)
        profile = src.profile.copy()

    # Identify valid pixels
    valid_mask = np.isfinite(lst)
    valid_values = lst[valid_mask]

    if len(valid_values) == 0:
        print("ERROR: No valid LST pixels.")
        continue

    # Calculate city-specific threshold
    mean_lst = float(np.mean(valid_values))
    std_lst = float(np.std(valid_values))
    threshold = mean_lst + std_lst

    # Create hotspot raster
    hotspot = np.full(lst.shape, 255, dtype=np.uint8)

    hotspot[valid_mask] = (
        lst[valid_mask] > threshold
    ).astype(np.uint8)

    # Calculate statistics
    valid_count = int(np.count_nonzero(valid_mask))
    hotspot_count = int(np.count_nonzero(hotspot == 1))
    hotspot_percentage = (
        hotspot_count / valid_count
    ) * 100

    # Save GeoTIFF
    profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(hotspot, 1)

    print("--------------------------------")
    print("CITY:", city)
    print("Mean LST:", round(mean_lst, 4), "°C")
    print("Standard deviation:", round(std_lst, 4), "°C")
    print("Hotspot threshold:", round(threshold, 4), "°C")
    print("Valid pixels:", valid_count)
    print("Hotspot pixels:", hotspot_count)
    print("Hotspot percentage:", round(hotspot_percentage, 2), "%")
    print("Saved:", output_path)
    print("--------------------------------")

print("\nALL CITY HOTSPOT MAPS PROCESSED.")
