import numpy as np
import rasterio

raster_path = "data/raw/Berhampur_UHI_2025.tif"

with rasterio.open(raster_path) as src:

    # Read all 10 bands
    data = src.read()

    print("Data shape:", data.shape)

    # -------------------------
    # Missing-data validation
    # -------------------------

    missing_mask = np.any(np.isnan(data), axis=0)
    valid_mask = ~missing_mask

    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum()
    missing_pixels = missing_mask.sum()

    print("Total pixels:", total_pixels)
    print("Valid pixels:", valid_pixels)
    print("Missing pixels:", missing_pixels)

    print("Valid percentage:", (valid_pixels / total_pixels) * 100)
    print("Missing percentage:", (missing_pixels / total_pixels) * 100)

    # -------------------------
    # Inspect one pixel
    # -------------------------

    row = 300
    col = 300

    print("\nPixel at row", row, "column", col)

    for band_number in range(1, 11):

        value = data[band_number - 1, row, col]

        print(f"Band {band_number:02d}: {value}")