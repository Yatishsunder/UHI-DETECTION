# Deep Learning-Based Urban Heat Island Detection and Spatial Temperature Prediction

## Project Overview

This project builds an **end-to-end deep learning system** for detecting Urban Heat Islands (UHI) and predicting spatial temperature distributions across five major cities in Odisha, India:

- **Bhubaneswar**
- **Cuttack**
- **Rourkela**
- **Sambalpur**
- **Berhampur**

The system uses **multi-temporal satellite imagery** from Landsat 8 and Landsat 9 to extract thermal and spectral features, enabling city-scale temperature prediction and UHI mapping.

---

## Project Objectives

1. **Data Processing**: Retrieve and preprocess satellite imagery with cloud masking and radiometric corrections
2. **Feature Engineering**: Extract spectral indices (NDVI, NDBI, NDWI) and land surface temperature (LST)
3. **Dataset Construction**: Create spatiotemporal sequences with proper train/validation/test split
4. **Baseline Models**: Establish performance baselines with statistical methods
5. **Deep Learning Models**:
   - CNN for feature extraction
   - U-Net for UHI segmentation
   - ConvLSTM for spatiotemporal temperature prediction
6. **Evaluation**: Report MAE, RMSE, R² (regression) and precision, recall, F1, IoU (segmentation)
7. **Visualization**: Generate prediction maps and analysis charts for the final report

---

## Project Structure

```
Odisha_UHI/
├── data/                          # Satellite data & processed datasets
│   ├── raw/                       # GeoTIFF exports from Google Earth Engine
│   ├── processed/                 # Cleaned & normalized datasets
│   └── splits/                    # Train/val/test splits
├── src/                           # Main Python package
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── data_loader.py             # GeoTIFF reading & inspection
│   ├── validation.py              # Dataset validation
│   ├── preprocessing.py           # Normalization, missing value handling
│   ├── models.py                  # Model architectures (future)
│   ├── training.py                # Training loops (future)
│   └── utils/                     # Utility functions
│       └── __init__.py
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_data_loader.py        # Tests for data loading
├── notebooks/                     # Jupyter notebooks for exploration
├── docs/                          # Documentation
├── AGENTS.md                      # Instructions for future development
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration (paths, parameters)
└── .gitignore
```

---

## Installation & Setup

### 1. Clone or Navigate to Project
```bash
cd Odisha_UHI
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Project
Edit `config.yaml` with your data paths and processing parameters.

---

## Key Dependencies

- **rasterio**: GeoTIFF reading and geospatial operations
- **numpy, pandas**: Data processing and analysis
- **scikit-learn**: Preprocessing, scaling, metrics
- **pytest**: Unit testing
- **pyyaml**: Configuration management
- **matplotlib, seaborn**: Visualization

Future additions:
- **TensorFlow/PyTorch**: Deep learning models
- **xarray**: Multidimensional array handling
- **fiona**: Vector data handling

---

## Workflow

### Phase 1: Data Validation (Current)
1. Load GeoTIFF files from Google Earth Engine exports
2. Validate band structure and data types
3. Generate dataset validation report
4. Handle missing/nodata values

### Phase 2: Data Preprocessing
1. Normalize features (standardization/scaling)
2. Create spatial patches (e.g., 128×128 pixels)
3. Generate multi-temporal sequences
4. Split data without temporal leakage

### Phase 3: Baseline Models
1. Statistical baselines (mean, median)
2. Simple linear regression

### Phase 4: Deep Learning
1. CNN for feature extraction
2. U-Net for UHI segmentation
3. ConvLSTM for spatiotemporal prediction

### Phase 5: Evaluation & Visualization
1. Calculate metrics (MAE, RMSE, R², precision, recall, F1, IoU)
2. Generate prediction maps
3. Create analysis charts

---

## Quick Start: Testing the Setup

Once you've installed dependencies, run:

```bash
# Run validation tests
pytest tests/ -v

# Inspect GeoTIFF files (requires sample data)
python -m src.data_loader
```

---

## Configuration

Edit `config.yaml` to specify:
- Data paths (input GeoTIFF, output directories)
- City coordinates
- Processing parameters (scale, cloud mask threshold)
- Model hyperparameters (future)

Example:
```yaml
data:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"

cities:
  - name: "Bhubaneswar"
    coordinates: [20.2961, 85.8245]

processing:
  scale: 30  # meters
  cloud_threshold: 0.1
```

---

## Important Notes

⚠️ **Reproducibility**: All processing is parameterized through `config.yaml`. Never hardcode paths or parameters.

⚠️ **No Fictional Results**: Metrics and predictions are only reported after actual model training and evaluation.

⚠️ **Comments for Beginners**: Code includes detailed comments explaining concepts and decisions.

⚠️ **Testing**: Validation scripts verify data integrity before modeling.

---

## Next Steps

1. Export GeoTIFF files from Google Earth Engine using the provided Earth Engine script
2. Place raw GeoTIFFs in `data/raw/`
3. Run `python -m src.validation` to validate data
4. Proceed with feature engineering and model development

---

## References

- **Landsat 8/9 Collection 2**: [USGS Documentation](https://www.usgs.gov/landsat-missions/landsat-collection-2)
- **Urban Heat Islands**: [EPA Overview](https://www.epa.gov/heatislands)
- **Deep Learning for Remote Sensing**: Recent literature on CNN and U-Net applications

---

## Author & License

**Project**: Deep Learning Urban Heat Island Detection  
**University**: Odisha Institute of Technology  
**Date**: 2026  
**License**: Academic Use

For questions or issues, refer to this README or the `docs/` folder.
