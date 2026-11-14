# Agent Instructions: Urban Heat Island Deep Learning Project

## Project Context

This repository contains a deep learning system for **Urban Heat Island (UHI) detection and spatial temperature prediction** across five cities in Odisha, India, using multi-temporal Landsat satellite imagery.

**Key Technologies**:
- Satellite Data: Landsat 8/9 Collection 2 (via Google Earth Engine)
- GIS Processing: rasterio, geopandas
- Deep Learning: CNN, U-Net, ConvLSTM (future phases)
- Python Stack: numpy, pandas, scikit-learn, pytest

---

## Development Principles

### 1. **Reproducibility First**
- ✅ **Use config.yaml** for ALL parameters (paths, thresholds, hyperparameters)
- ❌ **Never hardcode** paths, cities, or processing parameters
- ✅ **Document assumptions** and decisions in comments
- ✅ **Version control** data and models (use DVC if needed)

### 2. **No Fictional Results**
- ✅ Train and evaluate models with actual data before reporting metrics
- ❌ Never claim accuracy without evidence
- ✅ Include standard error bounds and confidence intervals
- ✅ Validate results through multiple evaluation methods

### 3. **Beginner-Friendly Code**
- ✅ Use clear variable names and function signatures
- ✅ Add docstrings to every function (explain inputs, outputs, logic)
- ✅ Comment non-obvious code sections
- ✅ Provide examples in docstrings
- ✅ Avoid obscure Python idioms

### 4. **Testing & Validation**
- ✅ Write unit tests for core functions
- ✅ Validate data before processing (check bands, dtypes, CRS)
- ✅ Generate validation reports
- ✅ Test edge cases (missing data, NaN values)

### 5. **Clear Module Structure**
Each module has a single responsibility:
- `config.py` → Configuration management
- `data_loader.py` → GeoTIFF reading and inspection
- `validation.py` → Data integrity checks
- `preprocessing.py` → Feature normalization and patching (future)
- `models.py` → Model architectures (future)
- `training.py` → Training loops and evaluation (future)

---

## File Organization

### Source Code (`src/`)
```
src/
├── __init__.py              # Package initialization
├── config.py                # Config loader, parameter management
├── data_loader.py           # GeoTIFF I/O, band inspection
├── validation.py            # Dataset validation & reporting
├── preprocessing.py         # Feature engineering, normalization
├── models.py                # CNN, U-Net, ConvLSTM (future)
├── training.py              # Training loops, callbacks (future)
└── utils/
    └── __init__.py          # Utility functions
```

### Tests (`tests/`)
```
tests/
├── __init__.py
├── test_data_loader.py      # Tests for data I/O
├── test_validation.py       # Tests for validation logic
├── test_preprocessing.py    # Tests for preprocessing (future)
└── test_models.py           # Tests for model architectures (future)
```

### Data (`data/`)
```
data/
├── raw/                     # GeoTIFF exports from GEE (DO NOT COMMIT)
├── processed/               # Cleaned datasets
└── splits/                  # Train/val/test splits
```

### Configuration & Docs
```
config.yaml                 # All parameters (paths, cities, thresholds)
docs/                       # Additional documentation
notebooks/                  # Jupyter notebooks for exploration
```

---

## Core Modules: Expected Behavior

### `config.py`
**Purpose**: Load YAML configuration and manage project parameters

**Key Functions**:
```python
load_config(path='config.yaml')
  → Returns dict with keys: data, cities, processing, model, training

get_data_path(key)
  → Returns resolved path (handles relative/absolute paths)

get_cities()
  → Returns list of city dicts with name, coordinates, bounds

validate_config()
  → Raises error if required keys missing or paths invalid
```

**Requirements**:
- ✅ Load from `config.yaml` by default
- ✅ Support environment variable overrides
- ✅ Validate required keys
- ✅ Resolve relative paths
- ✅ Log configuration on load

---

### `data_loader.py`
**Purpose**: Read GeoTIFF files, inspect bands, extract metadata

**Key Functions**:
```python
inspect_geotiff(file_path)
  → Returns dict: {bands, dtype, shape, crs, bounds, nodata_value, metadata}

load_geotiff_bands(file_path, band_names=None)
  → Returns dict: {band_name: numpy_array}

load_geotiff_with_metadata(file_path)
  → Returns tuple: (data_dict, metadata_dict)

list_geotiffs(directory)
  → Returns list of GeoTIFF paths in directory
```

**Requirements**:
- ✅ Use rasterio for reading
- ✅ Handle multiple bands
- ✅ Extract CRS, bounds, nodata values
- ✅ Return Float32 numpy arrays
- ✅ Log warnings for missing/nodata pixels

**Expected Bands**:
`SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, NDVI, NDBI, NDWI, LST`

---

### `validation.py`
**Purpose**: Validate dataset integrity and report issues

**Key Functions**:
```python
validate_geotiff(file_path, expected_bands=None)
  → Returns ValidationReport: {valid, errors, warnings, statistics}

validate_dataset(data_dir, city_name)
  → Returns report for all GeoTIFFs in a city directory

generate_validation_report(report, output_file)
  → Writes human-readable report to text file

check_temporal_sequence(file_list)
  → Validates dates are properly ordered, no gaps
```

**Validation Checks**:
- ✅ All expected bands present
- ✅ All bands are Float32
- ✅ Consistent shape across bands
- ✅ CRS is consistent (e.g., EPSG:32644 for Odisha)
- ✅ Nodata values < 10% per band
- ✅ Statistical bounds (LST in valid range, NDVI in [-1, 1], etc.)

---

### `preprocessing.py` (Future)
**Purpose**: Feature normalization, spatial patching, temporal sequences

**Key Functions**:
```python
normalize_features(data, method='standardization')
  → Returns normalized array with saved scaler

create_spatial_patches(data, patch_size=128, stride=64)
  → Returns list of patches with coordinates

create_temporal_sequences(patch_list, sequence_length=6)
  → Returns sequences without temporal leakage

split_data(data, train_pct=0.7, val_pct=0.15, test_pct=0.15, temporal_aware=True)
  → Returns (train, val, test) with no temporal leakage
```

---

### `models.py` (Future)
**Purpose**: Model architectures (CNN, U-Net, ConvLSTM)

**Models to Implement**:
- CNN for feature extraction (temperature prediction)
- U-Net for UHI segmentation
- ConvLSTM for spatiotemporal sequences

**Requirements**:
- ✅ Clear documentation of architecture
- ✅ Configurable hyperparameters from config.yaml
- ✅ Proper initialization
- ✅ Comments explaining each layer

---

### `training.py` (Future)
**Purpose**: Training loops, evaluation, callbacks

**Key Functions**:
```python
train_model(model, train_data, val_data, config)
  → Returns trained model and history

evaluate_model(model, test_data)
  → Returns metrics: {MAE, RMSE, R², precision, recall, F1, IoU}

generate_predictions(model, data)
  → Returns prediction maps with uncertainty bounds
```

---

## Testing Standards

### Unit Tests (`tests/`)
- ✅ Test every public function
- ✅ Test edge cases (empty data, NaN, single-band)
- ✅ Mock external dependencies (file I/O)
- ✅ Use pytest fixtures for common data
- ✅ Aim for >80% code coverage

### Example Test Pattern
```python
import pytest
from src.data_loader import load_geotiff_bands

def test_load_geotiff_valid_file(sample_geotiff):
    """Test loading a valid GeoTIFF file."""
    data = load_geotiff_bands(sample_geotiff)
    assert data is not None
    assert 'LST' in data
    assert data['LST'].dtype == 'float32'
```

---

## Configuration Guidelines (`config.yaml`)

### Structure
```yaml
data:
  raw_dir: "./data/raw"           # GeoTIFF exports location
  processed_dir: "./data/processed"
  splits_dir: "./data/splits"

cities:
  - name: "Bhubaneswar"
    coordinates: [20.2961, 85.8245]
    bounds: [xmin, ymin, xmax, ymax]
  - name: "Cuttack"
    coordinates: [20.4861, 85.9042]
  # ... etc

processing:
  scale: 30                       # Landsat resolution in meters
  cloud_threshold: 0.1            # Max cloud cover fraction
  valid_pixel_threshold: 0.9      # Min valid pixels required

model:
  temperature_normalization: "standardization"
  patch_size: 128
  sequence_length: 6

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

### Keys Required by Modules
- `data.raw_dir` → data_loader
- `data.processed_dir` → preprocessing
- `cities` → all modules (city selection)
- `processing.*` → validation, preprocessing
- `model.*` → models, training
- `training.*` → training

---

## Common Tasks & Patterns

### Adding a New Function
1. Add to appropriate module (`data_loader.py`, `validation.py`, etc.)
2. Write function with docstring:
   ```python
   def my_function(param1, param2):
       """
       One-line description.
       
       Detailed explanation if needed.
       
       Args:
           param1: Description and type
           param2: Description and type
       
       Returns:
           Description of return value and type
       
       Raises:
           ValueError: If input is invalid
       
       Examples:
           >>> result = my_function('test', 42)
           >>> print(result)
       """
   ```
3. Add unit test in `tests/test_<module>.py`
4. Update this file if changing architecture

### Adding a New City
1. Edit `config.yaml` → add to `cities` list
2. No changes needed in Python code (reads from config)
3. Place GeoTIFFs in `data/raw/<city_name>/`

### Running Tests
```bash
# All tests with coverage
pytest tests/ -v --cov=src

# Single test file
pytest tests/test_data_loader.py -v

# Single test function
pytest tests/test_data_loader.py::test_load_geotiff_valid_file -v
```

---

## Future Development Phases

### Phase 2: Preprocessing
- Implement `preprocessing.py`
- Handle missing data (interpolation, forward-fill)
- Create spatial patches and temporal sequences

### Phase 3: Baseline Models
- Statistical baselines (mean LST, simple regression)
- Baseline evaluation

### Phase 4: Deep Learning
- Implement CNN for temperature prediction
- Implement U-Net for UHI segmentation
- Implement ConvLSTM for sequences

### Phase 5: Evaluation & Visualization
- Calculate metrics
- Generate prediction maps
- Create analysis charts

---

## Key Reminders

✅ **Always check config.yaml first** — never hardcode  
✅ **Comment your code** — assume reader is a beginner  
✅ **Write tests** — validation before modeling  
✅ **No fictional results** — only report actual trained models  
✅ **Log everything** — use logging module for debugging  
✅ **Handle edge cases** — missing data, NaN, edge pixels  
✅ **Document assumptions** — CRS, scale, valid ranges  

---

## Questions?

Refer to:
- `README.md` → Project overview
- `config.yaml` → Parameter reference
- Individual module docstrings → API details
- `tests/` → Usage examples
