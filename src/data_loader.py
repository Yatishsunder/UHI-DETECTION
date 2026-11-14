"""
Data Loader Module: GeoTIFF Reading and Inspection

This module handles reading GeoTIFF files exported from Google Earth Engine
and inspecting their structure, bands, and metadata. Provides functions for
validating raster data and extracting band information.

Key Functions:
    - inspect_geotiff(): Get metadata about a GeoTIFF file
    - load_geotiff_bands(): Read specific bands as numpy arrays
    - load_geotiff_with_metadata(): Load data and metadata together
    - list_geotiffs(): Find all GeoTIFF files in a directory
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not installed. Install with: pip install rasterio")

from . import config

# Configure logging for this module
logger = logging.getLogger(__name__)


def inspect_geotiff(file_path: str) -> Dict[str, Any]:
    """
    Inspect a GeoTIFF file and return its metadata.
    
    Reads the file header without loading full data into memory. Useful for
    understanding file structure before processing.
    
    Args:
        file_path: Path to GeoTIFF file
    
    Returns:
        Dictionary containing:
        - bands: List of band indices (1-indexed)
        - band_count: Number of bands
        - width: Image width in pixels
        - height: Image height in pixels
        - dtype: Data type of bands (e.g., 'float32')
        - crs: Coordinate reference system (e.g., 'EPSG:32644')
        - bounds: Spatial bounds (minx, miny, maxx, maxy)
        - resolution: Pixel size (x_res, y_res) in map units
        - nodata: Nodata value for each band
        - count: Alias for band_count
        - transform: Affine transformation matrix
        - metadata: Full rasterio metadata dictionary
    
    Raises:
        FileNotFoundError: If file not found
        rasterio.errors.RasterioIOError: If not a valid raster file
        ImportError: If rasterio not installed
    
    Examples:
        >>> meta = inspect_geotiff('data/raw/Bhubaneswar/LST_2023.tif')
        >>> print(f"Bands: {meta['band_count']}, Size: {meta['width']}x{meta['height']}")
        >>> print(f"Data type: {meta['dtype']}, CRS: {meta['crs']}")
    """
    
    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for GeoTIFF reading. "
            "Install with: pip install rasterio"
        )
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")
    
    logger.debug(f"Inspecting GeoTIFF: {file_path}")
    
    try:
        with rasterio.open(file_path) as src:
            # Extract metadata
            metadata = {
                'bands': list(range(1, src.count + 1)),
                'band_count': src.count,
                'width': src.width,
                'height': src.height,
                'dtype': str(src.dtypes[0]),  # Assume all bands same dtype
                'crs': str(src.crs),
                'bounds': src.bounds,
                'resolution': (src.res[0], src.res[1]),
                'nodata': src.nodata,
                'transform': src.transform,
                'count': src.count,
                'metadata': src.meta.copy()
            }
            
            logger.debug(
                f"Successfully inspected {file_path}: "
                f"{src.count} bands, {src.width}x{src.height}, {src.dtypes[0]}"
            )
            
            return metadata
    
    except Exception as e:
        logger.error(f"Error inspecting GeoTIFF {file_path}: {e}")
        raise


def load_geotiff_bands(
    file_path: str,
    band_names: Optional[List[str]] = None,
    band_indices: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Load specific bands from a GeoTIFF file as numpy arrays.
    
    Reads raster data and returns as dictionary of numpy arrays, one per band.
    All arrays are converted to float32 for consistency.
    
    Args:
        file_path: Path to GeoTIFF file
        band_names: List of band names to load (e.g., ['LST', 'NDVI']).
                   If provided, assumes rasterio band descriptions match.
                   If not provided, loads all bands.
        band_indices: List of band indices to load (1-indexed).
                     If provided, band_names are ignored.
    
    Returns:
        Dictionary mapping band name/index to numpy array (float32)
        - Keys are band names if available, otherwise f"band_{i}"
        - Values are 2D numpy arrays (height, width) of float32
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: If band_names not found in file
        ImportError: If rasterio not installed
    
    Examples:
        >>> # Load all bands
        >>> data = load_geotiff_bands('data/raw/Bhubaneswar/LST_2023.tif')
        >>> lst = data['band_1']  # Access as dictionary
        >>> print(lst.shape, lst.dtype)
        
        >>> # Load specific bands by index
        >>> data = load_geotiff_bands('file.tif', band_indices=[1, 9, 10])
        >>> print(data.keys())
    """
    
    if not RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for GeoTIFF reading. "
            "Install with: pip install rasterio"
        )
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")
    
    logger.debug(f"Loading bands from GeoTIFF: {file_path}")
    
    try:
        with rasterio.open(file_path) as src:
            # Determine which bands to load
            if band_indices is not None:
                # Use explicit band indices (1-indexed)
                indices = band_indices
            elif band_names is not None:
                # Try to find bands by description
                indices = []
                for desired_name in band_names:
                    found = False
                    for idx in range(1, src.count + 1):
                        band_desc = src.descriptions[idx - 1]
                        if band_desc and band_desc.strip() == desired_name:
                            indices.append(idx)
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Band '{desired_name}' not found in {file_path}. "
                            f"Available bands: {list(src.descriptions)}"
                        )
            else:
                # Load all bands
                indices = list(range(1, src.count + 1))
            
            # Load bands into dictionary
            data = {}
            for idx in indices:
                try:
                    # Read band data
                    band_data = src.read(idx)
                    
                    # Convert to float32 (handles int/uint types)
                    band_data = band_data.astype(np.float32)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        # Replace nodata with NaN
                        band_data[band_data == src.nodata] = np.nan
                    
                    # Use band description as key, fallback to index
                    band_key = src.descriptions[idx - 1]
                    if band_key is None or band_key.strip() == '':
                        band_key = f"band_{idx}"
                    
                    data[band_key] = band_data
                    logger.debug(
                        f"Loaded band {idx} ({band_key}): "
                        f"shape={band_data.shape}, dtype={band_data.dtype}"
                    )
                
                except Exception as e:
                    logger.error(f"Error loading band {idx}: {e}")
                    raise
            
            logger.info(
                f"Successfully loaded {len(data)} bands from {file_path}"
            )
            return data
    
    except Exception as e:
        logger.error(f"Error reading GeoTIFF {file_path}: {e}")
        raise


def load_geotiff_with_metadata(
    file_path: str,
    band_names: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load bands and metadata from a GeoTIFF file together.
    
    Convenience function that loads both raster data and spatial metadata.
    Useful when you need coordinate information along with pixel values.
    
    Args:
        file_path: Path to GeoTIFF file
        band_names: List of band names to load (optional)
    
    Returns:
        Tuple of (data_dict, metadata_dict):
        - data_dict: Dictionary of band_name -> numpy_array
        - metadata_dict: Dictionary with spatial metadata (bounds, CRS, etc.)
    
    Examples:
        >>> data, meta = load_geotiff_with_metadata('file.tif')
        >>> lst = data['LST']
        >>> bounds = meta['bounds']
        >>> crs = meta['crs']
    """
    
    # Load metadata first
    metadata = inspect_geotiff(file_path)
    
    # Load band data
    data = load_geotiff_bands(file_path, band_names=band_names)
    
    return data, metadata


def list_geotiffs(directory: str, pattern: str = "*.tif") -> List[Path]:
    """
    Find all GeoTIFF files in a directory.
    
    Searches recursively for files matching the pattern (default: *.tif).
    Useful for batch processing multiple GeoTIFF files.
    
    Args:
        directory: Path to directory to search
        pattern: File pattern to match (default: *.tif)
               Supports wildcards: *.tif, *_LST.tif, etc.
    
    Returns:
        Sorted list of Path objects matching pattern
    
    Raises:
        FileNotFoundError: If directory not found
    
    Examples:
        >>> files = list_geotiffs('data/raw/Bhubaneswar')
        >>> print(f"Found {len(files)} GeoTIFF files")
        >>> for f in files:
        ...     print(f.name)
    """
    
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    # Find all matching files
    files = sorted(directory.glob(pattern))
    
    logger.debug(
        f"Found {len(files)} GeoTIFF files in {directory} matching '{pattern}'"
    )
    
    return files


def get_band_statistics(band_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a single band array.
    
    Useful for quality checking and understanding data distribution.
    Ignores NaN values in calculations.
    
    Args:
        band_data: 2D numpy array (height, width)
    
    Returns:
        Dictionary with statistics:
        - count: Number of valid (non-NaN) pixels
        - mean: Mean pixel value
        - std: Standard deviation
        - min: Minimum pixel value
        - max: Maximum pixel value
        - nancount: Number of NaN pixels
    
    Examples:
        >>> lst = data['LST']
        >>> stats = get_band_statistics(lst)
        >>> print(f"LST mean: {stats['mean']:.2f} K")
        >>> print(f"Valid pixels: {stats['count']}/{lst.size}")
    """
    
    # Count valid pixels
    valid_mask = ~np.isnan(band_data)
    valid_count = np.sum(valid_mask)
    nan_count = np.sum(~valid_mask)
    
    if valid_count == 0:
        logger.warning("Band contains only NaN values")
        return {
            'count': 0,
            'nancount': nan_count,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    
    # Calculate statistics (ignoring NaN)
    stats = {
        'count': int(valid_count),
        'nancount': int(nan_count),
        'mean': float(np.nanmean(band_data)),
        'std': float(np.nanstd(band_data)),
        'min': float(np.nanmin(band_data)),
        'max': float(np.nanmax(band_data))
    }
    
    return stats


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    """
    When run as a script, demonstrate data loading capabilities.
    Requires a sample GeoTIFF file to test.
    """
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(name)s - %(message)s'
    )
    
    print("=" * 80)
    print("DATA LOADER MODULE - DEMONSTRATION")
    print("=" * 80)
    
    # Try to load configuration
    try:
        config.load_config()
        raw_dir = config.get_data_path('raw_dir')
        print(f"\n✓ Configuration loaded")
        print(f"  Raw data directory: {raw_dir}")
        
        # Look for sample GeoTIFF files
        geotiff_files = list_geotiffs(str(raw_dir))
        
        if geotiff_files:
            print(f"\n✓ Found {len(geotiff_files)} GeoTIFF file(s)")
            
            # Inspect first file
            sample_file = geotiff_files[0]
            print(f"\n  Inspecting: {sample_file.name}")
            
            metadata = inspect_geotiff(str(sample_file))
            print(f"    Bands: {metadata['band_count']}")
            print(f"    Size: {metadata['width']}x{metadata['height']}")
            print(f"    Data type: {metadata['dtype']}")
            print(f"    CRS: {metadata['crs']}")
            print(f"    Bounds: {metadata['bounds']}")
        else:
            print(f"\n⚠ No GeoTIFF files found in {raw_dir}")
            print("  Place GeoTIFF exports from Google Earth Engine here.")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
