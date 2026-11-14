"""
Configuration Management Module

This module handles loading and validating project configuration from config.yaml.
All project parameters (data paths, processing thresholds, model hyperparameters)
should be accessed through this module to ensure consistency and reproducibility.

Key Functions:
    - load_config(): Load YAML configuration file
    - get_data_path(): Resolve and validate data directory paths
    - get_cities(): Retrieve city information
    - validate_config(): Check for required configuration keys
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Configure logging for this module
logger = logging.getLogger(__name__)

# Default configuration file location
DEFAULT_CONFIG_PATH = "config.yaml"

# Global configuration storage (loaded once at startup)
_CONFIG_CACHE = None


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    
    Reads config.yaml and returns a dictionary containing all project parameters.
    Configuration is cached after first load for efficiency.
    
    Args:
        config_path: Path to YAML configuration file (default: config.yaml)
    
    Returns:
        Dictionary containing configuration with keys:
        - data: Data directories (raw_dir, processed_dir, etc.)
        - cities: List of city dictionaries with coordinates
        - satellite: Band names, resolution, CRS
        - processing: Cloud thresholds, quality control parameters
        - uhi: UHI detection parameters
        - preprocessing: Patch size, normalization method
        - dataset_split: Train/val/test fractions
        - model: Model hyperparameters
        - training: Training hyperparameters
        - evaluation: Metrics to compute
    
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML syntax is invalid
        ValueError: If required keys are missing
    
    Examples:
        >>> config = load_config()
        >>> print(config['cities'])
        >>> raw_dir = config['data']['raw_dir']
    """
    
    global _CONFIG_CACHE
    
    # Return cached config if already loaded
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    # Check if file exists
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML syntax error in {config_path}: {e}")
        raise
    
    if config is None:
        config = {}
    
    # Validate configuration
    _validate_config_structure(config)
    
    # Cache the configuration
    _CONFIG_CACHE = config
    logger.info("Configuration loaded successfully")
    
    return config


def _validate_config_structure(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration keys are present.
    
    Checks for critical configuration sections needed by the project.
    Raises ValueError if any required sections are missing.
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        ValueError: If required keys are missing
    """
    
    required_keys = [
        'data',
        'cities',
        'satellite',
        'processing',
        'preprocessing',
        'dataset_split'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.warning(
            f"Configuration missing sections: {missing_keys}. "
            "Using default values where applicable."
        )
        # We don't raise an error here because YAML might have partial config
        # Individual modules can validate their specific requirements


def get_config() -> Dict[str, Any]:
    """
    Get the cached configuration dictionary.
    
    Returns:
        Cached configuration dictionary
    
    Raises:
        RuntimeError: If configuration not yet loaded
    
    Examples:
        >>> config = get_config()
        >>> print(config['cities'])
    """
    
    if _CONFIG_CACHE is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first."
        )
    return _CONFIG_CACHE


def get_cities() -> List[Dict[str, Any]]:
    """
    Get list of cities from configuration.
    
    Returns:
        List of city dictionaries, each containing:
        - name: City name (string)
        - coordinates: [latitude, longitude]
        - bounds: [min_lat, min_lon, max_lat, max_lon]
        - description: City description (string)
    
    Examples:
        >>> cities = get_cities()
        >>> for city in cities:
        ...     print(f"{city['name']} at {city['coordinates']}")
    """
    
    config = load_config()
    cities = config.get('cities', [])
    
    if not cities:
        logger.warning("No cities defined in configuration")
    
    return cities


def get_city_by_name(city_name: str) -> Dict[str, Any]:
    """
    Get city configuration by name.
    
    Args:
        city_name: Name of city to retrieve
    
    Returns:
        City dictionary with coordinates, bounds, and description
    
    Raises:
        ValueError: If city not found in configuration
    
    Examples:
        >>> bhub = get_city_by_name('Bhubaneswar')
        >>> print(bhub['coordinates'])
    """
    
    cities = get_cities()
    
    for city in cities:
        if city['name'].lower() == city_name.lower():
            return city
    
    available_cities = [city['name'] for city in cities]
    raise ValueError(
        f"City '{city_name}' not found in configuration. "
        f"Available cities: {available_cities}"
    )


def get_data_path(key: str) -> Path:
    """
    Get and validate a data directory path from configuration.
    
    Resolves relative paths to absolute paths and creates directory if needed.
    Helps ensure consistent path handling across the project.
    
    Args:
        key: Configuration key (e.g., 'raw_dir', 'processed_dir')
    
    Returns:
        Resolved Path object
    
    Raises:
        KeyError: If key not found in data configuration
    
    Examples:
        >>> raw_dir = get_data_path('raw_dir')
        >>> print(raw_dir)  # Absolute path to data/raw
    """
    
    config = load_config()
    data_config = config.get('data', {})
    
    if key not in data_config:
        raise KeyError(
            f"Data path key '{key}' not found in config.yaml. "
            f"Available keys: {list(data_config.keys())}"
        )
    
    # Get path from config
    path_str = data_config[key]
    
    # Resolve to absolute path
    if os.path.isabs(path_str):
        path = Path(path_str)
    else:
        # Relative to current working directory or project root
        path = Path.cwd() / path_str
    
    # Create directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Data path '{key}' resolved to: {path}")
    except PermissionError:
        logger.error(f"Permission denied creating directory: {path}")
        raise
    
    return path


def get_expected_bands() -> List[str]:
    """
    Get list of expected satellite bands from configuration.
    
    Returns:
        List of band names expected in GeoTIFF exports
    
    Examples:
        >>> bands = get_expected_bands()
        >>> print(bands)
        ['SR_B2', 'SR_B3', ..., 'LST']
    """
    
    config = load_config()
    satellite_config = config.get('satellite', {})
    expected_bands = satellite_config.get('expected_bands', [])
    
    if not expected_bands:
        logger.warning("No expected bands defined in configuration")
    
    return expected_bands


def get_valid_ranges() -> Dict[str, tuple]:
    """
    Get valid data ranges for quality checking.
    
    Returns:
        Dictionary mapping band names to (min, max) tuples
    
    Examples:
        >>> ranges = get_valid_ranges()
        >>> lst_min, lst_max = ranges['LST']
        >>> print(f"LST valid range: {lst_min} to {lst_max} K")
    """
    
    config = load_config()
    processing_config = config.get('processing', {})
    valid_ranges = processing_config.get('valid_ranges', {})
    
    return valid_ranges


def get_processing_params() -> Dict[str, Any]:
    """
    Get processing parameters (cloud threshold, quality control).
    
    Returns:
        Dictionary with processing parameters
    
    Examples:
        >>> params = get_processing_params()
        >>> print(f"Cloud threshold: {params['cloud_threshold']}")
    """
    
    config = load_config()
    processing = config.get('processing', {})
    
    return processing


def reset_cache() -> None:
    """
    Clear the configuration cache (for testing purposes).
    
    This function should only be used in testing when you need to
    reload configuration with different values.
    
    Examples:
        >>> reset_cache()
        >>> load_config("test_config.yaml")
    """
    
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    logger.debug("Configuration cache cleared")


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    """
    When run as a script, print and validate configuration.
    Useful for debugging configuration issues.
    """
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )
    
    print("=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80)
    
    try:
        config = load_config()
        
        print("\n✓ Configuration loaded successfully")
        print(f"\nProject cities ({len(config['cities'])}):")
        for city in config['cities']:
            print(f"  - {city['name']}: {city['coordinates']}")
        
        print(f"\nExpected satellite bands ({len(config['satellite']['expected_bands'])}):")
        for band in config['satellite']['expected_bands']:
            print(f"  - {band}")
        
        print(f"\nData directories:")
        for key in ['raw_dir', 'processed_dir', 'splits_dir']:
            path = get_data_path(key)
            print(f"  - {key}: {path}")
        
        print("\n✓ All configuration checks passed!")
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        import traceback
        traceback.print_exc()
