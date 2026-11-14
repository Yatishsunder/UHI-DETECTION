"""
Unit Tests for data_loader Module

Tests for GeoTIFF reading, band inspection, and metadata extraction.
Uses mock data since actual GeoTIFF files may not be available.

Test Classes:
    - TestConfigLoading: Configuration file loading
    - TestBandStatistics: Band statistics calculation
    - TestListGeotiffs: Finding GeoTIFF files (integration test)
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src import config
from src import data_loader


class TestConfigLoading:
    """Tests for configuration loading and management."""
    
    def test_load_config_success(self):
        """Test that configuration loads without error."""
        # Reset cache to ensure fresh load
        config.reset_cache()
        
        try:
            cfg = config.load_config()
            assert cfg is not None
            assert isinstance(cfg, dict)
            print("✓ Configuration loaded successfully")
        except FileNotFoundError:
            # Expected if config.yaml doesn't exist yet
            pytest.skip("config.yaml not found in current directory")
    
    def test_get_config_cached(self):
        """Test that configuration is cached after first load."""
        config.reset_cache()
        
        try:
            cfg1 = config.load_config()
            cfg2 = config.load_config()
            
            # Should return same object (cached)
            assert cfg1 is cfg2
            print("✓ Configuration caching works")
        except FileNotFoundError:
            pytest.skip("config.yaml not found")
    
    def test_get_cities_from_config(self):
        """Test retrieving cities from configuration."""
        config.reset_cache()
        
        try:
            cities = config.get_cities()
            
            # Should return a list
            assert isinstance(cities, list)
            
            # Each city should have required keys
            for city in cities:
                assert 'name' in city
                assert 'coordinates' in city
                print(f"  ✓ City: {city['name']} at {city['coordinates']}")
            
            if cities:
                print(f"✓ Retrieved {len(cities)} cities from config")
            else:
                print("⚠ No cities found in configuration")
        
        except FileNotFoundError:
            pytest.skip("config.yaml not found")
    
    def test_get_expected_bands(self):
        """Test retrieving expected satellite bands from config."""
        config.reset_cache()
        
        try:
            bands = config.get_expected_bands()
            
            assert isinstance(bands, list)
            assert len(bands) > 0
            assert 'LST' in bands  # LST should always be present
            assert 'NDVI' in bands  # NDVI should be present
            
            print(f"✓ Retrieved {len(bands)} expected bands:")
            for band in bands:
                print(f"  - {band}")
        
        except FileNotFoundError:
            pytest.skip("config.yaml not found")


class TestBandStatistics:
    """Tests for band statistics calculation."""
    
    def test_band_statistics_normal_data(self):
        """Test statistics calculation on normal data."""
        # Create synthetic band data
        band_data = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0]
        ], dtype=np.float32)
        
        stats = data_loader.get_band_statistics(band_data)
        
        assert stats['count'] == 9
        assert stats['nancount'] == 0
        assert stats['mean'] == pytest.approx(50.0)
        assert stats['min'] == 10.0
        assert stats['max'] == 90.0
        
        print("✓ Band statistics calculation correct")
    
    def test_band_statistics_with_nan(self):
        """Test statistics calculation with NaN values."""
        # Create synthetic data with NaN
        band_data = np.array([
            [10.0, 20.0, np.nan],
            [40.0, 50.0, 60.0],
            [np.nan, 80.0, 90.0]
        ], dtype=np.float32)
        
        stats = data_loader.get_band_statistics(band_data)
        
        assert stats['count'] == 7
        assert stats['nancount'] == 2
        assert not np.isnan(stats['mean'])  # Should calculate mean of valid pixels
        assert stats['min'] == 10.0
        assert stats['max'] == 90.0
        
        print("✓ NaN handling in statistics correct")
    
    def test_band_statistics_all_nan(self):
        """Test statistics calculation when all data is NaN."""
        band_data = np.full((3, 3), np.nan, dtype=np.float32)
        
        stats = data_loader.get_band_statistics(band_data)
        
        assert stats['count'] == 0
        assert stats['nancount'] == 9
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['min'])
        
        print("✓ All-NaN band handled correctly")
    
    def test_band_statistics_empty_array(self):
        """Test statistics calculation on empty array."""
        band_data = np.array([], dtype=np.float32).reshape(0, 3)
        
        stats = data_loader.get_band_statistics(band_data)
        
        assert stats['count'] == 0
        assert stats['nancount'] == 0
        
        print("✓ Empty array handled correctly")


class TestListGeotiffs:
    """Tests for finding GeoTIFF files (requires test directory)."""
    
    def test_list_geotiffs_empty_directory(self, tmp_path):
        """Test listing GeoTIFFs in empty directory."""
        # Create empty temp directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        files = data_loader.list_geotiffs(str(empty_dir))
        
        assert files == []
        print("✓ Empty directory handled correctly")
    
    def test_list_geotiffs_with_files(self, tmp_path):
        """Test listing GeoTIFFs when files exist."""
        # Create mock TIF files
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        
        # Create dummy .tif files
        (test_dir / "image1.tif").touch()
        (test_dir / "image2.tif").touch()
        (test_dir / "image3.tif").touch()
        (test_dir / "readme.txt").touch()  # Should not be listed
        
        files = data_loader.list_geotiffs(str(test_dir))
        
        assert len(files) == 3
        assert all(f.suffix == '.tif' for f in files)
        
        print(f"✓ Found {len(files)} GeoTIFF files")
    
    def test_list_geotiffs_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            data_loader.list_geotiffs("/nonexistent/directory")
        
        print("✓ Nonexistent directory raises FileNotFoundError")


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_config_and_cities_integration(self):
        """Test that config and cities work together."""
        config.reset_cache()
        
        try:
            # Load config and get cities
            cities = config.get_cities()
            
            # Try to get a specific city
            if cities:
                first_city = cities[0]
                retrieved = config.get_city_by_name(first_city['name'])
                
                assert retrieved['name'] == first_city['name']
                print(f"✓ Config and cities integration works")
            
        except FileNotFoundError:
            pytest.skip("config.yaml not found")


# ============================================================================
# Pytest Fixtures (reusable test setup)
# ============================================================================

@pytest.fixture
def sample_band_data():
    """Create sample band data for testing."""
    return np.random.rand(100, 100).astype(np.float32) * 100


@pytest.fixture
def sample_band_with_nan():
    """Create sample band data with some NaN values."""
    data = np.random.rand(100, 100).astype(np.float32) * 100
    # Add some NaN values randomly
    mask = np.random.rand(100, 100) < 0.1
    data[mask] = np.nan
    return data


@pytest.fixture
def tmp_geotiff_directory(tmp_path):
    """Create temporary directory with mock GeoTIFF files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create mock city directories
    for city in ["Bhubaneswar", "Cuttack"]:
        city_dir = data_dir / city
        city_dir.mkdir()
        
        # Create mock GeoTIFF files
        (city_dir / f"{city}_2023.tif").touch()
        (city_dir / f"{city}_2024.tif").touch()
    
    return data_dir


# ============================================================================
# Main Test Execution
# ============================================================================

if __name__ == "__main__":
    """
    Run tests with pytest.
    
    Command:
        pytest tests/test_data_loader.py -v
    """
    
    print("=" * 80)
    print("Running Unit Tests for data_loader Module")
    print("=" * 80)
    print()
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
