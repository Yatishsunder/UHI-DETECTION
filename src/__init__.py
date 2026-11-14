"""
Urban Heat Island Detection and Spatial Temperature Prediction
Main package for deep learning-based UHI analysis

This package provides tools for:
- Loading and inspecting GeoTIFF satellite data
- Validating dataset integrity
- Preprocessing features for deep learning
- Training and evaluating models
- Generating prediction maps and reports
"""

__version__ = "0.1.0"
__author__ = "University Project"

# Import key modules for convenience
from . import config
from . import data_loader
from . import validation

__all__ = ["config", "data_loader", "validation"]
