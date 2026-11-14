"""
Dataset Validation Module

This module validates dataset integrity, checks for errors, and generates
validation reports. Ensures data quality before preprocessing and modeling.

Key Functions:
    - validate_geotiff(): Check a single GeoTIFF file for issues
    - validate_dataset(): Validate all GeoTIFFs in a directory
    - generate_validation_report(): Create human-readable report
    - check_band_statistics(): Verify band values are within expected ranges
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

from . import config
from . import data_loader

# Configure logging for this module
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Data class to hold validation results for a single file.
    
    Attributes:
        file_path: Path to the GeoTIFF file
        valid: True if all checks passed
        errors: List of critical errors (data unusable)
        warnings: List of warnings (data usable but has issues)
        band_info: Dictionary with band count, names, dtype
        statistics: Dictionary with band statistics
        timestamp: When validation was performed
    """
    
    file_path: str
    valid: bool
    errors: List[str]
    warnings: List[str]
    band_info: Dict[str, Any]
    statistics: Dict[str, Any]
    timestamp: Optional[str] = None


def validate_geotiff(
    file_path: str,
    expected_bands: Optional[List[str]] = None,
    check_statistics: bool = True
) -> ValidationReport:
    """
    Validate a single GeoTIFF file.
    
    Performs comprehensive checks on file structure, bands, data types,
    and value ranges. Returns detailed validation report.
    
    Args:
        file_path: Path to GeoTIFF file to validate
        expected_bands: List of band names that should be present
                       If None, loads from config.yaml
        check_statistics: Whether to check value ranges for each band
    
    Returns:
        ValidationReport object containing validation results
    
    Raises:
        FileNotFoundError: If file not found
    
    Examples:
        >>> report = validate_geotiff('data/raw/Bhubaneswar/LST_2023.tif')
        >>> if report.valid:
        ...     print("✓ Data is valid")
        ... else:
        ...     print("✗ Validation failed:")
        ...     for error in report.errors:
        ...         print(f"  - {error}")
    """
    
    logger.info(f"Validating GeoTIFF: {file_path}")
    
    file_path_obj = Path(file_path)
    errors = []
    warnings = []
    band_info = {}
    statistics = {}
    
    # Load expected bands from config if not provided
    if expected_bands is None:
        expected_bands = config.get_expected_bands()
    
    try:
        # Step 1: Check file exists
        if not file_path_obj.exists():
            errors.append(f"File not found: {file_path}")
            return ValidationReport(
                file_path=str(file_path_obj),
                valid=False,
                errors=errors,
                warnings=warnings,
                band_info=band_info,
                statistics=statistics
            )
        
        # Step 2: Inspect file metadata
        metadata = data_loader.inspect_geotiff(str(file_path_obj))
        
        band_info = {
            'count': metadata['band_count'],
            'dtype': metadata['dtype'],
            'width': metadata['width'],
            'height': metadata['height'],
            'crs': metadata['crs'],
            'nodata': metadata['nodata']
        }
        
        # Step 3: Check data type
        expected_dtype = config.get_config().get('satellite', {}).get('expected_dtype', 'float32')
        if metadata['dtype'].lower() != expected_dtype.lower():
            warnings.append(
                f"Data type is {metadata['dtype']}, expected {expected_dtype}. "
                f"This may need conversion before modeling."
            )
        
        # Step 4: Check band count
        if metadata['band_count'] != len(expected_bands):
            errors.append(
                f"Band count mismatch: found {metadata['band_count']}, "
                f"expected {len(expected_bands)}"
            )
        
        # Step 5: Load and check bands
        try:
            band_data = data_loader.load_geotiff_bands(str(file_path_obj))
            
            # Check if expected bands are present
            for expected_band in expected_bands:
                if expected_band not in band_data:
                    errors.append(f"Expected band '{expected_band}' not found")
            
            # Step 6: Check band statistics
            if check_statistics:
                valid_ranges = config.get_valid_ranges()
                
                for band_name, band_array in band_data.items():
                    # Calculate statistics
                    band_stats = data_loader.get_band_statistics(band_array)
                    statistics[band_name] = band_stats
                    
                    # Check against valid ranges
                    if band_name in valid_ranges:
                        valid_min, valid_max = valid_ranges[band_name]
                        
                        # Skip NaN values when checking ranges
                        import numpy as np
                        valid_data = band_array[~np.isnan(band_array)]
                        
                        if len(valid_data) > 0:
                            actual_min = float(np.nanmin(band_array))
                            actual_max = float(np.nanmax(band_array))
                            
                            if actual_min < valid_min or actual_max > valid_max:
                                warnings.append(
                                    f"Band '{band_name}' values out of range: "
                                    f"[{actual_min:.2f}, {actual_max:.2f}], "
                                    f"expected [{valid_min}, {valid_max}]"
                                )
                    
                    # Check for excessive missing data
                    if band_stats['nancount'] > 0:
                        total_pixels = band_array.size
                        missing_fraction = band_stats['nancount'] / total_pixels
                        processing_params = config.get_processing_params()
                        valid_pixel_threshold = processing_params.get(
                            'valid_pixel_threshold', 0.9
                        )
                        
                        if missing_fraction > (1.0 - valid_pixel_threshold):
                            errors.append(
                                f"Band '{band_name}' has {missing_fraction:.1%} "
                                f"missing data (exceeds threshold)"
                            )
                        elif missing_fraction > 0:
                            warnings.append(
                                f"Band '{band_name}' has {missing_fraction:.1%} "
                                f"missing data (NaN values)"
                            )
        
        except Exception as e:
            errors.append(f"Error reading band data: {str(e)}")
    
    except Exception as e:
        errors.append(f"Unexpected error during validation: {str(e)}")
    
    # Determine overall validity
    valid = len(errors) == 0
    
    if valid:
        logger.info(f"✓ Validation passed for {file_path_obj.name}")
    else:
        logger.warning(f"✗ Validation failed for {file_path_obj.name}")
    
    return ValidationReport(
        file_path=str(file_path_obj),
        valid=valid,
        errors=errors,
        warnings=warnings,
        band_info=band_info,
        statistics=statistics
    )


def validate_dataset(
    directory: str,
    city_name: Optional[str] = None
) -> List[ValidationReport]:
    """
    Validate all GeoTIFF files in a directory.
    
    Searches directory for .tif files and validates each one.
    Generates a report for the entire dataset.
    
    Args:
        directory: Path to directory containing GeoTIFF files
        city_name: Optional city name for logging/reporting
    
    Returns:
        List of ValidationReport objects (one per file)
    
    Examples:
        >>> reports = validate_dataset('data/raw/Bhubaneswar')
        >>> valid_count = sum(1 for r in reports if r.valid)
        >>> print(f"Valid files: {valid_count}/{len(reports)}")
    """
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    # Find GeoTIFF files
    geotiff_files = data_loader.list_geotiffs(str(directory_path))
    
    if not geotiff_files:
        logger.warning(f"No GeoTIFF files found in {directory}")
        return []
    
    city_info = f" for {city_name}" if city_name else ""
    logger.info(f"Validating {len(geotiff_files)} GeoTIFF files{city_info}")
    
    # Validate each file
    reports = []
    for file_path in geotiff_files:
        report = validate_geotiff(str(file_path))
        reports.append(report)
    
    return reports


def generate_validation_report(
    reports: List[ValidationReport],
    output_file: Optional[str] = None
) -> str:
    """
    Generate a human-readable validation report from a list of reports.
    
    Creates a formatted text report summarizing validation results.
    Optionally saves to file.
    
    Args:
        reports: List of ValidationReport objects
        output_file: Optional path to save report as text file
    
    Returns:
        Formatted report as string
    
    Examples:
        >>> reports = validate_dataset('data/raw/Bhubaneswar')
        >>> report_text = generate_validation_report(reports)
        >>> print(report_text)
        >>> 
        >>> # Save to file
        >>> generate_validation_report(reports, 'validation_report.txt')
    """
    
    if not reports:
        return "No validation reports to generate."
    
    lines = []
    lines.append("=" * 80)
    lines.append("DATASET VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    valid_count = sum(1 for r in reports if r.valid)
    invalid_count = len(reports) - valid_count
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)
    
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total files: {len(reports)}")
    lines.append(f"Valid files: {valid_count}")
    lines.append(f"Invalid files: {invalid_count}")
    lines.append(f"Total errors: {total_errors}")
    lines.append(f"Total warnings: {total_warnings}")
    lines.append("")
    
    # Overall result
    if invalid_count == 0:
        lines.append("✓ ALL FILES VALID - Dataset is ready for processing")
    else:
        lines.append(f"✗ {invalid_count} FILE(S) HAVE ERRORS - Review below")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    # Individual file reports
    for i, report in enumerate(reports, 1):
        file_name = Path(report.file_path).name
        status = "✓ VALID" if report.valid else "✗ INVALID"
        
        lines.append(f"{i}. {file_name} [{status}]")
        lines.append("-" * 80)
        
        # Band info
        if report.band_info:
            lines.append(f"  Bands: {report.band_info.get('count', 'N/A')}")
            lines.append(f"  Size: {report.band_info.get('width', 'N/A')}x{report.band_info.get('height', 'N/A')}")
            lines.append(f"  Data type: {report.band_info.get('dtype', 'N/A')}")
            lines.append(f"  CRS: {report.band_info.get('crs', 'N/A')}")
        
        # Errors
        if report.errors:
            lines.append("")
            lines.append("  ERRORS:")
            for error in report.errors:
                lines.append(f"    ✗ {error}")
        
        # Warnings
        if report.warnings:
            lines.append("")
            lines.append("  WARNINGS:")
            for warning in report.warnings:
                lines.append(f"    ⚠ {warning}")
        
        # Statistics (truncated for readability)
        if report.statistics and len(report.statistics) <= 10:
            lines.append("")
            lines.append("  BAND STATISTICS:")
            for band_name, stats in report.statistics.items():
                lines.append(
                    f"    {band_name}: mean={stats['mean']:.2f}, "
                    f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                    f"NaN={stats['nancount']}"
                )
        
        lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding='utf-8')
        logger.info(f"Validation report saved to {output_file}")
    
    return report_text


# ============================================================================
# Module Initialization
# ============================================================================

if __name__ == "__main__":
    """
    When run as a script, validate all datasets and generate report.
    """
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(name)s - %(message)s'
    )
    
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    
    try:
        # Load configuration
        config.load_config()
        
        # Validate only the explicitly configured study files.
        raw_dir = config.get_data_path('raw_dir')
        expected_files = config.get_config().get('data', {}).get(
            'expected_files', []
        )
        print(f"\nValidating {len(expected_files)} configured study files...\n")

        all_reports = []
        for file_name in expected_files:
            file_path = raw_dir / file_name
            print(f"Validating {file_name}...")
            all_reports.append(validate_geotiff(str(file_path)))
        
        # Generate overall report
        if all_reports:
            report_text = generate_validation_report(all_reports)
            print("\n" + report_text)
            
            # Save the audit outside the raw-data directory.
            report_file = Path('outputs') / 'validation_report.txt'
            generate_validation_report(all_reports, str(report_file))
        else:
            print("\nNo configured GeoTIFF files were found.")
            print("Place the configured exports in: data/raw/")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
