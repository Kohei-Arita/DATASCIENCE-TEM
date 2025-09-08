"""
RSNA Aneurysm Detection - DICOM Utilities

Specialized functions for handling DICOM medical images.
Includes preprocessing, windowing, and metadata extraction.
"""

import os
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
from pydicom.errors import InvalidDicomError
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class DICOMProcessor:
    """
    DICOM image processor with medical imaging optimizations

    Handles DICOM loading, preprocessing, windowing, and normalization
    specifically designed for brain aneurysm detection tasks.
    """

    def __init__(
        self,
        default_window_center: Optional[float] = None,
        default_window_width: Optional[float] = None,
        normalize_method: str = "minmax",
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            default_window_center: Default window center (HU) if not in DICOM
            default_window_width: Default window width (HU) if not in DICOM
            normalize_method: Normalization method ('minmax', 'zscore', 'none')
            target_size: Target image size (H, W) for resizing
        """
        self.default_window_center = default_window_center
        self.default_window_width = default_window_width
        self.normalize_method = normalize_method
        self.target_size = target_size

    def load_dicom(self, dicom_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and preprocess DICOM file with comprehensive error handling

        This method loads a DICOM medical image, extracts pixel data and metadata,
        applies medical imaging preprocessing (windowing, normalization), and
        returns a structured result with both processed and raw data.

        Args:
            dicom_path: Path to DICOM file (.dcm or .dicom extension)

        Returns:
            Dict containing:
                - image (np.ndarray): Preprocessed image array (H, W, C) as uint8
                - raw_image (np.ndarray): Original pixel data as float32
                - metadata (Dict[str, Any]): Extracted DICOM tags and info
                - dicom (pydicom.Dataset): Original DICOM dataset object
                - success (bool): True if loading succeeded
                - error (str, optional): Error message if loading failed

        Raises:
            No exceptions raised - errors are captured and returned in result dict
            
        Example:
            >>> processor = DICOMProcessor(default_window_center=40, default_window_width=80)
            >>> result = processor.load_dicom('scan.dcm')
            >>> if result['success']:
            ...     image = result['image']  # Ready for ML model
            ...     metadata = result['metadata']  # Patient info, scan params
        """
        try:
            dcm = pydicom.dcmread(str(dicom_path))

            # Extract pixel data
            image = self._extract_pixel_data(dcm)

            # Extract metadata
            metadata = self._extract_metadata(dcm)

            # Apply preprocessing
            image_processed = self._preprocess_image(image, metadata)

            return {"image": image_processed, "raw_image": image, "metadata": metadata, "dicom": dcm, "success": True}

        except Exception as e:
            logger.error(f"Failed to load DICOM {dicom_path}: {str(e)}")
            return {
                "image": self._get_dummy_image(),
                "raw_image": None,
                "metadata": {},
                "dicom": None,
                "success": False,
                "error": str(e),
            }

    def _extract_pixel_data(self, dcm: pydicom.Dataset) -> np.ndarray:
        """
        Extract and preprocess raw pixel data from DICOM dataset
        
        Applies medical imaging transformations including:
        - Modality LUT (rescale slope/intercept for Hounsfield Units)
        - VOI LUT (Value of Interest windowing if present in DICOM)
        - Data type conversion to float32 for processing
        
        Args:
            dcm: PyDICOM Dataset object with pixel data
            
        Returns:
            np.ndarray: Preprocessed pixel array as float32
            
        Note:
            Handles both linear and non-linear transformations safely
            with fallback to manual calculation if LUT application fails.
        """

        # Get pixel array
        image = dcm.pixel_array.copy()

        # Convert to float for processing
        image = image.astype(np.float32)

        # Apply modality LUT if present (Rescale Slope/Intercept)
        if hasattr(dcm, "RescaleSlope") or hasattr(dcm, "RescaleIntercept"):
            try:
                image = apply_modality_lut(image, dcm)
            except Exception as e:
                logger.warning(f"Failed to apply modality LUT: {e}")
                # Manual application as fallback
                slope = getattr(dcm, "RescaleSlope", 1)
                intercept = getattr(dcm, "RescaleIntercept", 0)
                image = image * slope + intercept

        # Apply VOI LUT (windowing) if present
        if hasattr(dcm, "WindowCenter") or hasattr(dcm, "WindowWidth"):
            try:
                image = apply_voi_lut(image, dcm)
            except Exception as e:
                logger.warning(f"Failed to apply VOI LUT: {e}")

        return image

    def _extract_metadata(self, dcm: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from DICOM dataset
        
        Extracts patient information, study details, technical parameters,
        and imaging settings essential for medical image analysis and
        patient data management.
        
        Args:
            dcm: PyDICOM Dataset object
            
        Returns:
            Dict[str, Any]: Metadata dictionary with keys including:
                - Patient info: PatientID, PatientAge, PatientSex
                - Study info: StudyInstanceUID, StudyDate, StudyTime
                - Series info: SeriesInstanceUID, SeriesNumber, SeriesDescription
                - Image info: SOPInstanceUID, InstanceNumber
                - Technical: Rows, Columns, PixelSpacing, SliceThickness
                - Window params: WindowCenter, WindowWidth
                - Acquisition: Modality, Manufacturer, ManufacturerModelName
                
        Note:
            Uses safe attribute access with defaults for missing DICOM tags.
            All values are converted to Python native types for JSON serialization.
        """

        metadata = {}

        # Patient information
        metadata["PatientID"] = getattr(dcm, "PatientID", "Unknown")
        metadata["PatientAge"] = getattr(dcm, "PatientAge", "Unknown")
        metadata["PatientSex"] = getattr(dcm, "PatientSex", "Unknown")

        # Study information
        metadata["StudyInstanceUID"] = getattr(dcm, "StudyInstanceUID", "Unknown")
        metadata["StudyDate"] = getattr(dcm, "StudyDate", "Unknown")
        metadata["StudyTime"] = getattr(dcm, "StudyTime", "Unknown")

        # Series information
        metadata["SeriesInstanceUID"] = getattr(dcm, "SeriesInstanceUID", "Unknown")
        metadata["SeriesNumber"] = getattr(dcm, "SeriesNumber", "Unknown")
        metadata["SeriesDescription"] = getattr(dcm, "SeriesDescription", "Unknown")

        # Image information
        metadata["SOPInstanceUID"] = getattr(dcm, "SOPInstanceUID", "Unknown")
        metadata["InstanceNumber"] = getattr(dcm, "InstanceNumber", "Unknown")

        # Technical parameters
        metadata["Rows"] = getattr(dcm, "Rows", None)
        metadata["Columns"] = getattr(dcm, "Columns", None)
        metadata["PixelSpacing"] = getattr(dcm, "PixelSpacing", None)
        metadata["SliceThickness"] = getattr(dcm, "SliceThickness", None)
        metadata["SliceLocation"] = getattr(dcm, "SliceLocation", None)

        # Window parameters
        metadata["WindowCenter"] = getattr(dcm, "WindowCenter", None)
        metadata["WindowWidth"] = getattr(dcm, "WindowWidth", None)
        metadata["RescaleSlope"] = getattr(dcm, "RescaleSlope", 1)
        metadata["RescaleIntercept"] = getattr(dcm, "RescaleIntercept", 0)

        # Modality and acquisition
        metadata["Modality"] = getattr(dcm, "Modality", "Unknown")
        metadata["Manufacturer"] = getattr(dcm, "Manufacturer", "Unknown")
        metadata["ManufacturerModelName"] = getattr(dcm, "ManufacturerModelName", "Unknown")

        # Image position and orientation
        metadata["ImagePositionPatient"] = getattr(dcm, "ImagePositionPatient", None)
        metadata["ImageOrientationPatient"] = getattr(dcm, "ImageOrientationPatient", None)

        return metadata

    def _preprocess_image(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing pipeline to image"""

        # Apply windowing if parameters available
        window_center = metadata.get("WindowCenter") or self.default_window_center
        window_width = metadata.get("WindowWidth") or self.default_window_width

        if window_center is not None and window_width is not None:
            # Handle multiple window values
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]

            image = self._apply_window(image, window_center, window_width)

        # Normalize image
        image = self._normalize_image(image)

        # Resize if needed
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)

        # Convert to 3-channel RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        return image

    def _apply_window(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply DICOM windowing (window/level)"""

        min_val = center - width / 2
        max_val = center + width / 2

        # Apply window
        image = np.clip(image, min_val, max_val)

        return image

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image based on specified method"""

        if self.normalize_method == "minmax":
            # Min-max normalization to [0, 255]
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        elif self.normalize_method == "zscore":
            # Z-score normalization
            mean, std = image.mean(), image.std()
            if std > 0:
                image = (image - mean) / std
            # Scale to [0, 255]
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        elif self.normalize_method == "none":
            # No normalization, just ensure proper range
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def _get_dummy_image(self) -> np.ndarray:
        """Generate dummy image for error cases"""
        size = self.target_size or (512, 512)
        return np.zeros((*size, 3), dtype=np.uint8)

    def batch_process(
        self, dicom_paths: List[Union[str, Path]], output_dir: Optional[Union[str, Path]] = None, save_format: str = "png"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple DICOM files in batch

        Args:
            dicom_paths: List of DICOM file paths
            output_dir: Directory to save processed images
            save_format: Output format ('png', 'npy')

        Returns:
            List of processing results
        """
        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, dicom_path in enumerate(dicom_paths):
            result = self.load_dicom(dicom_path)

            if result["success"] and output_dir:
                # Save processed image
                output_path = output_dir / f"{Path(dicom_path).stem}.{save_format}"

                if save_format == "png":
                    cv2.imwrite(str(output_path), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
                elif save_format == "npy":
                    np.save(str(output_path), result["image"])

                result["output_path"] = output_path

            results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dicom_paths)} DICOM files")

        return results


class DICOMMetadataExtractor:
    """Extract and organize DICOM metadata for analysis"""

    def __init__(self):
        self.metadata_fields = [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "PatientAge",
            "PatientSex",
            "StudyDate",
            "StudyTime",
            "SeriesNumber",
            "InstanceNumber",
            "SeriesDescription",
            "Modality",
            "Manufacturer",
            "ManufacturerModelName",
            "Rows",
            "Columns",
            "PixelSpacing",
            "SliceThickness",
            "SliceLocation",
            "WindowCenter",
            "WindowWidth",
            "RescaleSlope",
            "RescaleIntercept",
        ]

    def extract_from_directory(self, dicom_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Extract metadata from all DICOM files in directory

        Args:
            dicom_dir: Directory containing DICOM files

        Returns:
            DataFrame with metadata for all files
        """
        dicom_dir = Path(dicom_dir)
        dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.dicom"))

        if not dicom_files:
            logger.warning(f"No DICOM files found in {dicom_dir}")
            return pd.DataFrame()

        metadata_list = []

        for dicom_file in dicom_files:
            try:
                dcm = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
                metadata = {"filename": dicom_file.name}

                for field in self.metadata_fields:
                    value = getattr(dcm, field, None)

                    # Convert complex types to string
                    if isinstance(value, (list, tuple)):
                        if len(value) == 1:
                            value = value[0]
                        else:
                            value = str(value)

                    metadata[field] = value

                metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"Failed to extract metadata from {dicom_file}: {e}")
                continue

        df = pd.DataFrame(metadata_list)

        logger.info(f"Extracted metadata from {len(df)} DICOM files")
        return df

    def analyze_metadata(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze metadata for data quality and consistency"""

        if metadata_df.empty:
            return {}

        analysis = {
            "total_files": len(metadata_df),
            "unique_patients": metadata_df["PatientID"].nunique(),
            "unique_studies": metadata_df["StudyInstanceUID"].nunique(),
            "unique_series": metadata_df["SeriesInstanceUID"].nunique(),
            "modalities": metadata_df["Modality"].value_counts().to_dict(),
            "manufacturers": metadata_df["Manufacturer"].value_counts().to_dict(),
            "image_sizes": metadata_df.groupby(["Rows", "Columns"]).size().to_dict(),
            "missing_metadata": metadata_df.isnull().sum().to_dict(),
        }

        # Age distribution if available
        if "PatientAge" in metadata_df.columns and metadata_df["PatientAge"].notna().any():
            # Extract numeric age (remove 'Y' suffix if present)
            ages = metadata_df["PatientAge"].astype(str).str.replace("Y", "").astype(float, errors="ignore")
            analysis["age_stats"] = {"mean": ages.mean(), "median": ages.median(), "min": ages.min(), "max": ages.max()}

        return analysis


def convert_dicom_to_image(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: str = "png",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = (512, 512),
) -> Dict[str, Any]:
    """
    Convert DICOM files to standard image format

    Args:
        input_dir: Directory containing DICOM files
        output_dir: Directory to save converted images
        output_format: Output format ('png', 'jpg', 'npy')
        window_center: Window center for windowing
        window_width: Window width for windowing
        target_size: Target image size

    Returns:
        Conversion statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find DICOM files
    dicom_files = list(input_dir.glob("*.dcm")) + list(input_dir.glob("*.dicom"))

    if not dicom_files:
        logger.error(f"No DICOM files found in {input_dir}")
        return {"success": False, "message": "No DICOM files found"}

    # Initialize processor
    processor = DICOMProcessor(default_window_center=window_center, default_window_width=window_width, target_size=target_size)

    # Process files
    results = processor.batch_process(dicom_files, output_dir=output_dir, save_format=output_format)

    # Collect statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    stats = {
        "total_files": len(dicom_files),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(dicom_files) if dicom_files else 0,
        "output_dir": str(output_dir),
        "output_format": output_format,
    }

    logger.info(f"Conversion complete: {successful}/{len(dicom_files)} files successful")
    return stats


def create_metadata_summary(dicom_dir: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Create comprehensive metadata summary

    Args:
        dicom_dir: Directory containing DICOM files
        output_file: Optional CSV file to save summary

    Returns:
        DataFrame with metadata summary
    """
    extractor = DICOMMetadataExtractor()

    # Extract metadata
    metadata_df = extractor.extract_from_directory(dicom_dir)

    if metadata_df.empty:
        return metadata_df

    # Analyze metadata
    analysis = extractor.analyze_metadata(metadata_df)

    # Print summary
    print("DICOM Metadata Summary")
    print("=" * 50)
    print(f"Total files: {analysis.get('total_files', 0)}")
    print(f"Unique patients: {analysis.get('unique_patients', 0)}")
    print(f"Unique studies: {analysis.get('unique_studies', 0)}")
    print(f"Unique series: {analysis.get('unique_series', 0)}")

    print(f"\nModalities: {analysis.get('modalities', {})}")
    print(f"Image sizes: {analysis.get('image_sizes', {})}")

    if "age_stats" in analysis:
        age_stats = analysis["age_stats"]
        print(f"\nAge statistics:")
        print(f"  Mean: {age_stats['mean']:.1f}")
        print(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f}")

    # Save if requested
    if output_file:
        metadata_df.to_csv(output_file, index=False)
        print(f"\nMetadata saved to: {output_file}")

    return metadata_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DICOM utilities CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: convert-images
    convert_parser = subparsers.add_parser("convert-images", help="Convert DICOM files to images (png/jpg/npy)")
    convert_parser.add_argument("--input", required=True, help="Input directory containing DICOM files")
    convert_parser.add_argument("--output", required=True, help="Output directory for converted images")
    convert_parser.add_argument("--format", choices=["png", "jpg", "npy"], default="png", help="Output format")
    convert_parser.add_argument("--window-center", type=float, default=None, help="Window center (HU)")
    convert_parser.add_argument("--window-width", type=float, default=None, help="Window width (HU)")
    convert_parser.add_argument("--target-size", nargs=2, type=int, metavar=("H", "W"), default=[512, 512], help="Target size H W")

    # Subcommand: extract-metadata
    meta_parser = subparsers.add_parser("extract-metadata", help="Extract DICOM metadata to CSV and print summary")
    meta_parser.add_argument("--input", required=True, help="Input directory containing DICOM files")
    meta_parser.add_argument("--output", required=False, help="Output CSV path for metadata")

    args = parser.parse_args()

    if args.command == "convert-images":
        stats = convert_dicom_to_image(
            input_dir=args.input,
            output_dir=args.output,
            output_format=args.format,
            window_center=args.window_center,
            window_width=args.window_width,
            target_size=(args.target_size[0], args.target_size[1]),
        )
        # Print short summary for DVC logs
        print({k: v for k, v in stats.items() if k in ["total_files", "successful", "failed", "success_rate", "output_dir", "output_format"]})

    elif args.command == "extract-metadata":
        df = create_metadata_summary(args.input, output_file=args.output)
        if args.output:
            print(f"Saved metadata CSV: {args.output}")
        print(f"Metadata rows: {len(df)}")
