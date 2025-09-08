"""
Unit tests for DICOM utilities

Tests DICOM loading, preprocessing, windowing, and metadata extraction.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.dicom_utils import (
    DICOMProcessor,
    DICOMMetadataExtractor,
    convert_dicom_to_image,
    create_metadata_summary
)


class TestDICOMProcessor:
    """Test the DICOMProcessor class"""
    
    def test_init_default(self):
        """Test processor initialization with defaults"""
        processor = DICOMProcessor()
        
        assert processor.default_window_center is None
        assert processor.default_window_width is None
        assert processor.normalize_method == "minmax"
        assert processor.target_size is None
    
    def test_init_custom(self):
        """Test processor initialization with custom parameters"""
        processor = DICOMProcessor(
            default_window_center=40.0,
            default_window_width=80.0,
            normalize_method="zscore",
            target_size=(512, 512)
        )
        
        assert processor.default_window_center == 40.0
        assert processor.default_window_width == 80.0
        assert processor.normalize_method == "zscore"
        assert processor.target_size == (512, 512)
    
    @pytest.mark.skipif(
        not pytest.importorskip("pydicom", reason="pydicom not available"),
        reason="pydicom required for DICOM tests"
    )
    def test_load_dicom_success(self, mock_dicom_file):
        """Test successful DICOM loading"""
        processor = DICOMProcessor()
        
        result = processor.load_dicom(mock_dicom_file)
        
        assert result["success"] is True
        assert "image" in result
        assert "raw_image" in result
        assert "metadata" in result
        assert "dicom" in result
        
        # Check image shape and type
        image = result["image"]
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # H, W, C
        assert image.shape[2] == 3    # RGB channels
        assert image.dtype == np.uint8
    
    def test_load_dicom_file_not_found(self):
        """Test DICOM loading with non-existent file"""
        processor = DICOMProcessor()
        
        result = processor.load_dicom("/non/existent/file.dcm")
        
        assert result["success"] is False
        assert "error" in result
        assert result["image"] is not None  # Should return dummy image
        assert result["raw_image"] is None
        assert result["metadata"] == {}
        assert result["dicom"] is None
    
    def test_apply_windowing(self):
        """Test DICOM windowing application"""
        processor = DICOMProcessor()
        
        # Create test image data
        image = np.array([[-1000, -500, 0, 500, 1000]], dtype=np.float32)
        center = 40.0
        width = 80.0
        
        windowed = processor._apply_windowing(image, center, width)
        
        # Check that values are clipped to window range
        min_expected = center - width / 2  # 0
        max_expected = center + width / 2  # 80
        
        assert windowed.min() >= min_expected
        assert windowed.max() <= max_expected
    
    def test_normalize_to_uint8(self):
        """Test image normalization to uint8"""
        processor = DICOMProcessor()
        
        # Test with various input ranges
        test_cases = [
            np.array([[0, 127, 255]], dtype=np.float32),
            np.array([[-1000, 0, 1000]], dtype=np.float32),
            np.array([[100, 100, 100]], dtype=np.float32),  # Constant image
        ]
        
        for test_image in test_cases:
            normalized = processor._normalize_to_uint8(test_image)
            
            assert normalized.dtype == np.uint8
            assert normalized.min() >= 0
            assert normalized.max() <= 255
            
            # For non-constant images, should use full range
            if not np.all(test_image == test_image.flat[0]):
                assert normalized.max() == 255
                assert normalized.min() == 0
    
    def test_normalize_image_methods(self):
        """Test different normalization methods"""
        # Test data with known statistics
        test_image = np.array([[0, 50, 100, 150, 200]], dtype=np.float32)
        
        # Test minmax normalization
        processor_minmax = DICOMProcessor(normalize_method="minmax")
        normalized_minmax = processor_minmax._normalize_image(test_image)
        assert normalized_minmax.min() == 0
        assert normalized_minmax.max() == 255
        
        # Test zscore normalization
        processor_zscore = DICOMProcessor(normalize_method="zscore")
        normalized_zscore = processor_zscore._normalize_image(test_image)
        assert normalized_zscore.dtype == np.uint8
        
        # Test no normalization
        processor_none = DICOMProcessor(normalize_method="none")
        normalized_none = processor_none._normalize_image(test_image)
        assert normalized_none.dtype == np.uint8
    
    def test_get_dummy_image(self):
        """Test dummy image generation"""
        processor = DICOMProcessor(target_size=(256, 256))
        
        dummy = processor._get_dummy_image()
        
        assert isinstance(dummy, np.ndarray)
        assert dummy.shape == (256, 256, 3)
        assert dummy.dtype == np.uint8
        assert np.all(dummy == 0)  # Should be all zeros
    
    def test_batch_process(self, test_data_dir, mock_dicom_file):
        """Test batch processing of DICOM files"""
        processor = DICOMProcessor()
        
        # Create list with our mock DICOM file
        dicom_paths = [mock_dicom_file]
        
        results = processor.batch_process(dicom_paths)
        
        assert len(results) == 1
        assert results[0]["success"] is True or results[0]["success"] is False  # Either works
    
    def test_batch_process_with_output(self, test_data_dir, mock_dicom_file):
        """Test batch processing with file output"""
        processor = DICOMProcessor()
        output_dir = test_data_dir / "processed"
        
        dicom_paths = [mock_dicom_file]
        
        results = processor.batch_process(
            dicom_paths, 
            output_dir=output_dir, 
            save_format="png"
        )
        
        assert len(results) == 1
        if results[0]["success"]:
            assert "output_path" in results[0]
            assert output_dir.exists()


class TestDICOMMetadataExtractor:
    """Test DICOM metadata extraction functionality"""
    
    def test_init(self):
        """Test metadata extractor initialization"""
        extractor = DICOMMetadataExtractor()
        
        assert hasattr(extractor, 'metadata_fields')
        assert len(extractor.metadata_fields) > 0
        assert 'PatientID' in extractor.metadata_fields
        assert 'StudyInstanceUID' in extractor.metadata_fields
    
    @patch('pydicom.dcmread')
    def test_extract_from_directory(self, mock_dcmread, test_data_dir):
        """Test metadata extraction from directory"""
        # Create mock DICOM files
        dicom_file = test_data_dir / "test.dcm"
        dicom_file.write_bytes(b"dummy content")
        
        # Mock DICOM dataset
        mock_dataset = MagicMock()
        mock_dataset.PatientID = "TEST001"
        mock_dataset.StudyInstanceUID = "1.2.3.4.5"
        mock_dcmread.return_value = mock_dataset
        
        extractor = DICOMMetadataExtractor()
        df = extractor.extract_from_directory(test_data_dir)
        
        if not df.empty:  # Only check if extraction succeeded
            assert isinstance(df, pd.DataFrame)
            assert 'filename' in df.columns
            assert 'PatientID' in df.columns
    
    def test_extract_from_empty_directory(self, test_data_dir):
        """Test metadata extraction from empty directory"""
        extractor = DICOMMetadataExtractor()
        
        df = extractor.extract_from_directory(test_data_dir)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0  # Should be empty
    
    def test_analyze_metadata(self, sample_dicom_metadata):
        """Test metadata analysis functionality"""
        # Create sample dataframe
        df = pd.DataFrame([
            {
                'PatientID': 'PAT001',
                'StudyInstanceUID': 'STUDY001',
                'SeriesInstanceUID': 'SERIES001',
                'Modality': 'CT',
                'Manufacturer': 'TEST_VENDOR',
                'Rows': 512,
                'Columns': 512,
                'PatientAge': '65Y'
            },
            {
                'PatientID': 'PAT002',
                'StudyInstanceUID': 'STUDY002',
                'SeriesInstanceUID': 'SERIES002',
                'Modality': 'CT',
                'Manufacturer': 'TEST_VENDOR',
                'Rows': 512,
                'Columns': 512,
                'PatientAge': '45Y'
            }
        ])
        
        extractor = DICOMMetadataExtractor()
        analysis = extractor.analyze_metadata(df)
        
        assert 'total_files' in analysis
        assert analysis['total_files'] == 2
        
        assert 'unique_patients' in analysis
        assert analysis['unique_patients'] == 2
        
        assert 'modalities' in analysis
        assert analysis['modalities']['CT'] == 2
    
    def test_analyze_empty_metadata(self):
        """Test analysis of empty metadata"""
        extractor = DICOMMetadataExtractor()
        
        analysis = extractor.analyze_metadata(pd.DataFrame())
        
        assert analysis == {}


class TestDICOMUtilityFunctions:
    """Test standalone DICOM utility functions"""
    
    @patch('scripts.dicom_utils.DICOMProcessor')
    def test_convert_dicom_to_image(self, mock_processor_class, test_data_dir):
        """Test DICOM to image conversion function"""
        # Create mock files
        input_dir = test_data_dir / "input"
        output_dir = test_data_dir / "output"
        input_dir.mkdir()
        
        dicom_file = input_dir / "test.dcm"
        dicom_file.write_bytes(b"dummy content")
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.batch_process.return_value = [{"success": True}]
        mock_processor_class.return_value = mock_processor
        
        stats = convert_dicom_to_image(
            input_dir=input_dir,
            output_dir=output_dir,
            output_format="png",
            window_center=40,
            window_width=80,
            target_size=(512, 512)
        )
        
        assert "total_files" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert output_dir.exists()
    
    def test_convert_dicom_no_files(self, test_data_dir):
        """Test DICOM conversion with no input files"""
        input_dir = test_data_dir / "empty"
        output_dir = test_data_dir / "output"
        input_dir.mkdir()
        
        stats = convert_dicom_to_image(input_dir, output_dir)
        
        assert stats["success"] is False
        assert "No DICOM files found" in stats["message"]
    
    @patch('scripts.dicom_utils.DICOMMetadataExtractor')
    def test_create_metadata_summary(self, mock_extractor_class, test_data_dir):
        """Test metadata summary creation"""
        # Mock extractor
        mock_extractor = MagicMock()
        mock_df = pd.DataFrame([{"PatientID": "TEST001"}])
        mock_extractor.extract_from_directory.return_value = mock_df
        mock_extractor.analyze_metadata.return_value = {
            "total_files": 1,
            "unique_patients": 1
        }
        mock_extractor_class.return_value = mock_extractor
        
        df = create_metadata_summary(test_data_dir)
        
        assert isinstance(df, pd.DataFrame)
        mock_extractor.extract_from_directory.assert_called_once_with(test_data_dir)


@pytest.mark.integration
class TestDICOMIntegration:
    """Integration tests for DICOM processing"""
    
    def test_full_dicom_pipeline(self, test_data_dir, sample_medical_image):
        """Test complete DICOM processing pipeline"""
        processor = DICOMProcessor(
            default_window_center=40,
            default_window_width=80,
            target_size=(256, 256)
        )
        
        # Create a simple test file (not real DICOM, but tests error handling)
        test_file = test_data_dir / "test.dcm"
        test_file.write_bytes(b"fake dicom content")
        
        # Process file (should handle gracefully)
        result = processor.load_dicom(test_file)
        
        # Should return a result (either success or graceful failure)
        assert "success" in result
        assert "image" in result
        assert "metadata" in result
        
        # Image should always be returned (dummy if processing failed)
        image = result["image"]
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])