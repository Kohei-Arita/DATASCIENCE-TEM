"""
Pytest configuration and shared fixtures for RSNA aneurysm detection tests

Provides common test fixtures, utilities, and configurations for the test suite.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    test_dir = Path(tempfile.mkdtemp(prefix="rsna_test_"))
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing model creation"""
    return {
        "model": {
            "architecture": "resnet18",  # Lightweight for testing
            "num_classes": 1,
            "pretrained": False,  # Faster initialization
            "dropout": 0.5,
            "hidden_dim": 128,
            "use_attention": False
        },
        "data": {
            "image_size": [224, 224],
            "channels": 3,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "train": {
            "batch_size": 4,
            "epochs": 2,
            "lr": 0.001
        }
    }


@pytest.fixture
def sample_medical_image() -> np.ndarray:
    """Generate sample medical image for testing"""
    # Create a realistic medical image with some structure
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some circular structures to simulate anatomy
    y, x = np.ogrid[:512, :512]
    
    # Add a few circular regions (simulating brain structures)
    for center_y, center_x, radius in [(200, 200, 50), (300, 350, 30), (150, 400, 40)]:
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        image[mask] = image[mask] * 0.7 + 80  # Darken regions
    
    return image


@pytest.fixture
def sample_dicom_metadata() -> Dict[str, Any]:
    """Sample DICOM metadata for testing"""
    return {
        "PatientID": "TEST001",
        "PatientAge": "65Y",
        "PatientSex": "M",
        "StudyInstanceUID": "1.2.3.4.5.6.7.8.9.10",
        "SeriesInstanceUID": "1.2.3.4.5.6.7.8.9.11",
        "SOPInstanceUID": "1.2.3.4.5.6.7.8.9.12",
        "StudyDate": "20230901",
        "StudyTime": "120000",
        "SeriesNumber": "1",
        "InstanceNumber": "1",
        "Modality": "CT",
        "Manufacturer": "TEST_MANUFACTURER",
        "Rows": 512,
        "Columns": 512,
        "PixelSpacing": [0.488281, 0.488281],
        "SliceThickness": 5.0,
        "WindowCenter": [40.0],
        "WindowWidth": [80.0],
        "RescaleSlope": 1.0,
        "RescaleIntercept": -1024.0
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing dataset functionality"""
    return pd.DataFrame({
        "image_id": [f"IMG_{i:03d}" for i in range(10)],
        "aneurysm": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "PatientID": [f"PAT_{i:03d}" for i in range(10)],
        "fold": [i % 5 for i in range(10)],
        "StudyInstanceUID": [f"1.2.3.4.5.{i}" for i in range(10)]
    })


@pytest.fixture
def sample_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample batch for testing model forward pass"""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    return images, labels


@pytest.fixture
def mock_dicom_file(test_data_dir, sample_medical_image):
    """Create a mock DICOM file for testing"""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        
        # Create a minimal DICOM dataset
        ds = Dataset()
        ds.PatientName = "TEST^PATIENT"
        ds.PatientID = "TEST001"
        ds.Modality = "CT"
        ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.10"
        ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.11"
        ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.12"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        
        # Add image data
        gray_image = np.mean(sample_medical_image, axis=2).astype(np.uint16)
        ds.Rows, ds.Columns = gray_image.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = gray_image.tobytes()
        
        # Add windowing parameters
        ds.WindowCenter = [40.0]
        ds.WindowWidth = [80.0]
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        
        # Save to file
        dicom_path = test_data_dir / "test_image.dcm"
        ds.save_as(str(dicom_path))
        
        return dicom_path
        
    except ImportError:
        # If pydicom not available, create a placeholder
        dicom_path = test_data_dir / "test_image.dcm"
        with open(dicom_path, 'wb') as f:
            f.write(b"DICM" + b"0" * 1000)  # Minimal fake DICOM
        return dicom_path


@pytest.fixture(autouse=True)
def set_test_seeds():
    """Set random seeds for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


class TestDeviceManager:
    """Utility class for managing test device (CPU/GPU)"""
    
    @staticmethod
    def get_test_device():
        """Get appropriate device for testing (prefer CPU for consistency)"""
        return torch.device("cpu")
    
    @staticmethod
    def requires_gpu():
        """Mark test as requiring GPU"""
        return pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="GPU not available"
        )


# Test markers
pytest.register_marker("slow", "Marks tests as slow")
pytest.register_marker("integration", "Marks tests as integration tests")
pytest.register_marker("unit", "Marks tests as unit tests")
pytest.register_marker("security", "Marks tests as security-related")
pytest.register_marker("gpu", "Marks tests as requiring GPU")