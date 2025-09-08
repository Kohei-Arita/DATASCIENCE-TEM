"""
RSNA Aneurysm Detection - Custom Exceptions

Domain-specific exceptions for medical imaging ML pipeline.
Provides structured error handling with contextual information.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RSNAException(Exception):
    """
    Base exception for all RSNA aneurysm detection errors
    
    Provides structured error handling with context and logging integration.
    All domain-specific exceptions should inherit from this base class.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize structured exception
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for programmatic handling
            context: Additional context information (patient_id, file_path, etc.)
            original_error: Original exception if this is a wrapped error
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.original_error = original_error
        
        # Create enhanced message with context
        full_message = f"[{self.error_code}] {message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            full_message += f" (Context: {context_str})"
        
        super().__init__(full_message)
        
        # Log the error automatically
        logger.error(f"{self.__class__.__name__}: {full_message}", 
                    extra={'error_code': self.error_code, 'context': self.context})


# Data Processing Exceptions
class DataProcessingError(RSNAException):
    """Raised when data processing operations fail"""
    pass


class DICOMProcessingError(DataProcessingError):
    """
    Raised when DICOM file processing fails
    
    Common scenarios:
    - Corrupted DICOM files
    - Missing required DICOM tags
    - Unsupported DICOM format
    - Windowing parameter errors
    """
    pass


class ImageProcessingError(DataProcessingError):
    """
    Raised when image preprocessing fails
    
    Common scenarios:
    - Invalid image dimensions
    - Unsupported image format
    - Transformation errors
    - Normalization failures
    """
    pass


class DataValidationError(DataProcessingError):
    """
    Raised when data validation checks fail
    
    Common scenarios:
    - Missing required columns
    - Invalid data ranges
    - Data quality issues
    - Schema validation failures
    """
    pass


# Model Training Exceptions
class ModelError(RSNAException):
    """Base class for model-related errors"""
    pass


class ModelArchitectureError(ModelError):
    """
    Raised when model architecture issues occur
    
    Common scenarios:
    - Unsupported architecture name
    - Invalid model parameters
    - Architecture compatibility issues
    """
    pass


class ModelTrainingError(ModelError):
    """
    Raised when model training fails
    
    Common scenarios:
    - Training convergence issues
    - Memory allocation errors
    - Invalid training parameters
    - Data loading failures during training
    """
    pass


class ModelLoadingError(ModelError):
    """
    Raised when model loading/saving fails
    
    Common scenarios:
    - Checkpoint file corruption
    - Model version incompatibility
    - Missing model files
    - Permission errors
    """
    pass


# Configuration Exceptions
class ConfigurationError(RSNAException):
    """
    Raised when configuration issues occur
    
    Common scenarios:
    - Missing required configuration keys
    - Invalid configuration values
    - Configuration file parsing errors
    - Environment variable issues
    """
    pass


# Security Exceptions (already defined in security_utils.py, but imported here)
class SecurityError(RSNAException):
    """
    Raised when security violations occur
    
    Common scenarios:
    - Command injection attempts
    - Path traversal attacks
    - Invalid input validation
    - Unauthorized operations
    """
    pass


# Experiment Management Exceptions
class ExperimentError(RSNAException):
    """
    Raised when experiment management issues occur
    
    Common scenarios:
    - Duplicate experiment IDs
    - Missing experiment artifacts
    - Experiment state inconsistencies
    - Tracking system failures
    """
    pass


class ReproducibilityError(ExperimentError):
    """
    Raised when reproducibility cannot be ensured
    
    Common scenarios:
    - Missing random seeds
    - Environment differences
    - Dependency version mismatches
    - Non-deterministic operations
    """
    pass


# Kaggle Integration Exceptions
class KaggleError(RSNAException):
    """
    Raised when Kaggle API operations fail
    
    Common scenarios:
    - API authentication failures
    - Competition access errors
    - Download/upload failures
    - Rate limit exceeded
    """
    pass


class SubmissionError(KaggleError):
    """
    Raised when submission to Kaggle fails
    
    Common scenarios:
    - Invalid submission format
    - Submission size limits
    - Competition deadline passed
    - Network connectivity issues
    """
    pass


# Resource Management Exceptions
class ResourceError(RSNAException):
    """Base class for resource-related errors"""
    pass


class MemoryError(ResourceError):
    """
    Raised when memory allocation fails
    
    Common scenarios:
    - Insufficient system memory
    - GPU memory exhaustion
    - Memory leak detection
    - Batch size too large
    """
    pass


class StorageError(ResourceError):
    """
    Raised when storage operations fail
    
    Common scenarios:
    - Disk space exhaustion
    - File permission errors
    - Network storage failures
    - Quota exceeded
    """
    pass


class GPUError(ResourceError):
    """
    Raised when GPU operations fail
    
    Common scenarios:
    - CUDA out of memory
    - GPU driver issues
    - Unsupported GPU operations
    - Multi-GPU synchronization errors
    """
    pass


# Utility Functions for Error Handling
def handle_dicom_error(func):
    """
    Decorator to handle DICOM processing errors gracefully
    
    Converts common DICOM exceptions into domain-specific exceptions
    with additional context and logging.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Add context if available
            context = {}
            if args and hasattr(args[0], '__dict__'):
                # If first argument is an object, try to extract context
                obj = args[0]
                if hasattr(obj, 'dicom_path'):
                    context['dicom_path'] = str(obj.dicom_path)
            
            raise DICOMProcessingError(
                f"DICOM processing failed in {func.__name__}: {str(e)}",
                context=context,
                original_error=e
            ) from e
    
    return wrapper


def handle_model_error(func):
    """
    Decorator to handle model-related errors gracefully
    
    Converts common model exceptions into domain-specific exceptions
    with model information and context.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Add model context if available
            context = {}
            if args and hasattr(args[0], 'architecture'):
                context['architecture'] = args[0].architecture
            if args and hasattr(args[0], 'num_classes'):
                context['num_classes'] = args[0].num_classes
            
            raise ModelError(
                f"Model operation failed in {func.__name__}: {str(e)}",
                context=context,
                original_error=e
            ) from e
    
    return wrapper


def create_error_context(
    patient_id: Optional[str] = None,
    image_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    file_path: Optional[str] = None,
    **additional_context
) -> Dict[str, Any]:
    """
    Create standardized error context dictionary
    
    Args:
        patient_id: Patient identifier
        image_id: Image identifier
        experiment_id: Experiment identifier
        file_path: File path related to error
        **additional_context: Additional context key-value pairs
        
    Returns:
        Dictionary with non-None context values
    """
    context = {}
    
    if patient_id:
        context['patient_id'] = patient_id
    if image_id:
        context['image_id'] = image_id
    if experiment_id:
        context['experiment_id'] = experiment_id
    if file_path:
        context['file_path'] = str(file_path)
    
    context.update(additional_context)
    
    return context


# Exception Registry for Error Analysis
EXCEPTION_REGISTRY = {
    'DATA_PROCESSING': [DataProcessingError, DICOMProcessingError, ImageProcessingError, DataValidationError],
    'MODEL': [ModelError, ModelArchitectureError, ModelTrainingError, ModelLoadingError],
    'CONFIGURATION': [ConfigurationError],
    'SECURITY': [SecurityError],
    'EXPERIMENT': [ExperimentError, ReproducibilityError],
    'KAGGLE': [KaggleError, SubmissionError],
    'RESOURCE': [ResourceError, MemoryError, StorageError, GPUError]
}


def get_exception_category(exception: Exception) -> Optional[str]:
    """
    Get the category of an exception for error analysis
    
    Args:
        exception: Exception to categorize
        
    Returns:
        Category name or None if not found
    """
    exception_type = type(exception)
    
    for category, exception_types in EXCEPTION_REGISTRY.items():
        if exception_type in exception_types:
            return category
    
    return None


if __name__ == "__main__":
    # Example usage and testing
    try:
        raise DICOMProcessingError(
            "Failed to process DICOM file",
            error_code="DICOM_CORRUPT",
            context=create_error_context(
                patient_id="PAT001",
                image_id="IMG123",
                file_path="/path/to/image.dcm"
            )
        )
    except DICOMProcessingError as e:
        print(f"Caught exception: {e}")
        print(f"Error code: {e.error_code}")
        print(f"Context: {e.context}")
        print(f"Category: {get_exception_category(e)}")