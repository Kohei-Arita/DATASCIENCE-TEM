"""
RSNA Aneurysm Detection - Dataset Classes

PyTorch Dataset implementations for medical image classification.
Supports DICOM and standard image formats with robust error handling.
"""

import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class AneurysmDataset(Dataset):
    """
    Dataset for RSNA Intracranial Aneurysm Detection
    
    Supports multiple image formats:
    - DICOM files (.dcm)
    - Standard images (.png, .jpg, .jpeg)
    - Numpy arrays (.npy)
    
    Features:
    - Robust error handling with fallbacks
    - Memory-efficient loading
    - Flexible transforms
    - Multi-modal data support
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        mode: str = 'train',
        image_column: str = 'image_id',
        target_column: str = 'aneurysm',
        transform: Optional[Callable] = None,
        image_format: str = 'auto',  # 'dicom', 'png', 'jpg', 'npy', 'auto'
        cache_images: bool = False,
        return_meta: bool = False
    ):
        """
        Args:
            df: DataFrame with image IDs and labels
            image_dir: Directory containing images
            mode: 'train', 'valid', or 'test'
            image_column: Column name for image IDs
            target_column: Column name for target labels
            transform: Albumentation transforms
            image_format: Image file format
            cache_images: Whether to cache loaded images in memory
            return_meta: Whether to return metadata along with images
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.image_column = image_column
        self.target_column = target_column
        self.transform = transform
        self.image_format = image_format
        self.cache_images = cache_images
        self.return_meta = return_meta
        
        # Image cache
        self._image_cache = {} if cache_images else None
        
        # Supported extensions
        self.extensions = {
            'dicom': ['.dcm', '.dicom'],
            'png': ['.png'],
            'jpg': ['.jpg', '.jpeg'],
            'npy': ['.npy']
        }
        
        logger.info(f"Dataset initialized: {len(self.df)} samples in '{mode}' mode")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get item by index
        
        Returns:
            For train/valid: (image_tensor, label_tensor)
            For test: image_tensor
            With return_meta=True: adds metadata dict
        """
        try:
            row = self.df.iloc[idx]
            image_id = row[self.image_column]
            
            # Load image
            image = self._load_image(image_id, idx)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            # Prepare output
            if self.mode == 'test':
                if self.return_meta:
                    meta = self._get_metadata(row, idx)
                    return image, meta
                return image
            else:
                label = torch.tensor(row[self.target_column], dtype=torch.float32)
                if self.return_meta:
                    meta = self._get_metadata(row, idx)
                    return image, label, meta
                return image, label
                
        except Exception as e:
            logger.error(f"Error loading sample {idx} (image_id: {image_id}): {str(e)}")
            # Return dummy data to prevent training interruption
            return self._get_dummy_sample()
    
    def _load_image(self, image_id: str, idx: int) -> np.ndarray:
        """Load image from file with format auto-detection and fallbacks"""
        
        # Check cache first
        if self.cache_images and image_id in self._image_cache:
            return self._image_cache[image_id]
        
        # Determine file path
        image_path = self._find_image_path(image_id)
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return self._get_dummy_image()
        
        try:
            # Load based on format
            if self.image_format == 'dicom' or image_path.suffix.lower() in self.extensions['dicom']:
                image = self._load_dicom(image_path)
            elif image_path.suffix.lower() in self.extensions['npy']:
                image = self._load_npy(image_path)
            else:
                image = self._load_standard_image(image_path)
            
            # Validate image
            image = self._validate_image(image)
            
            # Cache if enabled
            if self.cache_images:
                self._image_cache[image_id] = image
                
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return self._get_dummy_image()
    
    def _find_image_path(self, image_id: str) -> Path:
        """Find image file path with format detection"""
        
        if self.image_format != 'auto':
            # Use specified format
            extensions = self.extensions[self.image_format]
            for ext in extensions:
                path = self.image_dir / f"{image_id}{ext}"
                if path.exists():
                    return path
        else:
            # Auto-detect format
            for format_type, extensions in self.extensions.items():
                for ext in extensions:
                    path = self.image_dir / f"{image_id}{ext}"
                    if path.exists():
                        return path
        
        # Default to PNG if not found
        return self.image_dir / f"{image_id}.png"
    
    def _load_dicom(self, image_path: Path) -> np.ndarray:
        """Load DICOM image with proper preprocessing"""
        
        dcm = pydicom.dcmread(str(image_path))
        
        # Get pixel data
        image = dcm.pixel_array
        
        # Apply rescale slope/intercept if present
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Convert to Hounsfield Units if needed
        image = image.astype(np.float32)
        
        # Apply windowing if window center/width are specified
        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
            window_center = dcm.WindowCenter
            window_width = dcm.WindowWidth
            
            # Handle multiple values
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]
            
            # Apply windowing
            image = self._apply_windowing(image, window_center, window_width)
        
        # Convert to 8-bit and ensure 3-channel RGB
        image = self._normalize_to_uint8(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_npy(self, image_path: Path) -> np.ndarray:
        """Load numpy array image"""
        image = np.load(str(image_path))
        
        # Ensure 3D (H, W, C)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = np.transpose(image, (1, 2, 0))
            image = np.concatenate([image] * 3, axis=-1)
        
        return image.astype(np.uint8)
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        """Load standard image formats (PNG, JPG)"""
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _apply_windowing(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply DICOM windowing"""
        min_value = center - width / 2
        max_value = center + width / 2
        
        image = np.clip(image, min_value, max_value)
        return image
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 [0, 255]"""
        # Normalize to 0-1
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
        
        # Convert to uint8
        return (image * 255).astype(np.uint8)
    
    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate and fix image format"""
        # Ensure 3D
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = self._normalize_to_uint8(image)
        
        # Ensure minimum size
        if image.shape[0] < 32 or image.shape[1] < 32:
            image = cv2.resize(image, (224, 224))
        
        return image
    
    def _get_dummy_image(self) -> np.ndarray:
        """Generate dummy image for error cases"""
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _get_dummy_sample(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate dummy sample for error cases"""
        dummy_image = torch.zeros(3, 224, 224)
        
        if self.mode == 'test':
            return dummy_image
        else:
            dummy_label = torch.tensor(0.0, dtype=torch.float32)
            return dummy_image, dummy_label
    
    def _get_metadata(self, row: pd.Series, idx: int) -> dict:
        """Extract metadata for sample"""
        meta = {
            'index': idx,
            'image_id': row[self.image_column],
        }
        
        # Add available metadata columns
        meta_columns = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'fold']
        for col in meta_columns:
            if col in row:
                meta[col] = row[col]
        
        return meta
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        if self.mode == 'test' or self.target_column not in self.df.columns:
            return None
        
        targets = self.df[self.target_column].values
        pos_count = targets.sum()
        neg_count = len(targets) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return torch.tensor([1.0, 1.0])
        
        # Inverse frequency weighting
        pos_weight = neg_count / pos_count
        return torch.tensor([1.0, pos_weight])


class MultiViewDataset(AneurysmDataset):
    """
    Dataset for multi-view/multi-slice medical images
    
    Handles cases where each sample has multiple images/slices
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        views_per_sample: int = 3,
        view_column: str = 'view_id',
        **kwargs
    ):
        """
        Args:
            views_per_sample: Number of views/slices per sample
            view_column: Column containing view identifiers
        """
        super().__init__(df, image_dir, **kwargs)
        self.views_per_sample = views_per_sample
        self.view_column = view_column
        
        # Group by sample ID
        self.sample_groups = self.df.groupby(self.image_column)
        self.sample_ids = list(self.sample_groups.groups.keys())
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get multi-view sample"""
        sample_id = self.sample_ids[idx]
        sample_data = self.sample_groups.get_group(sample_id)
        
        # Load multiple views
        images = []
        for _, row in sample_data.head(self.views_per_sample).iterrows():
            view_id = f"{sample_id}_{row[self.view_column]}" if self.view_column in row else sample_id
            image = self._load_image(view_id, idx)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            images.append(image)
        
        # Pad if necessary
        while len(images) < self.views_per_sample:
            images.append(images[-1] if images else self._get_dummy_image())
        
        # Stack images
        images = torch.stack(images[:self.views_per_sample])
        
        if self.mode == 'test':
            return images
        else:
            # Use label from first row
            label = torch.tensor(sample_data.iloc[0][self.target_column], dtype=torch.float32)
            return images, label