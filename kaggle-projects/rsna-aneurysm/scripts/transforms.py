"""
RSNA Aneurysm Detection - Image Transforms

Albumentations-based image preprocessing and augmentation optimized for medical images.
Includes medical-specific augmentations and quality-preserving transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MedicalImageTransforms:
    """
    Medical image transforms factory
    
    Provides standardized transforms for medical image preprocessing
    with careful consideration of medical image characteristics.
    """
    
    @staticmethod
    def get_train_transforms(
        image_size: Tuple[int, int] = (512, 512),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augmentation_strength: str = 'medium',
        preserve_aspect_ratio: bool = False,
        **kwargs
    ) -> A.Compose:
        """
        Get training transforms
        
        Args:
            image_size: Target image size (H, W)
            mean: Normalization mean
            std: Normalization standard deviation
            augmentation_strength: 'light', 'medium', 'strong'
            preserve_aspect_ratio: Whether to preserve aspect ratio
        """
        
        transforms_list = []
        
        # Resize with aspect ratio handling
        if preserve_aspect_ratio:
            transforms_list.append(
                A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_LINEAR)
            )
            transforms_list.append(
                A.PadIfNeeded(
                    min_height=image_size[0],
                    min_width=image_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0
                )
            )
        else:
            transforms_list.append(
                A.Resize(
                    height=image_size[0],
                    width=image_size[1],
                    interpolation=cv2.INTER_LINEAR,
                    p=1.0
                )
            )
        
        # Add augmentations based on strength
        aug_transforms = MedicalImageTransforms._get_augmentation_transforms(augmentation_strength)
        transforms_list.extend(aug_transforms)
        
        # Final preprocessing
        transforms_list.extend([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ])
        
        return A.Compose(transforms_list)
    
    @staticmethod
    def get_valid_transforms(
        image_size: Tuple[int, int] = (512, 512),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        preserve_aspect_ratio: bool = False,
        **kwargs
    ) -> A.Compose:
        """
        Get validation transforms (minimal preprocessing)
        """
        
        transforms_list = []
        
        # Resize
        if preserve_aspect_ratio:
            transforms_list.extend([
                A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=image_size[0],
                    min_width=image_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0
                )
            ])
        else:
            transforms_list.append(
                A.Resize(
                    height=image_size[0],
                    width=image_size[1],
                    interpolation=cv2.INTER_LINEAR,
                    p=1.0
                )
            )
        
        # Normalize and convert to tensor
        transforms_list.extend([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ])
        
        return A.Compose(transforms_list)
    
    @staticmethod
    def get_test_transforms(
        image_size: Tuple[int, int] = (512, 512),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        tta_type: Optional[str] = None,
        preserve_aspect_ratio: bool = False,
        **kwargs
    ) -> A.Compose:
        """
        Get test transforms with optional TTA
        
        Args:
            tta_type: Type of TTA ('hflip', 'vflip', 'rotate', 'brightness', etc.)
        """
        
        transforms_list = []
        
        # Resize
        if preserve_aspect_ratio:
            transforms_list.extend([
                A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(
                    min_height=image_size[0],
                    min_width=image_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0
                )
            ])
        else:
            transforms_list.append(
                A.Resize(
                    height=image_size[0],
                    width=image_size[1],
                    interpolation=cv2.INTER_LINEAR,
                    p=1.0
                )
            )
        
        # Add TTA transforms
        if tta_type == 'hflip':
            transforms_list.append(A.HorizontalFlip(p=1.0))\n        elif tta_type == 'vflip':
            transforms_list.append(A.VerticalFlip(p=1.0))
        elif tta_type == 'rotate':
            transforms_list.append(A.Rotate(limit=10, p=1.0))
        elif tta_type == 'brightness':
            transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0))
        elif tta_type == 'contrast':
            transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.1, p=1.0))
        
        # Final preprocessing
        transforms_list.extend([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ])
        
        return A.Compose(transforms_list)
    
    @staticmethod
    def _get_augmentation_transforms(strength: str) -> List[A.BasicTransform]:
        """Get augmentation transforms based on strength level"""
        
        if strength == 'light':
            return [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ]
        
        elif strength == 'medium':
            return [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.4),
                A.Rotate(limit=15, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1, 
                    rotate_limit=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.4
                ),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.GaussNoise(var_limit=(10, 30), p=0.2),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.1),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.2
                ),
            ]
        
        elif strength == 'strong':
            return [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=15,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.4),
                    A.GaussianBlur(blur_limit=5, p=0.4),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.2),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                    A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=0.3),
                ], p=0.2),
                A.CoarseDropout(
                    max_holes=12,
                    max_height=48,
                    max_width=48,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                ], p=0.1),
            ]
        
        else:  # No augmentation
            return []


class DICOMSpecificTransforms:
    """DICOM-specific transforms for medical images"""
    
    @staticmethod
    def get_dicom_preprocessing(
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        apply_clahe: bool = True,
        **kwargs
    ) -> A.Compose:
        """
        DICOM-specific preprocessing
        
        Args:
            window_center: DICOM window center (HU)
            window_width: DICOM window width (HU) 
            apply_clahe: Whether to apply CLAHE for contrast enhancement
        """
        
        transforms_list = []
        
        # DICOM windowing (if parameters provided)
        if window_center is not None and window_width is not None:
            transforms_list.append(
                DICOMWindowing(window_center=window_center, window_width=window_width, p=1.0)
            )
        
        # CLAHE for medical image contrast enhancement
        if apply_clahe:
            transforms_list.append(
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
            )
        
        return A.Compose(transforms_list) if transforms_list else A.Compose([A.NoOp()])


class DICOMWindowing(A.ImageOnlyTransform):
    """Custom transform for DICOM windowing"""
    
    def __init__(self, window_center: float, window_width: float, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.window_center = window_center
        self.window_width = window_width
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply DICOM windowing"""
        min_value = self.window_center - self.window_width / 2
        max_value = self.window_center + self.window_width / 2
        
        # Apply windowing
        img = np.clip(img, min_value, max_value)
        
        # Normalize to 0-255
        if max_value > min_value:
            img = ((img - min_value) / (max_value - min_value) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
        
        return img
    
    def get_transform_init_args_names(self):
        return ('window_center', 'window_width')


class MultiScaleTransforms:
    """Multi-scale image transforms"""
    
    @staticmethod
    def get_multiscale_transforms(
        scales: List[int] = [224, 384, 512],
        base_transform: Optional[A.Compose] = None,
        **kwargs
    ) -> Dict[int, A.Compose]:
        """
        Get transforms for multiple scales
        
        Args:
            scales: List of image sizes
            base_transform: Base transform to apply before resizing
        """
        
        if base_transform is None:
            base_transform = A.Compose([A.NoOp()])
        
        scale_transforms = {}
        
        for scale in scales:
            transforms_list = []
            
            # Apply base transforms first
            if len(base_transform.transforms) > 0:
                transforms_list.extend(base_transform.transforms)
            
            # Resize to specific scale
            transforms_list.append(
                A.Resize(height=scale, width=scale, interpolation=cv2.INTER_LINEAR, p=1.0)
            )
            
            scale_transforms[scale] = A.Compose(transforms_list)
        
        return scale_transforms


class TTATransforms:
    """Test Time Augmentation transforms"""
    
    @staticmethod
    def get_tta_transforms(
        image_size: Tuple[int, int] = (512, 512),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        tta_methods: List[str] = ['original', 'hflip', 'vflip'],
        **kwargs
    ) -> Dict[str, A.Compose]:
        """
        Get TTA transforms dictionary
        
        Args:
            tta_methods: List of TTA methods to apply
        """
        
        tta_transforms = {}
        
        for method in tta_methods:
            if method == 'original':
                tta_transforms[method] = MedicalImageTransforms.get_test_transforms(
                    image_size=image_size, mean=mean, std=std, tta_type=None
                )
            else:
                tta_transforms[method] = MedicalImageTransforms.get_test_transforms(
                    image_size=image_size, mean=mean, std=std, tta_type=method
                )
        
        return tta_transforms


def create_transforms(config: Dict[str, Any], mode: str = 'train') -> A.Compose:
    """
    Factory function to create transforms from config
    
    Args:
        config: Configuration dictionary
        mode: 'train', 'valid', or 'test'
    """
    
    data_config = config.get('data', {})
    aug_config = config.get('augmentation', {})
    
    # Common parameters
    image_size = tuple(data_config.get('image_size', [512, 512]))
    normalization = data_config.get('normalization', {})
    mean = normalization.get('mean', [0.485, 0.456, 0.406])
    std = normalization.get('std', [0.229, 0.224, 0.225])
    
    if mode == 'train':
        augmentation_strength = aug_config.get('strength', 'medium')
        return MedicalImageTransforms.get_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            augmentation_strength=augmentation_strength,
            **aug_config
        )
    
    elif mode == 'valid':
        return MedicalImageTransforms.get_valid_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            **data_config
        )
    
    elif mode == 'test':
        return MedicalImageTransforms.get_test_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            **data_config
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_tta_transforms(config: Dict[str, Any]) -> Dict[str, A.Compose]:
    """Create TTA transforms from config"""
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    inference_config = model_config.get('inference', {})
    
    if not inference_config.get('tta_enabled', False):
        return {'original': create_transforms(config, mode='test')}
    
    # TTA parameters
    image_size = tuple(data_config.get('image_size', [512, 512]))
    normalization = data_config.get('normalization', {})
    mean = normalization.get('mean', [0.485, 0.456, 0.406])
    std = normalization.get('std', [0.229, 0.224, 0.225])
    tta_methods = inference_config.get('tta_transforms', ['hflip', 'vflip'])
    
    return TTATransforms.get_tta_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        tta_methods=['original'] + tta_methods
    )


if __name__ == "__main__":
    # Test transforms creation
    config = {
        'data': {
            'image_size': [512, 512],
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'augmentation': {
            'strength': 'medium'
        }
    }
    
    # Create transforms
    train_transform = create_transforms(config, mode='train')
    valid_transform = create_transforms(config, mode='valid')
    
    print(f"Train transforms: {len(train_transform.transforms)} steps")
    print(f"Valid transforms: {len(valid_transform.transforms)} steps")
    
    # Test TTA
    config['model'] = {
        'inference': {
            'tta_enabled': True,
            'tta_transforms': ['hflip', 'vflip', 'rotate']
        }
    }
    
    tta_transforms = create_tta_transforms(config)
    print(f"TTA transforms: {list(tta_transforms.keys())}")