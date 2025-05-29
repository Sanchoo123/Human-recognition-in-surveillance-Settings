# Fused Frame Dataset Module
# This module provides functionality for creating fused image pairs for biometric recognition
# It combines two images into a single 6-channel tensor for siamese network training

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random


def resize_and_pad(image, target_shape=(256, 128)):
    """
    Resizes image while maintaining aspect ratio and adds padding to reach target shape.
    
    This function ensures all images have consistent dimensions while preserving
    the original aspect ratio to avoid distortion.
    
    Args:
        image (numpy.ndarray): Input image as numpy array
        target_shape (tuple): Target size as (height, width), default: (256, 128)
    
    Returns:
        numpy.ndarray: Resized image with padding to match target shape
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image using area interpolation for better quality
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas with target dimensions and center the resized image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    
    return canvas


def create_fused_instance(img1, img2, output_shape=(256, 128)):
    """
    Creates a fused instance of two images for model input.
    
    This function combines two RGB images into a single 6-channel tensor
    by concatenating them depth-wise (img1 channels + img2 channels).
    
    Args:
        img1 (numpy.ndarray): First image as numpy array
        img2 (numpy.ndarray): Second image as numpy array
        output_shape (tuple): Output shape as (height, width), default: (256, 128)
    
    Returns:
        torch.Tensor: PyTorch tensor with img1 and img2 concatenated in depth (6 channels)
    """
    # Resize both images to consistent dimensions
    img1_resized = resize_and_pad(img1, output_shape)
    img2_resized = resize_and_pad(img2, output_shape)
    
    # Concatenate images depth-wise: img1 (3 channels) + img2 (3 channels) = 6 channels
    fused_image = np.dstack((img1_resized, img2_resized))
    
    # Convert to PyTorch tensor and normalize to [0, 1] range
    # Transpose from HWC to CHW format (channels first)
    fused_tensor = torch.from_numpy(fused_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
    
    return fused_tensor


def apply_augmentation(image):
    """
    Applies data augmentation transformations to improve model generalization.
    
    This function randomly applies various transformations including brightness,
    contrast, blur, and noise to create more diverse training data.
    
    Args:
        image (numpy.ndarray): Input image as numpy array in RGB format
    
    Returns:
        numpy.ndarray: Image with applied augmentation transformations
    """
    # Probability thresholds for applying each transformation
    p_brightness = 0.5
    p_contrast = 0.3
    p_blur = 0.2
    p_noise = 0.2
    
    # Work on a copy to avoid modifying original image
    result = image.copy()
    
    # 1. Brightness adjustment
    if random.random() < p_brightness:
        brightness_factor = random.uniform(0.8, 1.2)  # ±20% brightness variation
        result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
    
    # 2. Contrast adjustment
    if random.random() < p_contrast:
        contrast_factor = random.uniform(0.8, 1.2)  # ±20% contrast variation
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * contrast_factor + mean
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 3. Gaussian blur for slight defocus simulation
    if random.random() < p_blur:
        kernel_size = random.choice([3, 5])  # Small blur kernels
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # 4. Gaussian noise to simulate sensor noise
    if random.random() < p_noise:
        noise = np.random.normal(0, 5, result.shape).astype(np.uint8)
        result = cv2.add(result, noise)
    
    return result


class FusedFramePairsDataset(Dataset):
    """
    PyTorch Dataset that creates fused pairs of images for biometric recognition.
    
    This dataset loads image pairs, applies optional augmentation, and creates
    6-channel fused tensors suitable for siamese network training.
    """
    
    def __init__(self, pairs, transform=None, output_shape=(256, 128), use_augmentation=False):
        """
        Initialize the dataset with image pairs and configuration.
        
        Args:
            pairs (list): List of tuples (frame1_path, frame2_path, label)
                         where label is 1 for genuine pairs, 0 for impostor pairs
            transform (callable, optional): PyTorch transforms to apply after fusion
            output_shape (tuple): Output shape of images as (height, width)
            use_augmentation (bool): Whether to apply data augmentation during training
        """
        self.pairs = pairs
        self.transform = transform
        self.output_shape = output_shape
        self.use_augmentation = use_augmentation
        
        if use_augmentation:
            print("Data augmentation enabled in dataset")
    
    def __len__(self):
        """Return the total number of image pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Retrieve and process a single image pair from the dataset.
        
        Args:
            idx (int): Index of the pair to retrieve
            
        Returns:
            tuple: (fused_tensor, label) where:
                   - fused_tensor is a 6-channel PyTorch tensor
                   - label is a float tensor (1.0 for genuine, 0.0 for impostor)
        """
        frame1_path, frame2_path, label = self.pairs[idx]
        
        # Load images using OpenCV
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        # Validate that images were loaded successfully
        if frame1 is None:
            raise ValueError(f"Could not read image: {frame1_path}")
        
        if frame2 is None:
            raise ValueError(f"Could not read image: {frame2_path}")
        
        # Convert from BGR (OpenCV default) to RGB format
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Apply data augmentation if enabled (only on genuine pairs for better training)
        if self.use_augmentation and label == 1:
            # Apply augmentation with 50% probability to each image independently
            if random.random() < 0.5:
                frame1 = apply_augmentation(frame1)
            if random.random() < 0.5:
                frame2 = apply_augmentation(frame2)
        
        # Create fused 6-channel tensor from the two images
        fused_tensor = create_fused_instance(frame1, frame2, self.output_shape)
        
        # Apply additional transforms if specified (e.g., normalization)
        if self.transform:
            fused_tensor = self.transform(fused_tensor)
        
        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.float32)
        
        return fused_tensor, label