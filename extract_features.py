"""
Feature Extraction Module

This module provides utilities for extracting features from images using various
pre-trained models including DINO, DINOv3, and CLIP. It handles model loading,
batch processing, and memory management for efficient feature extraction.
"""

import gc
from typing import Tuple, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms

from ipadapter_model import extract_clip_embedding_tensor
from ipadapter_model import load_ipadapter


# Default hyperparameters
DEFAULT_BATCH_SIZE = 32


# ===== Image Transforms =====

# High-resolution transform for DINO models
dino_image_transform = transforms.Compose([
    transforms.Resize((256 * 2, 256 * 2)),  # High resolution for detailed features
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Standard resolution transform for CLIP models  
clip_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard ImageNet resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inverse transform to convert normalized tensors back to PIL images
image_inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage(),
])


# ===== Memory Management =====

def clear_gpu_memory():
    """Clear GPU cache and run garbage collection to free memory."""
    torch.cuda.empty_cache()
    gc.collect()


# ===== Feature Extraction Functions =====

@torch.no_grad()
def extract_dino_features(images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE) -> torch.Tensor:
    """
    Extract features using DINO ViT-S/16 model.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: DINO features of shape (N, L, D)
    """
    # Load DINO model
    #dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    dino_model = dino_model.eval().cuda()

    # Process images in batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    feature_batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, images.shape[0])
        
        batch_images = images[start_idx:end_idx].cuda()
        batch_features = dino_model.get_intermediate_layers(batch_images)[-1]
        feature_batches.append(batch_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(feature_batches, dim=0)
    
    # Clean up memory
    del dino_model
    clear_gpu_memory()

    return all_features


@torch.no_grad()
def extract_clip_features(images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE, ipadapter_version: str = "sd15") -> torch.Tensor:
    """
    Extract features using CLIP vision encoder.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: CLIP features of shape (N, L, D)
    """
    # Load IP-Adapter model (contains CLIP encoder)
    ip_adapter_model = load_ipadapter(version=ipadapter_version)
    
    # Process images in batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    feature_batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, images.shape[0])
        
        batch_images = images[start_idx:end_idx].cuda()
        batch_features = extract_clip_embedding_tensor(
            batch_images, ip_adapter_model, resize=False
        )
        feature_batches.append(batch_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(feature_batches, dim=0)
    
    # Clean up memory
    del ip_adapter_model
    clear_gpu_memory()

    return all_features

