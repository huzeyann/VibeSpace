import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple
import torch.nn.functional as F
from omegaconf import OmegaConf
from ipadapter_model import generate_images_from_clip_embeddings
from ipadapter_model import load_ipadapter
from intrinsic_dim import estimate_intrinsic_dimension
from vibespace_model import VibeSpaceModel, train_vibe_space, clear_gpu_memory
from dino_correspondence import kway_cluster_per_image, match_centers_two_images, get_cluster_center_features

from extract_features import extract_dino_features, extract_clip_features, dino_image_transform, clip_image_transform
import logging
import gradio as gr



DEFAULT_CONFIG_PATH = "./config.yaml"
def load_config(config_path: str):
    cfg_base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    cfg = OmegaConf.load(config_path)
    cfg_base.update(cfg)
    return cfg_base


def run_vibe_blend_safe(image1, image2, extra_images, negative_images, config_path, interpolation_weights: List[float], n_clusters: int = 25):
    success = False
    while not success:
        try:
            model, trainer = run_vibe_space_training(
                positive_images=[image1, image2, *extra_images],
                negative_images=negative_images,
                config_path=config_path,
            )
            success = True
        except Exception as e:
            logging.error(f"Error training model: {e}")
            torch.cuda.empty_cache()
            continue
        
    success = False
    while not success:
        try:
            blended_images = generate_blend_images(
                image1, 
                image2, 
                model,
                interpolation_weights,
                n_clusters=n_clusters, 
            )
            success = True
        except Exception as e:
            logging.error(f"Error generating images: {e}")
            torch.cuda.empty_cache()
            continue

    return blended_images


def run_vibe_blend_not_safe(image1, image2, extra_images, negative_images, config_path, interpolation_weights: List[float], n_clusters: int = 20):
    
    model, trainer = run_vibe_space_training(
        positive_images=[image1, image2, *extra_images],
        negative_images=negative_images,
        config_path=config_path,
    )
    blended_images = generate_blend_images(
        image1, 
        image2, 
        model,
        interpolation_weights,
        n_clusters=n_clusters, 
    )
    return blended_images


def run_vibe_space_training(positive_images: List[Image.Image], 
                            negative_images: List[Image.Image],
                            config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[VibeSpaceModel, object]:
    """
    Train a Mood Space compression model from input images.
    
    This function extracts DINO and CLIP features from the input images,
    estimates the intrinsic dimensionality if not provided, and trains
    a neural compression model to learn a meaningful embedding space.
    
    Args:
        pil_images: List of PIL Images for training
    """
    # Load and configure training parameters
    config = load_config(config_path)
    positive_images = [img for img in positive_images if img is not None]
    negative_images = [img for img in negative_images or [] if img is not None]
    if len(positive_images) == 0:
        raise ValueError("No valid positive images provided for Vibe Space training")
    has_negative_images = len(negative_images) > 0
    
    # Transform images for feature extraction
    dino_input_images = torch.stack([dino_image_transform(image) for image in positive_images])
    clip_input_images = torch.stack([clip_image_transform(image) for image in positive_images])
    if has_negative_images:
        negative_dino_input_images = torch.stack([dino_image_transform(image) for image in negative_images])
    else:
        negative_dino_input_images = None
    
    # Extract features using pre-trained models
    dino_image_embeds = extract_dino_features(dino_input_images)
    clip_image_embeds = extract_clip_features(clip_input_images)
    if has_negative_images:
        negative_dino_embeds = extract_dino_features(negative_dino_input_images)
    else:
        negative_dino_embeds = None
    
    # Determine intrinsic dimensionality
    flattened_features = dino_image_embeds.flatten(end_dim=-2)
    estimated_dim = estimate_intrinsic_dimension(flattened_features)
    hidden_dim = int(estimated_dim)
    config.vibe_dim = hidden_dim
    
    if len(positive_images) > 2:
        # increase training steps for extra images
        config.steps = config.steps * 2
    
    # Create and train model
    model = VibeSpaceModel(config, enable_gradio_progress=True)
    trainer = train_vibe_space(
        model,
        config,
        dino_image_embeds,
        clip_image_embeds,
        negative_dino_embeds,
    )
    
    return model, trainer


def _compute_direction_from_two_images(image_embeds: torch.Tensor, 
                                    eigenvectors: torch.Tensor | List[torch.Tensor],
                                    a_to_b_mapping: np.ndarray, 
                                    use_unit_norm: bool = False) -> torch.Tensor:
    
    # Compute cluster centers
    a_center_features = get_cluster_center_features(
        image_embeds[0], eigenvectors[0].argmax(-1).cpu(), eigenvectors[0].shape[-1])
    b_center_features = get_cluster_center_features(
        image_embeds[1], eigenvectors[1].argmax(-1).cpu(), eigenvectors[1].shape[-1])
    
    # Compute direction vectors
    direction_vectors = []
    for i_a, i_b in enumerate(a_to_b_mapping):
        direction = b_center_features[i_b] - a_center_features[i_a]
        if use_unit_norm:
            direction = F.normalize(direction, dim=-1)
        direction_vectors.append(direction)
    direction_vectors = torch.stack(direction_vectors)
    
    
    # Apply direction based on cluster assignments
    cluster_labels = eigenvectors[0].argmax(-1).cpu()
    direction_field = torch.zeros_like(image_embeds[0])
    
    for i_cluster in range(eigenvectors[0].shape[-1]):
        cluster_mask = cluster_labels == i_cluster
        if cluster_mask.sum() > 0:
            direction_field[cluster_mask] = direction_vectors[i_cluster]
    
    return direction_field


def generate_blend_images(image1: Image.Image, 
                        image2: Image.Image,
                        model: VibeSpaceModel, 
                        interpolation_weights: List[float],
                        n_clusters: int = 20, 
                        seed: Optional[int] = None,
                        ) -> List[Image.Image]:
    """
    Interpolate between two images using the trained compression model.
    
    Args:
        image1, image2: Input PIL Images
        model: Trained compression model
        interpolation_weights: Weights for interpolation
        n_clusters: Number of clusters for correspondence matching
        seed: Random seed for generation
        
    Returns:
        List[Image.Image]: Generated interpolated images
    """
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(img) for img in [image1, image2]])
    dino_image_embeds = extract_dino_features(images)
    compressed_image_embeds = model.encoder(dino_image_embeds)
 
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=None)
    a_to_b_mapping = match_centers_two_images(
        dino_image_embeds[0], dino_image_embeds[1],
        cluster_eigenvectors[0], cluster_eigenvectors[1], 
        match_method='hungarian'
    )
    direction_field = _compute_direction_from_two_images(
        compressed_image_embeds, cluster_eigenvectors, a_to_b_mapping, use_unit_norm=False
    )
    
    # Generate interpolated images
    ip_model = load_ipadapter()
    
    progress_tracker = gr.Progress()
    generated_images = []
    for i, weight in enumerate(interpolation_weights):
        progress_tracker(i / len(interpolation_weights), desc=f"Generating images, α = {weight:.2f}")
        interpolated_embedding = compressed_image_embeds[0] + direction_field * weight
        decompressed_embedding = model.decoder(interpolated_embedding)
        
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decompressed_embedding, num_samples=1, seed=seed
        )
        if np.all(np.array(batch_images[0]) == 0):
            raise ValueError("Generated image is all black")
        generated_images.extend(batch_images)
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return generated_images

