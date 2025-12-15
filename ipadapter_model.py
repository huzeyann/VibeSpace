"""
IP-Adapter Model Interface

This module provides utilities for working with IP-Adapter models, including:
- Loading Stable Diffusion pipelines with IP-Adapter
- Extracting CLIP embeddings from images
- Generating images from CLIP embeddings
- Utility functions for image processing
"""

from typing import List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL

# Fix for torch 2.5.0 compatibility
torch.backends.cuda.enable_cudnn_sdp(False)

from ip_adapter import IPAdapterPlus, IPAdapterPlusXL


# ===== Image Utility Functions =====

def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    # Get dimensions from first image (assumes all images are same size)
    width, height = images[0].size
    
    # Create empty grid canvas
    grid = Image.new('RGB', size=(cols * width, rows * height))
    
    # Paste each image into the grid
    for i, img in enumerate(images):
        x_pos = (i % cols) * width
        y_pos = (i // cols) * height
        grid.paste(img, box=(x_pos, y_pos))
    
    return grid


# ===== CLIP Embedding Extraction Functions =====

@torch.inference_mode()
def extract_clip_embeddings_from_pil(pil_image: Union[Image.Image, List[Image.Image]], 
                                    ip_model) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: CLIP embeddings of shape (batch_size, seq_len, embed_dim)
    """
    if isinstance(pil_image, Image.Image):
        pil_image = [pil_image]
    
    # Process images through CLIP processor
    processed_images = ip_model.clip_image_processor(
        images=pil_image, return_tensors="pt"
    ).pixel_values
    
    # Move to model device with appropriate dtype
    processed_images = processed_images.to(ip_model.device, dtype=torch.float16)
    
    # Extract embeddings from penultimate layer (better for downstream tasks)
    clip_embeddings = ip_model.image_encoder(
        processed_images, output_hidden_states=True
    ).hidden_states[-2]
    
    # Convert to float32 for better numerical stability
    return clip_embeddings.float()


@torch.inference_mode()
def extract_clip_embeddings_from_pil_batch(pil_images: List[Image.Image], 
                                          ip_model) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: Concatenated CLIP embeddings of shape (batch, seq_len, embed_dim)
    """
    embeddings_batch = []
    
    for image in pil_images:
        embeddings = extract_clip_embeddings_from_pil(image, ip_model)
        embeddings_batch.append(embeddings)
    
    return torch.cat(embeddings_batch, dim=0)


@torch.inference_mode()
def extract_clip_embeddings_from_tensor(tensor_image: torch.Tensor, 
                                       ip_model, 
                                       resize: bool = True) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: CLIP embeddings of shape (batch_size, seq_len, embed_dim)
    """
    # Move tensor to model device with appropriate dtype
    tensor_image = tensor_image.to(ip_model.device, dtype=torch.float16)
    
    # Resize to CLIP input resolution if requested
    if resize:
        tensor_image = torch.nn.functional.interpolate(
            tensor_image, 
            size=(224, 224), 
            mode="bilinear", 
            align_corners=False
        )
    
    # Extract embeddings with positional encoding interpolation
    clip_embeddings = ip_model.image_encoder(
        tensor_image, 
        output_hidden_states=True, 
        interpolate_pos_encoding=True
    ).hidden_states[-2]
    
    # Convert to float32 for numerical stability
    return clip_embeddings.float()


# ===== IP-Adapter Helper Functions =====

@torch.inference_mode()
def _enhanced_get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    """
    Enhanced version of IP-Adapter's get_image_embeds method.
    
    This method processes either PIL images or pre-computed CLIP embeddings
    and returns both conditional and unconditional embeddings for generation.
    
    Args:
        pil_image: PIL Image(s) to process (optional)
        clip_image_embeds: Pre-computed CLIP embeddings (optional)
        
    Returns:
        Tuple of (conditional_embeds, unconditional_embeds)
    """
    # Process PIL images if provided
    if pil_image is not None:
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        
        # Convert PIL to tensor and extract CLIP embeddings
        processed_images = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        processed_images = processed_images.to(self.device, dtype=torch.float16)
        
        clip_image_embeds = self.image_encoder(
            processed_images, output_hidden_states=True
        ).hidden_states[-2]
    
    # Project CLIP embeddings to IP-Adapter space
    conditional_embeds = self.image_proj_model(clip_image_embeds)
    
    # Generate unconditional embeddings (for classifier-free guidance)
    zero_tensor = torch.zeros(1, 3, 224, 224).to(self.device, dtype=torch.float16)
    uncond_clip_embeds = self.image_encoder(
        zero_tensor, output_hidden_states=True
    ).hidden_states[-2]
    unconditional_embeds = self.image_proj_model(uncond_clip_embeds)
    
    return conditional_embeds, unconditional_embeds


# ===== Model Loading Functions =====

@torch.inference_mode()
def load_stable_diffusion_pipeline(device: str = "cuda") -> StableDiffusionPipeline:
    # Model paths
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"

    # Configure DDIM scheduler for high-quality sampling
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Load VAE separately for better quality
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    
    # Create Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,  # Disable safety checker for faster inference
        safety_checker=None,
    )
    
    return pipeline


@torch.inference_mode()
def load_ip_adapter_model(device: str = "cuda", sd_only: bool = False) -> IPAdapterPlus:
    # Model and checkpoint paths
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "./downloads/models/image_encoder"
    ip_checkpoint_path = "./downloads/models/ip-adapter-plus_sd15.bin"

    # Configure DDIM scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Load high-quality VAE
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    
    # Create base Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )
    
    if sd_only:
        return pipeline
    
    # Initialize IP-Adapter with 16 tokens for better image conditioning
    ip_model = IPAdapterPlus(
        pipeline, 
        image_encoder_path, 
        ip_checkpoint_path, 
        device, 
        num_tokens=16
    )

    # Enhance the model with our improved get_image_embeds method
    setattr(ip_model.__class__, "get_image_embeds", _enhanced_get_image_embeds)
    
    return ip_model


def load_ip_adapter_xl_model(device: str = "cuda") -> IPAdapterPlusXL:
    base_model_path = "SG161222/RealVisXL_V1.0"
    image_encoder_path = "./downloads/models/image_encoder"
    ip_ckpt = "./downloads/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    return ip_model

def load_ipadapter(version: str = "sd15", device: str = "cuda") -> IPAdapterPlus | IPAdapterPlusXL:
    if version == "sd15":
        return load_ip_adapter_model(device)
    elif version == "sdxl":
        return load_ip_adapter_xl_model(device)
    else:
        raise ValueError(f"Invalid version: {version}")


# ===== Image Generation Functions =====

@torch.inference_mode()
def generate_images_from_clip_embeddings(ip_model : IPAdapterPlus,
                                       clip_embeddings: torch.Tensor,
                                       num_samples: int = 4, 
                                       num_inference_steps: int = 50, 
                                       seed: Optional[int] = 42) -> List[Image.Image]:
    """Generate images from CLIP embeddings using IP-Adapter.
    clip_embeddings is (batch, seq_len, embed_dim)
    """
    # Ensure embeddings have correct shape and dtype
    if clip_embeddings.ndim == 2:
        clip_embeddings = clip_embeddings.unsqueeze(0)
    
    if clip_embeddings.ndim != 3:
        raise ValueError(f"Expected 3D embeddings (batch, seq, dim), got {clip_embeddings.shape}")
    
    # Move to appropriate device and dtype
    clip_embeddings = clip_embeddings.half().to(ip_model.device)
    
    # Generate images using IP-Adapter
    negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    generated_images = ip_model.generate(
        clip_image_embeds=clip_embeddings,
        negative_prompt=negative_prompt,
        pil_image=None,
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    return generated_images


# ===== Legacy Function Aliases =====

# Maintain backward compatibility with existing code
image_grid = create_image_grid
extract_clip_embedding_pil = extract_clip_embeddings_from_pil
extract_clip_embedding_pil_batch = extract_clip_embeddings_from_pil_batch
extract_clip_embedding_tensor = extract_clip_embeddings_from_tensor
load_sdxl = load_stable_diffusion_pipeline
generate = generate_images_from_clip_embeddings