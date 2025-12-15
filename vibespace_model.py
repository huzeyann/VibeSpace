"""
Neural Compression Model for Feature Space Learning

This module implements a compression model that learns to compress and decompress
image features while preserving their geometric and semantic properties using
normalized cuts (NCut).
"""

import gc
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from omegaconf import DictConfig
import gradio as gr

from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut
from ncut_pytorch.utils.math import rbf_affinity


def compute_ncut_eigenvectors(features: torch.Tensor, n_eig: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = features.var(0).sum().item()
    affinity_matrix = rbf_affinity(features, gamma=gamma)
    eigenvectors, eigenvalues = _plain_ncut(affinity_matrix, n_eig)
    return eigenvectors, eigenvalues


# ===== Neural Network Components =====

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, hidden_dim: int = 4096):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class SpatialPoolingAvgPool(nn.Module):
    """
    AvgPool layer for spatial pooling of feature maps with support for sequence inputs.
    
    Handles inputs with CLS tokens and reshapes appropriately for 2D convolution.
    """
    def __init__(self, downsample_factor: int = 2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.avg_pool = nn.AvgPool2d(downsample_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both (batch, seq_len, channels) and (seq_len, channels) inputs.
        """
        # Handle input shape variations
        added_batch_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch_dim = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, L, C) or (L, C), got {x.shape}")

        batch_size, seq_len, channels = x.shape
        
        if seq_len < 2:
            raise ValueError("Sequence length must be at least 2 (1 CLS token + 1 patch)")

        # Validate that seq_len-1 is a perfect square (for spatial arrangement)
        spatial_size = int(round((seq_len - 1) ** 0.5))
        if spatial_size * spatial_size != (seq_len - 1):
            raise ValueError(f"seq_len-1 must be perfect square. Got {seq_len-1}")

        # Separate CLS token and spatial features
        cls_tokens = x[:, :1, :]  # (B, 1, C)
        spatial_features = x[:, 1:, :]  # (B, H*W, C)
        
        # Reshape to 2D for convolution
        spatial_2d = rearrange(
            spatial_features, 'b (h w) c -> b c h w', 
            h=spatial_size, w=spatial_size
        )
        
        # Apply pooling
        pooled_features = self.avg_pool(spatial_2d)
        
        # Reshape back to sequence format
        pooled_sequence = rearrange(pooled_features, 'b c h w -> b (h w) c')
        
        # Concatenate CLS token back
        output = torch.cat([cls_tokens, pooled_sequence], dim=1)

        # Remove batch dimension if it was added
        if added_batch_dim:
            output = output.squeeze(0)
        
        return output

class MLPWithSpatialPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, 
                 hidden_dim: int = 4096, downsample_factor: int = 2):
        super().__init__()
        
        self.pooling = SpatialPoolingAvgPool(downsample_factor)
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        return self.network(x)


# ===== Main Compression Model =====

class VibeSpaceModel(pl.LightningModule):
    """
    Neural compression model for learning compressed feature representations.
    
    This model compresses input features to a lower-dimensional "vibe space" and
    then decompresses them back, while preserving geometric and semantic properties
    through various loss functions including NCut-based losses.
    """
    
    def __init__(self, config: DictConfig, enable_gradio_progress: bool = False, downsample_factor: int = 2):
        super().__init__()
        
        self.config = config
        self.downsample_factor = downsample_factor
        
        self.encoder = MultiLayerPerceptron(
            config.in_dim, config.vibe_dim, config.n_layer, config.latent_dim
        )
        
        self.decoder = MLPWithSpatialPooling(
            config.vibe_dim, config.out_dim, config.n_layer, 
            config.latent_dim, self.downsample_factor
        )
        
        self.loss_history = defaultdict(list)
        self.enable_gradio_progress = enable_gradio_progress
        if enable_gradio_progress:
            self.progress_tracker = gr.Progress()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def training_step(self, batch, batch_idx):
        # Update progress bar if using Gradio
        if (self.enable_gradio_progress and 
            self.trainer.global_step % 10 == 0 and 
            self.trainer.global_step > 0 and
            self.loss_history['recon']):
            
            progress = self.trainer.global_step / self.config.steps
            recent_loss = self.loss_history['recon'][-1]
            self.progress_tracker(progress, desc=f"Training Vibe Space, loss = {recent_loss:.4f}")

        positive_features, negative_features, target_features, negative_mask = batch
        negative_mask = negative_mask.bool()
        has_negatives = bool(negative_mask.any().item())

        if has_negatives:
            if bool(negative_mask.all().item()):
                batch_negative_features = negative_features
            else:
                batch_negative_features = negative_features[negative_mask]
        else:
            batch_negative_features = None
        
        compressed_features = self.encoder(positive_features)
        reconstructed_features = self.decoder(compressed_features)
        
        
        total_loss = self._compute_total_loss(
            positive_features,
            batch_negative_features,
            target_features,
            compressed_features,
            reconstructed_features,
        )
        
        self.log("loss/total", total_loss, prog_bar=True)
        return total_loss
    
    def _compute_ncut_eigenvectors(self, features: torch.Tensor) -> torch.Tensor:
        """Compute NCut eigenvectors for features."""
        # Accept inputs shaped either (batch, length, channels) or (length, channels)
        flattened_features = features
        if flattened_features.dim() >= 3:
            flattened_features = flattened_features.flatten(0, 1)
        elif flattened_features.dim() == 1:
            # rbf_affinity expects at least 2D; treat single vector as one sample with channels
            flattened_features = flattened_features.unsqueeze(0)

        if flattened_features.numel() > 0 and flattened_features.dim() == 2:
            eigenvectors, _ = compute_ncut_eigenvectors(flattened_features, self.config.n_eig)
            return eigenvectors
        else:
            # Return zero tensor if no features
            device = features.device if isinstance(features, torch.Tensor) else 'cpu'
            return torch.zeros((1, self.config.n_eig), device=device)
    
    def _compute_multiscale_similarity(self, eigenvectors: torch.Tensor, 
                                      start_n_eig: int = 4, step_mult: int = 2) -> torch.Tensor:
        """Compute multi-scale similarity matrix from eigenvectors.
        eigenvectors is (batch*length, n_eig)
        """
        total_similarity = 0.0
        num_scales = 0
        max_available = eigenvectors.shape[1]
        current_n_eig = min(start_n_eig, max_available)
        
        if self.config.single_scale_flag:
            current_n_eig = max_available
        
        while current_n_eig <= max_available:
            eigvec_subset = eigenvectors[:, :current_n_eig]
            eigvec_normalized = F.normalize(eigvec_subset, dim=-1)
            
            total_similarity += eigvec_normalized @ eigvec_normalized.T
            
            num_scales += 1
            current_n_eig *= step_mult
        
        return total_similarity / num_scales if num_scales > 0 else total_similarity
    
    def _compute_flag_decoder_loss(
        self,
        compressed_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        negative_input_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        compressed_features is (batch, length, channels)
        reconstructed_features is (batch, length, channels)
        """
        pooled_compressed = self.decoder.pooling(compressed_features)
        pooled_compressed = pooled_compressed.flatten(0, 1)
        reconstructed_features = reconstructed_features.flatten(0, 1)

        has_negative = (
            negative_input_features is not None and negative_input_features.numel() > 0
        )

        # sample points from the compressed feature space (only when no negatives available)
        dim_mins = pooled_compressed.min(0).values
        dim_maxs = pooled_compressed.max(0).values
        dim_mins -= 0.25 * (dim_maxs - dim_mins) * torch.rand_like(dim_mins)
        dim_maxs += 0.25 * (dim_maxs - dim_mins) * torch.rand_like(dim_maxs)
        
        num_samples = 0 if has_negative else self.config.n_negative_sample
        sample_points = torch.rand(num_samples, pooled_compressed.shape[1], device=pooled_compressed.device)
        sample_points = sample_points * (dim_maxs - dim_mins) + dim_mins
        
        # reconstruct the sample points
        sample_reconstructed = self.decoder.network(sample_points)
        
        all_compressed = torch.cat([pooled_compressed, sample_points], dim=0)
        all_reconstructed = torch.cat([reconstructed_features, sample_reconstructed], dim=0)
        
        # flag loss on the sample points 
        similarity = all_compressed @ all_compressed.T
        eigenvectors_pos, _ = compute_ncut_eigenvectors(all_reconstructed, self.config.n_eig)

        if has_negative and self.config.get('do_decoder_negative_flag', False):
            negative_compressed = self.encoder(negative_input_features)
            negative_reconstructed = self.decoder(negative_compressed)
            negative_reconstructed = negative_reconstructed.flatten(0, 1)

            neg_eigenvectors, _ = compute_ncut_eigenvectors(negative_reconstructed, self.config.n_eig)

            max_available = min(eigenvectors_pos.shape[1], neg_eigenvectors.shape[1])
            if max_available == 0:
                eig_similarity = self._compute_multiscale_similarity(eigenvectors_pos)
            else:
                if self.config.single_scale_flag:
                    current_n_eig = max_available
                else:
                    current_n_eig = min(self.config.get('start_n_eig', 4), max_available)
                    current_n_eig = max(current_n_eig, 1)

                total_filtered_similarity = similarity.new_zeros(similarity.shape)
                num_scales = 0
                beta = self.config.get('decoder_negative_beta', self.config.get('negative_beta', 1.0))
                step_mult = self.config.get('step_mult', 2)

                while current_n_eig <= max_available:
                    P = eigenvectors_pos[:, :current_n_eig]
                    N = neg_eigenvectors[:, :current_n_eig]

                    N_norm = F.normalize(N, dim=0)
                    projection = torch.matmul(N_norm.T, P)
                    P_filtered = P - beta * torch.matmul(N_norm, projection)

                    P_filtered_norm = F.normalize(P_filtered, dim=-1)
                    total_filtered_similarity += P_filtered_norm @ P_filtered_norm.T

                    num_scales += 1
                    current_n_eig *= step_mult

                if num_scales > 0:
                    eig_similarity = total_filtered_similarity / num_scales
                else:
                    eig_similarity = self._compute_multiscale_similarity(eigenvectors_pos)
        else:
            eig_similarity = self._compute_multiscale_similarity(eigenvectors_pos)

        loss = F.smooth_l1_loss(eig_similarity, similarity)
        return loss
    
    def _compute_flag_encoder_loss(self, input_features: torch.Tensor, compressed_features: torch.Tensor) -> torch.Tensor:
        """
        input_features is (batch, length, channels)
        compressed_features is (batch, length, channels)
        """
        sample_indices = torch.randperm(input_features.shape[0])[:self.config.n_sample_eigsolve]
        gt_eigenvectors = self._compute_ncut_eigenvectors(input_features.flatten(0, 1)[sample_indices])
        gt_similarity = self._compute_multiscale_similarity(gt_eigenvectors)
        flattened_compressed = compressed_features.flatten(0, 1)[sample_indices]
        pred_similarity = flattened_compressed @ flattened_compressed.T
        loss = F.smooth_l1_loss(gt_similarity, pred_similarity)
        return loss
    
    def _compute_total_loss(
        self,
        positive_features: torch.Tensor,
        negative_features: Optional[torch.Tensor],
        target_features: torch.Tensor,
        compressed_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        positive_features is (batch, length, channels)
        target_features is (batch, length, channels)
        compressed_features is (batch, length, channels)
        reconstructed_features is (batch, length, channels)
        """
        total_loss = positive_features.new_tensor(0.0)
        has_negative_features = (
            negative_features is not None and negative_features.numel() > 0
        )
        beta = self.config.get('negative_beta', 1.0)
        
        # Flag encoder loss - guide the structure from encoder to compressed features
        if self.config.flag_encoder_loss > 0 and has_negative_features:
            gt_eigenvectors_pos = self._compute_ncut_eigenvectors(positive_features)
            gt_eigenvectors_neg = self._compute_ncut_eigenvectors(negative_features)

            total_filtered_similarity = 0.0
            num_scales = 0
            max_available = min(gt_eigenvectors_pos.shape[1], gt_eigenvectors_neg.shape[1])

            if max_available == 0:
                gt_similarity = self._compute_multiscale_similarity(gt_eigenvectors_pos)
            else:
                if self.config.single_scale_flag:
                    current_n_eig = max_available
                else:
                    current_n_eig = min(self.config.get('start_n_eig', 4), max_available)
                    current_n_eig = max(current_n_eig, 1)

                step_mult = self.config.get('step_mult', 2)
                while current_n_eig <= max_available and current_n_eig > 0:
                    P = gt_eigenvectors_pos[:, :current_n_eig]
                    N = gt_eigenvectors_neg[:, :current_n_eig]

                    N_norm = F.normalize(N, dim=0)
                    projection = torch.matmul(N_norm.T, P)
                    P_filtered = P - beta * torch.matmul(N_norm, projection)

                    P_filtered_norm = F.normalize(P_filtered, dim=-1)
                    total_filtered_similarity += P_filtered_norm @ P_filtered_norm.T

                    num_scales += 1
                    current_n_eig *= step_mult

                if num_scales > 0:
                    gt_similarity = total_filtered_similarity / num_scales
                else:
                    gt_similarity = self._compute_multiscale_similarity(gt_eigenvectors_pos)
            flattened_compressed = compressed_features.flatten(0, 1)
            pred_similarity = flattened_compressed @ flattened_compressed.T

            flag_encoder_loss = F.smooth_l1_loss(gt_similarity, pred_similarity)
            self.log("loss/flag_encoder", flag_encoder_loss, prog_bar=True)
            total_loss += flag_encoder_loss * self.config.flag_encoder_loss
            self.loss_history['flag_encoder'].append(flag_encoder_loss.item())
        elif self.config.flag_encoder_loss > 0:
            flag_encoder_loss = self._compute_flag_encoder_loss(positive_features, compressed_features)
            self.log("loss/flag_encoder", flag_encoder_loss, prog_bar=True)
            total_loss += flag_encoder_loss * self.config.flag_encoder_loss
            self.loss_history['flag_encoder'].append(flag_encoder_loss.item())
        
        # Flag decoder loss - guide the structure from compressed to decoded features
        if self.config.flag_decoder_loss > 0:
            if self.trainer.global_step >= 500:  # warmup period
                flag_decoder_loss = self._compute_flag_decoder_loss(
                    compressed_features,
                    reconstructed_features,
                    negative_features,
                )
                self.log("loss/flag_decoder", flag_decoder_loss, prog_bar=True)
                total_loss += flag_decoder_loss * self.config.flag_decoder_loss
                self.loss_history['flag_decoder'].append(flag_decoder_loss.item())

        # Reconstruction loss
        if self.config.recon_loss > 0:
            recon_loss = F.smooth_l1_loss(target_features, reconstructed_features)
            self.log("loss/recon", recon_loss, prog_bar=True)
            total_loss += recon_loss * self.config.recon_loss
            self.loss_history['recon'].append(recon_loss.item())

        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self.config.lr)


# ===== Dataset and Training Utilities =====

class FeatureDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        positive_features: torch.Tensor,
        target_features: torch.Tensor,
        negative_features: Optional[torch.Tensor] = None,
    ):
        self.positive_features = positive_features
        self.target_features = target_features
        if negative_features is not None and negative_features.numel() > 0:
            self.negative_features = negative_features
        else:
            self.negative_features = None
    
    def __len__(self) -> int:
        return len(self.positive_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive = self.positive_features[idx]
        target = self.target_features[idx]

        if self.negative_features is None:
            negative = torch.zeros_like(positive)
            has_negative = torch.tensor(False, dtype=torch.bool)
        else:
            neg_idx = torch.randint(0, self.negative_features.shape[0], (1,)).item()
            negative = self.negative_features[neg_idx]
            has_negative = torch.tensor(True, dtype=torch.bool)

        return positive, negative, target, has_negative


def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def train_vibe_space(model: VibeSpaceModel, 
                          config: DictConfig,
                          input_features: torch.Tensor,
                          target_features: torch.Tensor,
                          negative_features: Optional[torch.Tensor] = None,
                          devices: List[int] = [0]) -> pl.Trainer:
    clear_gpu_memory()
    dataset = FeatureDataset(input_features, target_features, negative_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    trainer = pl.Trainer(
        max_steps=config.steps,
        gradient_clip_val=1.0,
        accelerator="gpu", 
        devices=devices,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False  # Disable default logger
    )
    
    trainer.fit(model, dataloader)
    
    return trainer