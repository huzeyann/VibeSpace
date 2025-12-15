"""
DINO Correspondence Analysis Module

This module provides functions for analyzing visual correspondences between images
using DINO features, normalized cuts (NCut), and clustering techniques.
"""

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from einops import rearrange

from .extract_features import image_inverse_transform
from .ipadapter_model import image_grid
from ncut_pytorch import ncut_fn, kway_ncut, convert_to_lab_color
from ncut_pytorch.color import tsne_color
from ncut_pytorch.utils.gamma import find_gamma_by_degree


# ===== Core NCut and Clustering Functions =====

def ncut_tsne_multiple_images(image_embeds, n_eig=50, gamma=None, degree=0.5):
    """
    Apply NCut and t-SNE coloring to multiple image embeddings.
    
    image_embeds is (batch, length, channels)
    """
    batch_size, length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    
    eigenvectors, eigenvalues = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    rgb_colors = tsne_color(eigenvectors, n_dim=3, device='cuda', perplexity=50)
    rgb_colors = convert_to_lab_color(rgb_colors)
    
    # Reshape back to original batch structure
    rgb_colors = rearrange(rgb_colors, '(b l) c -> b l c', b=batch_size)
    eigenvectors = rearrange(eigenvectors, '(b l) c -> b l c', b=batch_size)
    
    return eigenvectors, rgb_colors


def _kway_cluster_single_image(image_embeds, n_clusters, gamma=None, degree=0.5):
    length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    else:
        gamma = gamma * image_embeds.var(0).sum().item()
    
    # Calculate number of eigenvectors needed
    n_eig = min(n_clusters * 2 + 6, flattened_input.shape[0] // 2 - 1)
    
    eigenvectors, _ = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    continuous_clusters = kway_ncut(eigenvectors[:, :n_clusters])
    return continuous_clusters


def kway_cluster_per_image(image_embeds, n_clusters, gamma=None, degree=0.5):
    """
    Perform k-way clustering on each image separately.
    
    image_embeds is (batch, length, channels)
    return (batch, length, clusters)
    """
    clustered_eigenvectors = []
    
    for i in range(image_embeds.shape[0]):
        eigenvector = _kway_cluster_single_image(
            image_embeds[i], n_clusters, gamma, degree
        )
        clustered_eigenvectors.append(eigenvector)
    
    return torch.stack(clustered_eigenvectors)


def kway_cluster_multiple_images(image_embeds, n_clusters, gamma=None, degree=0.5):
    """
    Perform k-way clustering on multiple images jointly.
    
    image_embeds is (batch, length, channels)
    return (batch, length, clusters)
    """
    batch_size, length, channels = image_embeds.shape
    flattened_input = image_embeds.flatten(end_dim=-2)
    
    if gamma is None:
        gamma = find_gamma_by_degree(flattened_input, degree)
    
    # Calculate number of eigenvectors needed
    n_eig = min(n_clusters * 2 + 6, flattened_input.shape[0] // 2 - 1)
    
    eigenvectors, _ = ncut_fn(
        flattened_input, n_eig=n_eig, gamma=gamma, device='cuda'
    )
    
    continuous_clusters = kway_ncut(eigenvectors[:, :n_clusters])
    continuous_clusters = rearrange(
        continuous_clusters, '(b l) c -> b l c', b=batch_size
    )
    
    return continuous_clusters


# ===== Color and Visualization Functions =====

def get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors):

    n_clusters = cluster_eigenvectors.shape[-1]
    discrete_colors = np.zeros_like(joint_colors)
    
    for img_idx in range(joint_colors.shape[0]):
        colors = joint_colors[img_idx]
        eigenvector = cluster_eigenvectors[img_idx].cpu().numpy()
        cluster_labels = eigenvector.argmax(-1)
        discrete_img_colors = np.zeros_like(colors)
        
        for cluster_idx in range(n_clusters):
            cluster_mask = cluster_labels == cluster_idx
            if cluster_mask.sum() > 0:
                # Use mean color for each cluster
                discrete_img_colors[cluster_mask] = colors[cluster_mask].mean(0)
        
        discrete_colors[img_idx] = discrete_img_colors
    
    # Convert to uint8 format
    discrete_colors = (discrete_colors * 255).astype(np.uint8)
    return discrete_colors


# ===== Center Matching Functions =====

def get_cluster_center_features(image_embeds, cluster_labels, n_clusters):

    center_features = torch.zeros((n_clusters, image_embeds.shape[-1]))
    
    for cluster_idx in range(n_clusters):
        cluster_mask = cluster_labels == cluster_idx
        
        if cluster_mask.sum() > 0:
            center_features[cluster_idx] = image_embeds[cluster_mask].mean(0)
        else:
            # Use a unique identifier for empty clusters
            center_features[cluster_idx] = torch.ones_like(image_embeds[0]) * 114514
    
    return center_features


def cosine_similarity(matrix_a, matrix_b):
    normalized_a = matrix_a / matrix_a.norm(dim=-1, keepdim=True)
    normalized_b = matrix_b / matrix_b.norm(dim=-1, keepdim=True)
    return normalized_a @ normalized_b.T


def hungarian_match_centers(center_features1, center_features2):
    distances = torch.cdist(center_features1, center_features2)
    distances = distances.cpu().detach().numpy()
    _, column_indices = linear_sum_assignment(distances)
    return column_indices


def argmin_matching(center_features1, center_features2):
    distances = torch.cdist(center_features1, center_features2)
    distances = distances.cpu().detach().numpy()
    return np.argmin(distances, axis=-1)


def match_cluster_centers(image_embed1, image_embed2, eigvec1, eigvec2, 
                         match_method='hungarian'):
    cluster_labels1 = eigvec1.argmax(-1).cpu().numpy()
    cluster_labels2 = eigvec2.argmax(-1).cpu().numpy()
    
    center_features1 = get_cluster_center_features(
        image_embed1, cluster_labels1, eigvec1.shape[-1]
    )
    center_features2 = get_cluster_center_features(
        image_embed2, cluster_labels2, eigvec2.shape[-1]
    )
    
    if match_method == 'hungarian':
        mapping = hungarian_match_centers(center_features1, center_features2)
    elif match_method == 'argmin':
        mapping = argmin_matching(center_features1, center_features2)
    else:
        raise ValueError(f"Unknown match_method: {match_method}")
    
    return mapping


def match_centers_three_images(image_embeds, eigenvectors, match_method='hungarian'):
    """
    Match cluster centers across three images (A2 -> A1 -> B1).
    
    Args:
        image_embeds (torch.Tensor): Embeddings for 3 images [A2, A1, B1]
        eigenvectors (torch.Tensor): Eigenvectors for 3 images
        match_method (str): Matching method
        
    Returns:
        tuple: (A2_to_A1_mapping, A1_to_B1_mapping)
    """
    a2_to_a1_mapping = match_cluster_centers(
        image_embeds[0], image_embeds[1], 
        eigenvectors[0], eigenvectors[1], 
        match_method=match_method
    )
    
    a1_to_b1_mapping = match_cluster_centers(
        image_embeds[1], image_embeds[2], 
        eigenvectors[1], eigenvectors[2], 
        match_method=match_method
    )
    
    return a2_to_a1_mapping, a1_to_b1_mapping


def match_centers_two_images(image_embed1, image_embed2, eigvec1, eigvec2, 
                            match_method='hungarian'):
    return match_cluster_centers(
        image_embed1, image_embed2, eigvec1, eigvec2, match_method=match_method
    )


# ===== Two-Step Clustering Functions =====

def kway_cluster_per_image_two_step(
    image_embeds,
    n_superclusters,
    n_subclusters_per_supercluster,
    supercluster_gamma=None,
    subcluster_gamma=None,
    degree=0.5
):
    """
    Perform 2-step hierarchical clustering on each image separately.
    First finds superclusters, then subdivides each supercluster into subclusters.
    
    Args:
        image_embeds: (batch, length, channels) - Image embeddings
        n_superclusters: Number of coarse superclusters to find
        n_subclusters_per_supercluster: Number of subclusters within each supercluster
        supercluster_gamma: Gamma parameter for supercluster NCut (None = auto)
        subcluster_gamma: Gamma parameter for subcluster NCut (None = auto)
        degree: Degree parameter for gamma estimation
        
    Returns:
        tuple: (supercluster_eigenvectors, subcluster_eigenvectors, subcluster_to_supercluster_mapping)
            - supercluster_eigenvectors: (batch, length, n_superclusters)
            - subcluster_eigenvectors: (batch, length, total_subclusters)
            - subcluster_to_supercluster_mapping: (batch, total_subclusters) mapping each subcluster to its supercluster
    """
    batch_size = image_embeds.shape[0]
    
    # Step 1: Compute superclusters for each image
    supercluster_eigenvectors = []
    for i in range(batch_size):
        eigenvector = _kway_cluster_single_image(
            image_embeds[i], n_superclusters, supercluster_gamma, degree
        )
        supercluster_eigenvectors.append(eigenvector)
    supercluster_eigenvectors = torch.stack(supercluster_eigenvectors)
    
    # Step 2: For each supercluster in each image, compute subclusters
    subcluster_eigenvectors = []
    subcluster_to_supercluster_mapping = []
    
    for img_idx in range(batch_size):
        img_subclusters = []
        img_mapping = []
        
        supercluster_labels = supercluster_eigenvectors[img_idx].argmax(-1)
        
        # For each supercluster, extract tokens and compute subclusters
        for supercluster_idx in range(n_superclusters):
            supercluster_mask = supercluster_labels == supercluster_idx
            
            if supercluster_mask.sum() == 0:
                # Empty supercluster - create dummy subclusters
                for sub_idx in range(n_subclusters_per_supercluster):
                    img_mapping.append(supercluster_idx)
                continue
            
            # Extract features belonging to this supercluster
            supercluster_features = image_embeds[img_idx][supercluster_mask]
            
            # Perform clustering on this subset
            if supercluster_features.shape[0] <= n_subclusters_per_supercluster:
                # Too few tokens - each token becomes its own subcluster
                n_actual_subclusters = supercluster_features.shape[0]
                subcluster_labels = torch.arange(n_actual_subclusters).to(supercluster_features.device)
                # Pad with dummy subclusters if needed
                for sub_idx in range(n_subclusters_per_supercluster):
                    img_mapping.append(supercluster_idx)
            else:
                # Perform subclustering
                subcluster_eigvecs = _kway_cluster_single_image(
                    supercluster_features, 
                    n_subclusters_per_supercluster,
                    subcluster_gamma,
                    degree
                )
                subcluster_labels = subcluster_eigvecs.argmax(-1)
                
                # Track which supercluster these subclusters belong to
                for sub_idx in range(n_subclusters_per_supercluster):
                    img_mapping.append(supercluster_idx)
            
            # Store subcluster assignments for this supercluster
            for sub_idx in range(n_subclusters_per_supercluster):
                img_subclusters.append((supercluster_mask, subcluster_labels == sub_idx if supercluster_features.shape[0] > n_subclusters_per_supercluster else None))
        
        # Convert to full eigenvector representation
        total_subclusters = n_superclusters * n_subclusters_per_supercluster
        img_subcluster_eigvec = torch.zeros((image_embeds.shape[1], total_subclusters)).to(image_embeds.device)
        
        for subcluster_global_idx, (supercluster_mask, subcluster_mask) in enumerate(img_subclusters):
            if subcluster_mask is not None:
                # Combine masks: belongs to supercluster AND subcluster
                final_mask = torch.zeros(image_embeds.shape[1], dtype=torch.bool).to(image_embeds.device)
                supercluster_indices = torch.where(supercluster_mask)[0]
                subcluster_within_super = torch.where(subcluster_mask)[0]
                if len(subcluster_within_super) > 0:
                    final_indices = supercluster_indices[subcluster_within_super]
                    final_mask[final_indices] = True
                    img_subcluster_eigvec[final_mask, subcluster_global_idx] = 1.0
            # else: leave as zeros (empty subcluster)
        
        subcluster_eigenvectors.append(img_subcluster_eigvec)
        subcluster_to_supercluster_mapping.append(torch.tensor(img_mapping))
    
    subcluster_eigenvectors = torch.stack(subcluster_eigenvectors)
    subcluster_to_supercluster_mapping = torch.stack(subcluster_to_supercluster_mapping)
    
    return supercluster_eigenvectors, subcluster_eigenvectors, subcluster_to_supercluster_mapping


def match_centers_two_step(
    image_embed1,
    image_embed2,
    supercluster_eigvec1,
    supercluster_eigvec2,
    subcluster_eigvec1,
    subcluster_eigvec2,
    subcluster_to_supercluster_mapping1,
    subcluster_to_supercluster_mapping2,
    supercluster_match_method='hungarian',
    subcluster_match_method='hungarian'
):
    """
    Match clusters using 2-step hierarchical approach.
    First matches superclusters, then matches subclusters only within matched superclusters.
    
    Args:
        image_embed1, image_embed2: Image embeddings (length, channels)
        supercluster_eigvec1, supercluster_eigvec2: Supercluster eigenvectors (length, n_superclusters)
        subcluster_eigvec1, subcluster_eigvec2: Subcluster eigenvectors (length, total_subclusters)
        subcluster_to_supercluster_mapping1, subcluster_to_supercluster_mapping2: (total_subclusters,)
        supercluster_match_method: Matching method for superclusters
        subcluster_match_method: Matching method for subclusters
        
    Returns:
        np.ndarray: Mapping from image1 subclusters to image2 subclusters
    """
    n_superclusters = supercluster_eigvec1.shape[-1]
    n_subclusters_total = subcluster_eigvec1.shape[-1]
    
    # Step 1: Match superclusters
    supercluster_mapping = match_cluster_centers(
        image_embed1, image_embed2,
        supercluster_eigvec1, supercluster_eigvec2,
        match_method=supercluster_match_method
    )
    
    # Step 2: For each matched supercluster pair, match subclusters within them
    subcluster_mapping = np.zeros(n_subclusters_total, dtype=np.int64)
    
    for supercluster1_idx in range(n_superclusters):
        # Find which supercluster in image2 this maps to
        supercluster2_idx = supercluster_mapping[supercluster1_idx]
        
        # Find all subclusters belonging to these superclusters
        subclusters1_mask = (subcluster_to_supercluster_mapping1 == supercluster1_idx).cpu().numpy()
        subclusters2_mask = (subcluster_to_supercluster_mapping2 == supercluster2_idx).cpu().numpy()
        
        subclusters1_indices = np.where(subclusters1_mask)[0]
        subclusters2_indices = np.where(subclusters2_mask)[0]
        
        if len(subclusters1_indices) == 0 or len(subclusters2_indices) == 0:
            # No subclusters in one or both superclusters - use identity mapping
            for sub1_idx in subclusters1_indices:
                if sub1_idx < len(subclusters2_indices):
                    subcluster_mapping[sub1_idx] = subclusters2_indices[sub1_idx]
                else:
                    subcluster_mapping[sub1_idx] = subclusters2_indices[0] if len(subclusters2_indices) > 0 else 0
            continue
        
        # Extract subcluster eigenvectors for matching
        sub_eigvec1 = subcluster_eigvec1[:, subclusters1_indices]
        sub_eigvec2 = subcluster_eigvec2[:, subclusters2_indices]
        
        # Compute cluster centers for these subclusters
        cluster_labels1 = sub_eigvec1.argmax(-1).cpu()
        cluster_labels2 = sub_eigvec2.argmax(-1).cpu()
        
        center_features1 = get_cluster_center_features(
            image_embed1, cluster_labels1, len(subclusters1_indices)
        )
        center_features2 = get_cluster_center_features(
            image_embed2, cluster_labels2, len(subclusters2_indices)
        )
        
        # Match subclusters within this supercluster pair
        if subcluster_match_method == 'hungarian':
            local_mapping = hungarian_match_centers(center_features1, center_features2)
        elif subcluster_match_method == 'argmin':
            local_mapping = argmin_matching(center_features1, center_features2)
        else:
            raise ValueError(f"Unknown subcluster_match_method: {subcluster_match_method}")
        
        # Convert local mapping to global subcluster indices
        for local_idx, global_idx1 in enumerate(subclusters1_indices):
            global_idx2 = subclusters2_indices[local_mapping[local_idx]]
            subcluster_mapping[global_idx1] = global_idx2
    
    return subcluster_mapping


def kway_cluster_per_image_two_step_fgbg(
    image_embeds,
    n_foreground_subclusters,
    n_background_subclusters,
    supercluster_gamma=None,
    subcluster_gamma=None,
    degree=0.5
):
    """
    Perform 2-step hierarchical clustering with automatic foreground/background separation.
    First separates foreground (FG) and background (BG) using 2 clusters, identifying FG 
    by the cluster with highest max eigenvector value. Then subdivides FG and BG separately.
    
    Args:
        image_embeds: (batch, length, channels) - Image embeddings
        n_foreground_subclusters: Number of subclusters within foreground
        n_background_subclusters: Number of subclusters within background
        supercluster_gamma: Gamma parameter for FG/BG clustering (None = auto)
        subcluster_gamma: Gamma parameter for subcluster NCut (None = auto)
        degree: Degree parameter for gamma estimation
        
    Returns:
        tuple: (supercluster_eigenvectors, subcluster_eigenvectors, subcluster_to_supercluster_mapping, fg_indices)
            - supercluster_eigenvectors: (batch, length, 2) - [BG, FG] clusters
            - subcluster_eigenvectors: (batch, length, total_subclusters)
            - subcluster_to_supercluster_mapping: (batch, total_subclusters) - 0=BG, 1=FG
            - fg_indices: (batch,) - which supercluster index is foreground for each image
    """
    batch_size = image_embeds.shape[0]
    n_superclusters = 2  # Always FG and BG
    
    # Step 1: Compute FG/BG separation for each image
    supercluster_eigenvectors = []
    fg_indices = []
    
    for i in range(batch_size):
        eigenvector = _kway_cluster_single_image(
            image_embeds[i], n_clusters=2, gamma=supercluster_gamma, degree=degree
        )
        supercluster_eigenvectors.append(eigenvector)
        
        # Identify foreground: cluster with highest max eigenvector value
        fg_idx = eigenvector.max(0).values.argmax().item()
        fg_indices.append(fg_idx)
    
    supercluster_eigenvectors = torch.stack(supercluster_eigenvectors)
    fg_indices = torch.tensor(fg_indices)
    
    # Step 2: For each image, compute subclusters within FG and BG
    subcluster_eigenvectors = []
    subcluster_to_supercluster_mapping = []
    
    for img_idx in range(batch_size):
        img_subclusters = []
        img_mapping = []
        
        supercluster_labels = supercluster_eigenvectors[img_idx].argmax(-1)
        fg_idx = fg_indices[img_idx].item()
        bg_idx = 1 - fg_idx
        
        # Process BG and FG in order (BG first, then FG)
        for is_foreground, n_subclusters in [(False, n_background_subclusters), (True, n_foreground_subclusters)]:
            supercluster_idx = fg_idx if is_foreground else bg_idx
            supercluster_mask = supercluster_labels == supercluster_idx
            
            # Mark which supercluster type (0=BG, 1=FG)
            supercluster_type = 1 if is_foreground else 0
            
            if supercluster_mask.sum() == 0:
                # Empty supercluster - create dummy subclusters
                for sub_idx in range(n_subclusters):
                    img_mapping.append(supercluster_type)
                    img_subclusters.append((supercluster_mask, None))
                continue
            
            # Extract features belonging to this supercluster
            supercluster_features = image_embeds[img_idx][supercluster_mask]
            
            # Perform clustering on this subset
            if supercluster_features.shape[0] <= n_subclusters:
                # Too few tokens - each token becomes its own subcluster
                n_actual_subclusters = supercluster_features.shape[0]
                subcluster_labels = torch.arange(n_actual_subclusters).to(supercluster_features.device)
                # Pad with dummy subclusters if needed
                for sub_idx in range(n_subclusters):
                    img_mapping.append(supercluster_type)
                    if sub_idx < n_actual_subclusters:
                        img_subclusters.append((supercluster_mask, subcluster_labels == sub_idx))
                    else:
                        img_subclusters.append((supercluster_mask, None))
            else:
                # Perform subclustering
                subcluster_eigvecs = _kway_cluster_single_image(
                    supercluster_features, 
                    n_subclusters,
                    subcluster_gamma,
                    degree
                )
                subcluster_labels = subcluster_eigvecs.argmax(-1)
                
                # Store subcluster assignments
                for sub_idx in range(n_subclusters):
                    img_mapping.append(supercluster_type)
                    img_subclusters.append((supercluster_mask, subcluster_labels == sub_idx))
        
        # Convert to full eigenvector representation
        total_subclusters = n_background_subclusters + n_foreground_subclusters
        img_subcluster_eigvec = torch.zeros((image_embeds.shape[1], total_subclusters)).to(image_embeds.device)
        
        for subcluster_global_idx, (supercluster_mask, subcluster_mask) in enumerate(img_subclusters):
            if subcluster_mask is not None:
                # Combine masks: belongs to supercluster AND subcluster
                final_mask = torch.zeros(image_embeds.shape[1], dtype=torch.bool).to(image_embeds.device)
                supercluster_indices = torch.where(supercluster_mask)[0]
                subcluster_within_super = torch.where(subcluster_mask)[0]
                if len(subcluster_within_super) > 0:
                    final_indices = supercluster_indices[subcluster_within_super]
                    final_mask[final_indices] = True
                    img_subcluster_eigvec[final_mask, subcluster_global_idx] = 1.0
            # else: leave as zeros (empty subcluster)
        
        subcluster_eigenvectors.append(img_subcluster_eigvec)
        subcluster_to_supercluster_mapping.append(torch.tensor(img_mapping))
    
    subcluster_eigenvectors = torch.stack(subcluster_eigenvectors)
    subcluster_to_supercluster_mapping = torch.stack(subcluster_to_supercluster_mapping)
    
    return supercluster_eigenvectors, subcluster_eigenvectors, subcluster_to_supercluster_mapping, fg_indices


def match_centers_two_step_fgbg(
    image_embed1,
    image_embed2,
    subcluster_eigvec1,
    subcluster_eigvec2,
    subcluster_to_supercluster_mapping1,
    subcluster_to_supercluster_mapping2,
    n_background_subclusters,
    n_foreground_subclusters,
    background_match_method='hungarian',
    foreground_match_method='hungarian'
):
    """
    Match clusters using 2-step FG/BG hierarchical approach.
    FG and BG are automatically matched (no need for supercluster matching).
    Subclusters are matched within their respective FG or BG groups.
    
    Args:
        image_embed1, image_embed2: Image embeddings (length, channels)
        subcluster_eigvec1, subcluster_eigvec2: Subcluster eigenvectors (length, total_subclusters)
        subcluster_to_supercluster_mapping1, subcluster_to_supercluster_mapping2: (total_subclusters,) - 0=BG, 1=FG
        n_background_subclusters: Number of background subclusters
        n_foreground_subclusters: Number of foreground subclusters
        background_match_method: Matching method for background subclusters
        foreground_match_method: Matching method for foreground subclusters
        
    Returns:
        np.ndarray: Mapping from image1 subclusters to image2 subclusters
    """
    total_subclusters = n_background_subclusters + n_foreground_subclusters
    subcluster_mapping = np.zeros(total_subclusters, dtype=np.int64)
    
    # Process BG (supercluster_type=0) and FG (supercluster_type=1) separately
    for supercluster_type in [0, 1]:  # 0=BG, 1=FG
        # Find subclusters belonging to this supercluster type
        subclusters1_mask = (subcluster_to_supercluster_mapping1 == supercluster_type).cpu().numpy()
        subclusters2_mask = (subcluster_to_supercluster_mapping2 == supercluster_type).cpu().numpy()
        
        subclusters1_indices = np.where(subclusters1_mask)[0]
        subclusters2_indices = np.where(subclusters2_mask)[0]
        
        if len(subclusters1_indices) == 0 or len(subclusters2_indices) == 0:
            # No subclusters in one or both - use identity mapping
            for sub1_idx in subclusters1_indices:
                if sub1_idx < len(subclusters2_indices):
                    subcluster_mapping[sub1_idx] = subclusters2_indices[sub1_idx]
                else:
                    subcluster_mapping[sub1_idx] = subclusters2_indices[0] if len(subclusters2_indices) > 0 else 0
            continue
        
        # Extract subcluster eigenvectors for matching
        sub_eigvec1 = subcluster_eigvec1[:, subclusters1_indices]
        sub_eigvec2 = subcluster_eigvec2[:, subclusters2_indices]
        
        # Compute cluster centers for these subclusters
        cluster_labels1 = sub_eigvec1.argmax(-1).cpu()
        cluster_labels2 = sub_eigvec2.argmax(-1).cpu()
        
        center_features1 = get_cluster_center_features(
            image_embed1, cluster_labels1, len(subclusters1_indices)
        )
        center_features2 = get_cluster_center_features(
            image_embed2, cluster_labels2, len(subclusters2_indices)
        )
        
        # Match subclusters within this FG/BG group
        match_method = foreground_match_method if supercluster_type == 1 else background_match_method
        
        if match_method == 'hungarian':
            local_mapping = hungarian_match_centers(center_features1, center_features2)
        elif match_method == 'argmin':
            local_mapping = argmin_matching(center_features1, center_features2)
        else:
            raise ValueError(f"Unknown match_method: {match_method}")
        
        # Convert local mapping to global subcluster indices
        for local_idx, global_idx1 in enumerate(subclusters1_indices):
            global_idx2 = subclusters2_indices[local_mapping[local_idx]]
            subcluster_mapping[global_idx1] = global_idx2
    
    return subcluster_mapping


# ===== Visualization Functions =====

def plot_cluster_masks(image, eigenvector, cluster_order, hw=16):
    """
    blend the image with the cluster masks
    # image is (c, h, w)
    # eigenvector is (h*w, n_eig)
    # cluster_order is (n_eig), the order of the clusters
    """
    cluster_images = []
    base_img = image_inverse_transform(image).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    for cluster_idx in cluster_order:
        # Create cluster mask
        cluster_mask = eigenvector.argmax(-1) == cluster_idx
        mask_array = cluster_mask.cpu().numpy()[1:].reshape(hw, hw)
        mask_array = (mask_array * 255).astype(np.uint8)
        
        # Resize mask to match image
        mask_img = Image.fromarray(mask_array).resize(
            (128, 128), resample=Image.Resampling.NEAREST
        )
        
        # Apply mask to image
        mask_normalized = np.array(mask_img).astype(np.float32) / 255
        img_array = np.array(base_img).astype(np.float32) / 255
        
        # Create 3-channel mask and apply
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        mask_3ch[mask_3ch == 0] = 0.1  # Dim non-masked areas
        
        masked_img = img_array * mask_3ch
        masked_img = (masked_img * 255).astype(np.uint8)
        
        cluster_images.append(Image.fromarray(masked_img))
    
    return cluster_images


def create_image_grid_row(image, eigenvector, cluster_order, discrete_colors, 
                         hw=16, n_cols=10):

    cluster_images = plot_cluster_masks(image, eigenvector, cluster_order, hw)
    
    # Prepare base images
    base_img = image_inverse_transform(image).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    ncut_visualization = discrete_colors[1:].reshape(hw, hw, 3)
    ncut_img = Image.fromarray(ncut_visualization).resize(
        (128, 128), resample=Image.Resampling.NEAREST
    )
    
    # Pad cluster images to fill grid
    num_missing = n_cols - len(cluster_images) % n_cols
    if num_missing != n_cols:
        empty_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        cluster_images.extend([empty_img] * num_missing)
    
    # Create grid rows
    prepend_images = [base_img, ncut_img]
    n_rows = len(cluster_images) // n_cols
    grid_rows = []
    
    for row_idx in range(n_rows):
        start_idx = row_idx * n_cols
        end_idx = (row_idx + 1) * n_cols
        row_images = prepend_images + cluster_images[start_idx:end_idx]
        grid_rows.append(row_images)
    
    return grid_rows


def create_multi_image_grid(images, eigenvectors, cluster_orders, discrete_colors, 
                           hw=16, n_cols=10):
    all_grid_rows = []
    
    for image, eigvec, cluster_order, discrete_rgb in zip(
        images, eigenvectors, cluster_orders, discrete_colors
    ):
        grid_rows = create_image_grid_row(
            image, eigvec, cluster_order, discrete_rgb, hw, n_cols
        )
        all_grid_rows.append(grid_rows)
    
    # Interleave rows from different images
    interleaved_rows = []
    for row_idx in range(len(all_grid_rows[0])):
        for img_idx in range(len(all_grid_rows)):
            interleaved_rows.append(all_grid_rows[img_idx][row_idx])
    
    return interleaved_rows


def get_correspondence_plot(images, eigenvectors, cluster_orders, discrete_colors, 
                           hw=16, n_cols=10):
    n_clusters = eigenvectors.shape[-1]
    n_cols = min(n_cols, n_clusters)
    
    interleaved_rows = create_multi_image_grid(
        images, eigenvectors, cluster_orders, discrete_colors, hw, n_cols
    )
    
    n_rows = len(interleaved_rows)
    n_cols = len(interleaved_rows[0])
    
    # Flatten all images and create final grid
    all_images = sum(interleaved_rows, [])
    final_grid = image_grid(all_images, n_rows, n_cols)
    
    return final_grid