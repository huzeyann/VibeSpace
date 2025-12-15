"""
Intrinsic Dimensionality Estimation Module

This module provides utilities for estimating the intrinsic dimensionality of 
high-dimensional feature representations using Maximum Likelihood Estimation (MLE).
The intrinsic dimension represents the true underlying dimensionality of the data
manifold, which is often much lower than the ambient feature space dimension.
"""

import logging
from typing import Union, Optional

import numpy as np
import torch
import skdim
from ncut_pytorch.utils.sample import farthest_point_sampling


# ===== Constants =====

DEFAULT_MAX_SAMPLES = 2000
MIN_SAMPLES_REQUIRED = 10


# ===== Intrinsic Dimensionality Estimation =====

def estimate_intrinsic_dimension(features: Union[torch.Tensor, np.ndarray], 
                               max_samples: int = DEFAULT_MAX_SAMPLES,
                               use_global_estimation: bool = True) -> float:
    """
    Estimate the intrinsic dimensionality of feature representations.
    
    This function uses Maximum Likelihood Estimation (MLE) to determine the intrinsic
    dimensionality of high-dimensional features. If the dataset is large, it uses
    farthest point sampling to select a representative subset for efficient computation.
    
    Args:
        features (Union[torch.Tensor, np.ndarray]): Input features of any shape.
                                                   Will be flattened to (N, D) format.
        max_samples (int): Maximum number of samples to use for estimation.
                          Larger values give more accurate estimates but are slower.
        use_global_estimation (bool): Whether to prefer global over local estimation.
        
    Returns:
        float: Estimated intrinsic dimensionality of the feature manifold.
        
    Raises:
        ValueError: If input features are empty or have insufficient samples.
        RuntimeError: If dimensionality estimation fails completely.
        
    Example:
        >>> features = torch.randn(1000, 512)  # 1000 samples, 512-dim features
        >>> intrinsic_dim = estimate_intrinsic_dimension(features)
        >>> print(f"Intrinsic dimension: {intrinsic_dim:.2f}")
    """
    # Input validation
    if features is None:
        raise ValueError("Features cannot be None")
    
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        if features.numel() == 0:
            raise ValueError("Input tensor is empty")
        numpy_features = features.cpu().detach().numpy()
    else:
        numpy_features = np.asarray(features)
        if numpy_features.size == 0:
            raise ValueError("Input array is empty")
    
    # Reshape to 2D format (N_samples, N_features)
    original_shape = numpy_features.shape
    flattened_features = numpy_features.reshape(-1, numpy_features.shape[-1])
    
    n_samples, n_features = flattened_features.shape
    
    # Validate minimum requirements
    if n_samples < MIN_SAMPLES_REQUIRED:
        raise ValueError(
            f"Insufficient samples for dimensionality estimation. "
            f"Need at least {MIN_SAMPLES_REQUIRED}, got {n_samples}"
        )
    
    if n_features < 2:
        raise ValueError(
            f"Feature dimension must be at least 2, got {n_features}"
        )
    
    # Apply farthest point sampling if dataset is too large
    if n_samples > max_samples:
        logging.info(
            f"Dataset has {n_samples} samples, downsampling to {max_samples} "
            f"using farthest point sampling for efficiency"
        )
        
        # Convert back to tensor for sampling
        tensor_features = torch.tensor(flattened_features, dtype=torch.float32)
        sample_indices = farthest_point_sampling(tensor_features, max_samples)
        sampled_features = flattened_features[sample_indices]
    else:
        sampled_features = flattened_features
    
    # Validate sampled data quality
    if np.any(np.isnan(sampled_features)) or np.any(np.isinf(sampled_features)):
        logging.warning("Input features contain NaN or infinite values, which may affect estimation")
    
    # Estimate intrinsic dimensionality using MLE
    try:
        mle_estimator = skdim.id.MLE()
        fitted_estimator = mle_estimator.fit(sampled_features)
        estimated_dimension = fitted_estimator.dimension_
        
        # Handle failed global estimation
        if estimated_dimension <= 0 or not np.isfinite(estimated_dimension):
            if hasattr(fitted_estimator, 'dimension_pw_') and fitted_estimator.dimension_pw_ is not None:
                # Fallback to local (pairwise) dimension estimates
                local_dimensions = fitted_estimator.dimension_pw_
                valid_local_dims = local_dimensions[np.isfinite(local_dimensions) & (local_dimensions > 0)]
                
                if len(valid_local_dims) > 0:
                    estimated_dimension = float(np.mean(valid_local_dims))
                    logging.warning(
                        f"Global intrinsic dimension estimation failed (got {fitted_estimator.dimension_}). "
                        f"Using mean of {len(valid_local_dims)} local estimates: {estimated_dimension:.2f}"
                    )
                else:
                    raise RuntimeError("Both global and local dimensionality estimation failed")
            else:
                raise RuntimeError("Global dimensionality estimation failed and no local estimates available")
        
        # Sanity check: intrinsic dimension should not exceed ambient dimension
        if estimated_dimension > n_features:
            logging.warning(
                f"Estimated intrinsic dimension ({estimated_dimension:.2f}) exceeds "
                f"ambient dimension ({n_features}). Capping to ambient dimension."
            )
            estimated_dimension = float(n_features)
        
        # Log results
        compression_ratio = n_features / estimated_dimension if estimated_dimension > 0 else np.inf
        logging.info(
            f"Intrinsic dimensionality estimation completed: "
            f"{estimated_dimension:.2f} (compression ratio: {compression_ratio:.1f}x)"
        )
        
        return float(estimated_dimension)
        
    except Exception as e:
        raise RuntimeError(f"Intrinsic dimensionality estimation failed: {str(e)}") from e

