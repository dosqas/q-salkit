# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Distribution metrics for saliency map quality."""

import numpy as np


def saliency_entropy(saliency: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute entropy of normalized saliency distribution.
    
    Higher entropy indicates more uniform (less focused) attribution,
    while lower entropy indicates attribution concentrated on few features.
    
    Entropy is computed as:
        H = -Σ p_j log(p_j)
        
    where p_j = |s_j| / Σ|s_k| is the normalized saliency.
    
    Parameters
    ----------
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
    eps : float
        Small constant for numerical stability. Default 1e-10.
        
    Returns
    -------
    float
        Entropy value in nats. Range [0, log(n_features)].
        Lower = more concentrated, Higher = more uniform.
        
    Example
    -------
    >>> entropy = saliency_entropy(saliency_scores)
    >>> print(f"Saliency entropy: {entropy:.3f}")
    """
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    
    # Normalize to probability distribution
    abs_sal = np.abs(saliency)
    total = abs_sal.sum()
    
    if total < eps:
        # Uniform distribution if all zeros
        n = len(saliency)
        return np.log(n) if n > 0 else 0.0
    
    p = abs_sal / total
    
    # Compute entropy (avoid log(0))
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p * np.log(p_safe))
    
    return float(entropy)


def saliency_sparseness(saliency: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute sparseness of saliency map (L1/L2 ratio).
    
    Sparseness is computed as the ratio of L1 to L2 norms:
        sparseness = L1 / L2 = Σ|s_j| / √(Σ s_j²)
    
    **Interpretation:**
    - Lower values → more sparse (concentrated on few features)
    - Higher values → less sparse (spread across features)
    - Range: [1, √n] where n = number of features
    - Minimum (1) when only one feature has non-zero value
    - Maximum (√n) when all features have equal values
    
    Parameters
    ----------
    saliency : np.ndarray
        Saliency scores of shape (n_features,) or (n_samples, n_features)
    eps : float
        Small constant for numerical stability. Default 1e-12.
        
    Returns
    -------
    float
        Sparseness value. Lower = more concentrated.
        
    Example
    -------
    >>> sparseness = saliency_sparseness(saliency_scores)
    >>> print(f"Sparseness (L1/L2): {sparseness:.3f}")
    """
    saliency = np.asarray(saliency, dtype=np.float64)
    
    # Handle batch: average sparseness across samples
    if saliency.ndim == 2:
        return float(np.mean([saliency_sparseness(s) for s in saliency]))
    
    saliency = saliency.flatten()
    
    # L1 and L2 norms
    l1 = np.sum(np.abs(saliency))
    l2 = np.sqrt(np.sum(saliency ** 2))
    
    if l2 < eps:
        return float(np.sqrt(len(saliency)))  # Uniform zero case
    
    return float(l1 / (l2 + eps))


def hoyer_sparseness(saliency: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute normalized Hoyer sparseness measure.
    
    The Hoyer sparseness normalizes L1/L2 ratio to [0, 1]:
        sparseness = (√n - L1/L2) / (√n - 1)
    
    **Interpretation:**
    - 0 = perfectly uniform distribution
    - 1 = perfectly sparse (single feature)
    
    Parameters
    ----------
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
    eps : float
        Small constant for numerical stability.
        
    Returns
    -------
    float
        Normalized sparseness in [0, 1]. Higher = more sparse.
        
    References
    ----------
    Hoyer, P.O. "Non-negative matrix factorization with sparseness 
    constraints." JMLR 5 (2004): 1457-1469.
    """
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    n = len(saliency)
    
    if n <= 1:
        return 1.0
    
    l1 = np.sum(np.abs(saliency))
    l2 = np.sqrt(np.sum(saliency ** 2))
    
    if l2 < eps:
        return 0.0
    
    ratio = l1 / l2
    sqrt_n = np.sqrt(n)
    sparseness = (sqrt_n - ratio) / (sqrt_n - 1)
    
    return float(np.clip(sparseness, 0.0, 1.0))


def saliency_concentration(
    saliency: np.ndarray,
    top_k: int = 1
) -> float:
    """
    Compute what fraction of total saliency is in top-k features.
    
    This is a simple measure of how concentrated the saliency is.
    
    Parameters
    ----------
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
    top_k : int
        Number of top features to consider. Default 1.
        
    Returns
    -------
    float
        Fraction of total saliency in top-k features.
        Range [0, 1] where 1 = all in top-k.
        
    Example
    -------
    >>> conc = saliency_concentration(scores, top_k=2)
    >>> print(f"{conc*100:.1f}% of saliency in top 2 features")
    """
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    abs_sal = np.abs(saliency)
    
    total = abs_sal.sum()
    if total < 1e-10:
        return 0.0
    
    # Top-k sum
    top_k = min(top_k, len(saliency))
    sorted_sal = np.sort(abs_sal)[::-1]
    top_k_sum = sorted_sal[:top_k].sum()
    
    return float(top_k_sum / total)


def gini_coefficient(saliency: np.ndarray) -> float:
    """
    Compute Gini coefficient of saliency distribution.
    
    The Gini coefficient measures inequality in the distribution.
    - 0 = perfect equality (uniform saliency)
    - 1 = perfect inequality (all saliency on one feature)
    
    Parameters
    ----------
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
        
    Returns
    -------
    float
        Gini coefficient in [0, 1].
        
    Example
    -------
    >>> gini = gini_coefficient(saliency_scores)
    >>> print(f"Gini: {gini:.3f}")
    """
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    abs_sal = np.abs(saliency)
    n = len(abs_sal)
    
    if n == 0 or abs_sal.sum() < 1e-10:
        return 0.0
    
    # Sort in ascending order
    sorted_sal = np.sort(abs_sal)
    
    # Gini formula
    cumsum = np.cumsum(sorted_sal)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_sal))) / (n * cumsum[-1]) - (n + 1) / n
    
    return float(np.clip(gini, 0.0, 1.0))
