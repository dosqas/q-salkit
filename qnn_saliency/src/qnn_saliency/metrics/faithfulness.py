# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Faithfulness metrics for evaluating saliency quality."""

from typing import Optional, Union, Tuple
import numpy as np

from ..models.base import BaseModelWrapper


def deletion_score(
    model: BaseModelWrapper,
    x: np.ndarray,
    saliency: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    target: Optional[int] = None,
    n_steps: Optional[int] = None,
    return_curve: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute deletion metric (faithfulness of saliency).
    
    Progressively removes features in order of decreasing saliency
    and measures how quickly model confidence drops. A faithful
    saliency map should cause rapid confidence drop when top
    features are removed.
    
    The deletion score is the Area Under the Deletion Curve (AUDC),
    where lower values indicate more faithful explanations.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    x : np.ndarray
        Input sample of shape (n_features,)
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
    baseline : np.ndarray, optional
        Baseline values for deleted features. Default zeros.
    target : int, optional
        Target class to track probability for. If None, uses predicted class.
        This is important: tracking the TRUE class probability is standard.
    n_steps : int, optional
        Number of deletion steps. Default is n_features.
    return_curve : bool
        If True, also return the deletion curve data.
        
    Returns
    -------
    float or tuple
        If return_curve=False: AUDC score (lower = more faithful)
        If return_curve=True: (AUDC, k_values, confidence_values)
        
    Example
    -------
    >>> audc = deletion_score(model, X[0], saliency_scores, target=y[0])
    >>> print(f"Deletion AUDC: {audc:.3f}")
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    n_features = len(x)
    
    if baseline is None:
        baseline = np.zeros_like(x)
    else:
        baseline = np.asarray(baseline).flatten()
    
    if n_steps is None:
        n_steps = n_features
    
    # Sort features by saliency (descending)
    sorted_indices = np.argsort(-np.abs(saliency))
    
    # Determine target class if not provided
    if target is None:
        target = model.predict_class(x)
        if isinstance(target, np.ndarray):
            target = int(target.flat[0])
    
    # Compute deletion curve
    k_values = []
    confidences = []
    
    x_modified = x.copy()
    
    # k=0: original prediction
    k_values.append(0)
    proba = model.predict(x_modified)
    proba = np.atleast_1d(proba).flatten()
    
    # Get probability for target class
    if len(proba) > 1:
        conf = float(proba[target])
    else:
        # Binary: proba is P(class=1), adjust for target
        conf = float(proba[0]) if target == 1 else float(1 - proba[0])
    confidences.append(conf)
    
    # Progressive deletion: delete one feature at a time
    for k in range(1, min(n_steps, n_features) + 1):
        feat_idx = sorted_indices[k - 1]
        x_modified[feat_idx] = baseline[feat_idx]
        
        k_values.append(k)
        proba = model.predict(x_modified)
        proba = np.atleast_1d(proba).flatten()
        
        if len(proba) > 1:
            conf = float(proba[target])
        else:
            conf = float(proba[0]) if target == 1 else float(1 - proba[0])
        confidences.append(conf)
    
    k_values = np.array(k_values, dtype=float)
    confidences = np.array(confidences, dtype=float)
    
    # Normalize k to [0, 1]
    k_norm = k_values / n_features
    
    # Compute Area Under Deletion Curve using trapezoidal rule
    try:
        audc = np.trapezoid(confidences, k_norm)
    except AttributeError:
        # Fallback for older numpy versions
        audc = np.trapz(confidences, k_norm)
    
    if return_curve:
        return audc, k_values, confidences
    return audc


def insertion_score(
    model: BaseModelWrapper,
    x: np.ndarray,
    saliency: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    target: Optional[int] = None,
    n_steps: Optional[int] = None,
    return_curve: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute insertion metric (complementary to deletion).
    
    Starts from baseline and progressively reveals features in
    order of decreasing saliency. A faithful saliency map should
    cause rapid confidence increase when top features are inserted.
    
    The insertion score is the Area Under the Insertion Curve (AUIC),
    where higher values indicate more faithful explanations.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    x : np.ndarray
        Input sample of shape (n_features,)
    saliency : np.ndarray
        Saliency scores of shape (n_features,)
    baseline : np.ndarray, optional
        Starting baseline. Default zeros.
    target : int, optional
        Target class to track probability for. If None, uses predicted class.
    n_steps : int, optional
        Number of insertion steps. Default is n_features.
    return_curve : bool
        If True, also return the insertion curve data.
        
    Returns
    -------
    float or tuple
        If return_curve=False: AUIC score (higher = more faithful)
        If return_curve=True: (AUIC, k_values, confidence_values)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    n_features = len(x)
    
    if baseline is None:
        baseline = np.zeros_like(x)
    else:
        baseline = np.asarray(baseline).flatten()
    
    if n_steps is None:
        n_steps = n_features
    
    # Sort features by saliency (descending)
    sorted_indices = np.argsort(-np.abs(saliency))
    
    # Determine target class if not provided (based on full input)
    if target is None:
        target = model.predict_class(x)
        if isinstance(target, np.ndarray):
            target = int(target.flat[0])
    
    # Start from baseline
    x_modified = baseline.copy()
    
    k_values = []
    confidences = []
    
    # Helper to get target class probability
    def get_target_prob(proba):
        proba = np.atleast_1d(proba).flatten()
        if len(proba) > 1:
            return float(proba[target])
        else:
            return float(proba[0]) if target == 1 else float(1 - proba[0])
    
    # k=0: baseline prediction
    k_values.append(0)
    proba = model.predict(x_modified)
    confidences.append(get_target_prob(proba))
    
    # Progressive insertion: insert one feature at a time
    for k in range(1, min(n_steps, n_features) + 1):
        feat_idx = sorted_indices[k - 1]
        x_modified[feat_idx] = x[feat_idx]
        
        k_values.append(k)
        proba = model.predict(x_modified)
        confidences.append(get_target_prob(proba))
    
    k_values = np.array(k_values, dtype=float)
    confidences = np.array(confidences, dtype=float)
    
    # Normalize k
    k_norm = k_values / n_features
    
    try:
        auic = np.trapezoid(confidences, k_norm)
    except AttributeError:
        auic = np.trapz(confidences, k_norm)
    
    if return_curve:
        return auic, k_values, confidences
    return auic


def average_sensitivity(
    model: BaseModelWrapper,
    X: np.ndarray,
    saliency_fn: callable,
    delta: float = 1e-3,
    n_samples: int = 10
) -> float:
    """
    Compute average sensitivity of saliency explanations.
    
    Measures how much the saliency map changes under small
    perturbations of the input. Lower values indicate more
    stable/robust explanations.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model (for context)
    X : np.ndarray
        Input samples of shape (n_samples, n_features)
    saliency_fn : callable
        Function that takes x and returns saliency scores
    delta : float
        Perturbation magnitude. Default 1e-3.
    n_samples : int
        Number of perturbations per sample. Default 10.
        
    Returns
    -------
    float
        Average sensitivity (lower = more stable)
        
    Example
    -------
    >>> def get_saliency(x):
    ...     return saliency_method.attribute(x)
    >>> sens = average_sensitivity(model, X_test, get_saliency)
    """
    X = np.atleast_2d(X)
    total_diff = 0.0
    count = 0
    
    for x in X:
        # Original saliency
        sal_orig = saliency_fn(x)
        
        for _ in range(n_samples):
            # Perturbed input
            noise = np.random.randn(*x.shape) * delta
            x_pert = x + noise
            
            # Perturbed saliency
            sal_pert = saliency_fn(x_pert)
            
            # Mean absolute difference
            total_diff += np.mean(np.abs(sal_orig - sal_pert))
            count += 1
    
    return total_diff / count if count > 0 else 0.0
