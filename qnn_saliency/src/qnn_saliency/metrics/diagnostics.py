# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Diagnostic metrics for saliency analysis."""

from typing import Dict, Any, Optional
import numpy as np

from ..models.base import BaseModelWrapper


def bug_detection_signal(
    model: BaseModelWrapper,
    X: np.ndarray,
    y: np.ndarray,
    saliency: np.ndarray
) -> Dict[str, float]:
    """
    Compute bug detection signal for saliency maps.
    
    Compares average saliency magnitude between correctly and
    incorrectly classified samples. If the model relies on
    different features for wrong predictions, this may indicate
    potential bugs or spurious correlations.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    X : np.ndarray
        Input samples of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    saliency : np.ndarray
        Saliency maps of shape (n_samples, n_features)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'correct_mean_saliency': Mean saliency for correct predictions
        - 'incorrect_mean_saliency': Mean saliency for incorrect predictions
        - 'ratio': Ratio of incorrect/correct (>1 may indicate issues)
        - 'n_correct': Number of correct predictions
        - 'n_incorrect': Number of incorrect predictions
        
    Example
    -------
    >>> signal = bug_detection_signal(model, X_test, y_test, saliency_maps)
    >>> print(f"Bug signal ratio: {signal['ratio']:.3f}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()
    saliency = np.atleast_2d(saliency)
    
    # Get predictions
    y_pred = model.predict_class(X)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array([y_pred])
    
    # Separate correct and incorrect
    correct_mask = (y_pred == y)
    incorrect_mask = ~correct_mask
    
    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()
    
    # Mean saliency magnitude
    eps = 1e-10
    
    if n_correct > 0:
        correct_saliency = np.mean(np.abs(saliency[correct_mask]))
    else:
        correct_saliency = 0.0
    
    if n_incorrect > 0:
        incorrect_saliency = np.mean(np.abs(saliency[incorrect_mask]))
    else:
        incorrect_saliency = 0.0
    
    # Ratio (avoid division by zero)
    if correct_saliency > eps:
        ratio = incorrect_saliency / correct_saliency
    else:
        ratio = float('inf') if incorrect_saliency > eps else 1.0
    
    return {
        'correct_mean_saliency': float(correct_saliency),
        'incorrect_mean_saliency': float(incorrect_saliency),
        'ratio': float(ratio),
        'n_correct': int(n_correct),
        'n_incorrect': int(n_incorrect)
    }


def minimum_efficacy(
    model: BaseModelWrapper,
    X: np.ndarray,
    saliency: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    top_k: int = 1
) -> float:
    """
    Compute minimum efficacy of saliency explanations.
    
    Measures whether removing the single most salient feature
    causes a meaningful drop in model confidence. This tests
    if the saliency correctly identifies at least one truly
    important feature.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    X : np.ndarray
        Input samples of shape (n_samples, n_features)
    saliency : np.ndarray
        Saliency maps of shape (n_samples, n_features)
    baseline : np.ndarray, optional
        Baseline for deleted features. Default zeros.
    top_k : int
        Number of top features to remove. Default 1.
        
    Returns
    -------
    float
        Average confidence drop when removing top-k features.
        Higher values indicate more effective saliency maps.
        
    Example
    -------
    >>> efficacy = minimum_efficacy(model, X_test, saliency_maps, top_k=1)
    >>> print(f"Min efficacy: {efficacy:.4f}")
    """
    X = np.atleast_2d(X)
    saliency = np.atleast_2d(saliency)
    n_samples, n_features = X.shape
    
    if baseline is None:
        baseline = np.zeros(n_features)
    else:
        baseline = np.asarray(baseline).flatten()
    
    confidence_drops = []
    
    for i in range(n_samples):
        x = X[i].copy()
        sal = saliency[i]
        
        # Base confidence
        base_proba = model.predict(x)
        if base_proba.ndim > 0:
            base_proba = base_proba.flatten()
        base_conf = float(np.max(base_proba))
        
        # Remove top-k features
        top_indices = np.argsort(-np.abs(sal))[:top_k]
        x_modified = x.copy()
        for idx in top_indices:
            x_modified[idx] = baseline[idx]
        
        # New confidence
        new_proba = model.predict(x_modified)
        if new_proba.ndim > 0:
            new_proba = new_proba.flatten()
        new_conf = float(np.max(new_proba))
        
        confidence_drops.append(base_conf - new_conf)
    
    return float(np.mean(confidence_drops))


def overreliance_risk(
    model: BaseModelWrapper,
    X: np.ndarray,
    y: np.ndarray,
    saliency: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    top_k: int = 1
) -> float:
    """
    Compute overreliance risk score.
    
    Measures correlation between model confidence and negative
    deletion fidelity. High correlation indicates that users might
    overtrust confident predictions even when explanations are weak.
    
    Risk = corr(confidence, -fidelity)
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    X : np.ndarray
        Input samples of shape (n_samples, n_features)
    y : np.ndarray
        True labels of shape (n_samples,)
    saliency : np.ndarray
        Saliency maps of shape (n_samples, n_features)
    baseline : np.ndarray, optional
        Baseline for deleted features. Default zeros.
    top_k : int
        Number of top features to remove. Default 1.
        
    Returns
    -------
    float
        Correlation coefficient in [-1, 1].
        Positive = high risk (confidence misleading)
        Negative/zero = low risk (explanations reliable)
        
    Example
    -------
    >>> risk = overreliance_risk(model, X_test, y_test, saliency_maps)
    >>> print(f"Overreliance risk: {risk:.3f}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()
    saliency = np.atleast_2d(saliency)
    n_samples, n_features = X.shape
    
    if baseline is None:
        baseline = np.zeros(n_features)
    else:
        baseline = np.asarray(baseline).flatten()
    
    confidences = []
    fidelities = []
    
    for i in range(n_samples):
        x = X[i].copy()
        sal = saliency[i]
        label = y[i]
        
        # Base prediction
        base_proba = model.predict(x)
        if base_proba.ndim > 0:
            base_proba = base_proba.flatten()
        
        # Confidence (max probability)
        confidence = float(np.max(base_proba))
        confidences.append(confidence)
        
        # Confidence for true class
        if len(base_proba) > 1:
            base_true_conf = float(base_proba[label])
        else:
            base_true_conf = float(base_proba[0]) if label == 1 else 1 - float(base_proba[0])
        
        # Remove top-k features
        top_indices = np.argsort(-np.abs(sal))[:top_k]
        x_modified = x.copy()
        for idx in top_indices:
            x_modified[idx] = baseline[idx]
        
        # New prediction
        new_proba = model.predict(x_modified)
        if new_proba.ndim > 0:
            new_proba = new_proba.flatten()
        
        if len(new_proba) > 1:
            new_true_conf = float(new_proba[label])
        else:
            new_true_conf = float(new_proba[0]) if label == 1 else 1 - float(new_proba[0])
        
        # Fidelity = confidence drop for true class
        fidelity = base_true_conf - new_true_conf
        fidelities.append(fidelity)
    
    confidences = np.array(confidences)
    fidelities = np.array(fidelities)
    
    # Correlation between confidence and negative fidelity
    if len(confidences) < 2:
        return 0.0
    
    corr_matrix = np.corrcoef(confidences, -fidelities)
    return float(corr_matrix[0, 1])


def feature_agreement(
    saliency1: np.ndarray,
    saliency2: np.ndarray,
    top_k: int = 3
) -> float:
    """
    Compute agreement between two saliency methods.
    
    Measures what fraction of top-k features are shared between
    two different saliency methods. Higher agreement suggests
    more robust explanations.
    
    Parameters
    ----------
    saliency1 : np.ndarray
        First saliency scores of shape (n_features,) or (n_samples, n_features)
    saliency2 : np.ndarray
        Second saliency scores of shape (n_features,) or (n_samples, n_features)
    top_k : int
        Number of top features to compare. Default 3.
        
    Returns
    -------
    float
        Agreement ratio in [0, 1]. 1 = perfect agreement.
        
    Example
    -------
    >>> agreement = feature_agreement(ig_scores, occlusion_scores, top_k=2)
    >>> print(f"Top-2 agreement: {agreement:.2%}")
    """
    saliency1 = np.atleast_2d(saliency1)
    saliency2 = np.atleast_2d(saliency2)
    
    agreements = []
    
    for s1, s2 in zip(saliency1, saliency2):
        top1 = set(np.argsort(-np.abs(s1))[:top_k])
        top2 = set(np.argsort(-np.abs(s2))[:top_k])
        
        intersection = len(top1 & top2)
        agreement = intersection / top_k
        agreements.append(agreement)
    
    return float(np.mean(agreements))
