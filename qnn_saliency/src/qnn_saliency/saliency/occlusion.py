# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Occlusion-based saliency method."""

from typing import Optional, Dict, Any, Union
import numpy as np

from .base import BaseSaliency
from ..models.base import BaseModelWrapper


class Occlusion(BaseSaliency):
    """
    Occlusion (perturbation-based) saliency method.
    
    Measures feature importance by replacing each feature with a
    baseline value and observing the change in model output:
    
        importance_j = |f(x) - f(x with x_j = baseline_j)|
    
    Features whose occlusion causes large output changes are important.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    baseline : str or np.ndarray
        Baseline for occluded features: 'zero', 'mean', or custom array.
        Default 'zero'.
    metric : str
        How to measure importance:
        - 'prob_change': Absolute change in predicted probability
        - 'loss_change': Change in BCE loss (requires true labels)
        Default 'prob_change'.
        
    Example
    -------
    >>> occ = Occlusion(wrapper, baseline='mean', metric='prob_change')
    >>> scores = occ.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        baseline: Union[str, np.ndarray] = "zero",
        metric: str = "prob_change"
    ):
        """
        Initialize Occlusion method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model
        baseline : str or np.ndarray
            Occlusion baseline
        metric : str
            Importance metric type
        """
        super().__init__(model)
        self.baseline = baseline
        self.metric = metric
        self._reference_data = None
    
    def set_reference_data(self, X: np.ndarray):
        """
        Set reference data for computing mean baseline.
        
        Parameters
        ----------
        X : np.ndarray
            Reference dataset of shape (n_samples, n_features)
        """
        self._reference_data = np.asarray(X, dtype=np.float64)
    
    def _get_baseline(self, x: np.ndarray) -> np.ndarray:
        """Get baseline vector."""
        if isinstance(self.baseline, np.ndarray):
            return self.baseline.flatten()
        elif self.baseline == "zero":
            return np.zeros_like(x)
        elif self.baseline == "mean":
            if self._reference_data is not None:
                return self._reference_data.mean(axis=0)
            else:
                raise ValueError(
                    "Baseline 'mean' requires reference data. "
                    "Call set_reference_data() first."
                )
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")
    
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute occlusion-based attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class for loss computation
        **kwargs
            baseline : np.ndarray, optional
                Override default baseline
            y_true : int, optional
                True label (required for 'loss_change' metric)
            
        Returns
        -------
        np.ndarray
            Importance scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        n_features = len(x)
        
        # Get baseline
        if 'baseline' in kwargs:
            baseline_vec = np.asarray(kwargs['baseline']).flatten()
        else:
            baseline_vec = self._get_baseline(x)
        
        # Base prediction
        base_proba = self.model.predict(x)
        if base_proba.ndim > 0:
            base_proba = base_proba.flatten()
        
        importance = np.zeros(n_features, dtype=np.float64)
        
        for j in range(n_features):
            # Occlude feature j
            x_occluded = x.copy()
            x_occluded[j] = baseline_vec[j]
            
            # Get occluded prediction
            occ_proba = self.model.predict(x_occluded)
            if occ_proba.ndim > 0:
                occ_proba = occ_proba.flatten()
            
            if self.metric == "prob_change":
                # Mean absolute change in probabilities
                importance[j] = np.mean(np.abs(occ_proba - base_proba))
            
            elif self.metric == "loss_change":
                # Change in BCE loss
                y_true = kwargs.get('y_true', target)
                if y_true is None:
                    raise ValueError(
                        "y_true or target required for 'loss_change' metric"
                    )
                
                base_loss = self._bce_loss(base_proba, y_true)
                occ_loss = self._bce_loss(occ_proba, y_true)
                
                # Positive if occluding increases loss (feature important)
                importance[j] = max(0.0, occ_loss - base_loss)
            
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
        
        return importance
    
    @staticmethod
    def _bce_loss(proba: np.ndarray, y: int, eps: float = 1e-15) -> float:
        """Binary cross-entropy loss."""
        if len(proba) == 1:
            p = np.clip(proba[0], eps, 1 - eps)
            return -y * np.log(p) - (1 - y) * np.log(1 - p)
        else:
            # Multi-class: negative log probability of true class
            p = np.clip(proba[y], eps, 1.0)
            return -np.log(p)
    
    def get_config(self) -> Dict[str, Any]:
        baseline_config = (
            self.baseline if isinstance(self.baseline, str) 
            else "custom_array"
        )
        return {
            "method": self.name,
            "baseline": baseline_config,
            "metric": self.metric
        }


class NoiseSensitivity(BaseSaliency):
    """
    Noise Sensitivity saliency method.
    
    Measures how sensitive the model output is to noise perturbations
    on each individual feature:
    
        sensitivity_j = E[|f(x) - f(x + ε_j)|]
        
    where ε_j ~ N(0, σ²) is added only to feature j.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    sigma : float
        Standard deviation of Gaussian noise. Default 0.1.
    n_samples : int
        Number of noise samples for Monte Carlo estimation. Default 50.
        
    Example
    -------
    >>> ns = NoiseSensitivity(wrapper, sigma=0.1, n_samples=100)
    >>> scores = ns.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        sigma: float = 0.1,
        n_samples: int = 50
    ):
        """
        Initialize Noise Sensitivity method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model
        sigma : float
            Noise standard deviation
        n_samples : int
            Number of Monte Carlo samples
        """
        super().__init__(model)
        self.sigma = sigma
        self.n_samples = n_samples
    
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute noise sensitivity attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class (not used)
        **kwargs
            sigma : float, optional
                Override default noise level
            n_samples : int, optional
                Override default sample count
            
        Returns
        -------
        np.ndarray
            Sensitivity scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        n_features = len(x)
        
        sigma = kwargs.get('sigma', self.sigma)
        n_samples = kwargs.get('n_samples', self.n_samples)
        
        # Base prediction
        base_out = self.model.predict(x)
        if hasattr(base_out, 'flatten'):
            base_out = base_out.flatten()
        base_val = float(base_out[0]) if len(base_out) > 0 else float(base_out)
        
        sensitivity = np.zeros(n_features, dtype=np.float64)
        
        for j in range(n_features):
            diffs = []
            for _ in range(n_samples):
                # Perturb only feature j
                x_noisy = x.copy()
                x_noisy[j] += np.random.normal(0.0, sigma)
                
                # Get perturbed output
                noisy_out = self.model.predict(x_noisy)
                if hasattr(noisy_out, 'flatten'):
                    noisy_out = noisy_out.flatten()
                noisy_val = float(noisy_out[0]) if len(noisy_out) > 0 else float(noisy_out)
                
                diffs.append(abs(noisy_val - base_val))
            
            sensitivity[j] = np.mean(diffs)
        
        return sensitivity
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "sigma": self.sigma,
            "n_samples": self.n_samples
        }
