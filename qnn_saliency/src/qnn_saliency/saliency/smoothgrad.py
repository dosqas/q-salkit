# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""SmoothGrad saliency method."""

from typing import Optional, Dict, Any
import numpy as np

from .base import BaseSaliency
from .gradient import GradientSaliency
from ..models.base import BaseModelWrapper
from ..gradients.base import BaseGradientEngine


class SmoothGrad(BaseSaliency):
    """
    SmoothGrad saliency method.
    
    Reduces noise in gradient-based saliency by averaging gradients
    over multiple noisy copies of the input:
    
        SmoothGrad(x) = (1/K) × Σ_k |∇f(x + ε_k)|
        
    where ε_k ~ N(0, σ²I)
    
    This produces smoother, more interpretable saliency maps by
    filtering out gradient noise artifacts.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    gradient_engine : BaseGradientEngine, optional
        Gradient computation engine
    n_samples : int
        Number of noisy samples to average. Default 30.
    sigma : float
        Standard deviation of Gaussian noise. Default 0.1.
        
    References
    ----------
    Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017
    https://arxiv.org/abs/1706.03825
    
    Example
    -------
    >>> saliency = SmoothGrad(wrapper, n_samples=50, sigma=0.15)
    >>> scores = saliency.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        gradient_engine: Optional[BaseGradientEngine] = None,
        n_samples: int = 30,
        sigma: float = 0.1
    ):
        """
        Initialize SmoothGrad method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model
        gradient_engine : BaseGradientEngine, optional
            Gradient computation engine
        n_samples : int
            Number of noisy samples (K)
        sigma : float
            Noise standard deviation (σ)
        """
        super().__init__(model)
        
        self.gradient_saliency = GradientSaliency(
            model, 
            gradient_engine=gradient_engine,
            absolute=True
        )
        self.n_samples = n_samples
        self.sigma = sigma
    
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute SmoothGrad attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class (passed to gradient computation)
        **kwargs
            n_samples : int, optional
                Override default number of samples
            sigma : float, optional
                Override default noise level
            
        Returns
        -------
        np.ndarray
            Smoothed saliency scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        
        n_samples = kwargs.get('n_samples', self.n_samples)
        sigma = kwargs.get('sigma', self.sigma)
        
        gradients = []
        for _ in range(n_samples):
            # Add Gaussian noise
            noise = np.random.normal(0.0, sigma, size=x.shape)
            x_noisy = x + noise
            
            # Compute gradient saliency
            grad = self.gradient_saliency.attribute(x_noisy, target=target)
            gradients.append(grad)
        
        # Average across noisy samples
        return np.mean(np.stack(gradients), axis=0)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_samples": self.n_samples,
            "sigma": self.sigma,
            "gradient_engine": self.gradient_saliency.gradient_engine.get_config()
        }
