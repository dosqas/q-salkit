# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Integrated Gradients saliency method."""

from typing import Optional, Dict, Any, Union
import numpy as np

from .base import BaseSaliency
from ..models.base import BaseModelWrapper
from ..gradients.base import BaseGradientEngine
from ..gradients.finite_difference import FiniteDifferenceGradient


class IntegratedGradients(BaseSaliency):
    """
    Integrated Gradients attribution method.
    
    Computes attribution by integrating gradients along a straight path
    from a baseline to the input:
    
        IG_j(x) = (x_j - x'_j) × ∫₀¹ (∂f(x' + α(x-x'))/∂x_j) dα
        
    Approximated with Riemann sum:
    
        IG_j ≈ (x_j - x'_j) × (1/m) × Σ_k (∂f(x' + k/m × (x-x'))/∂x_j)
    
    Integrated Gradients satisfies the axioms of completeness and
    sensitivity, making it a theoretically grounded attribution method.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    gradient_engine : BaseGradientEngine, optional
        Gradient computation engine
    n_steps : int
        Number of interpolation steps. Default 25.
    baseline : str or np.ndarray
        Baseline choice: 'zero', 'mean', or custom array. Default 'zero'.
        
    References
    ----------
    Sundararajan et al., "Axiomatic Attribution for Deep Networks", 2017
    https://arxiv.org/abs/1703.01365
    
    Example
    -------
    >>> ig = IntegratedGradients(wrapper, n_steps=50, baseline='mean')
    >>> scores = ig.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        gradient_engine: Optional[BaseGradientEngine] = None,
        n_steps: int = 25,
        baseline: Union[str, np.ndarray] = "zero"
    ):
        """
        Initialize Integrated Gradients method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model
        gradient_engine : BaseGradientEngine, optional
            Gradient computation engine
        n_steps : int
            Number of Riemann steps (m)
        baseline : str or np.ndarray
            Baseline: 'zero', 'mean', or custom array
        """
        super().__init__(model)
        
        if gradient_engine is None:
            self.gradient_engine = FiniteDifferenceGradient(aggregate="target")
        else:
            self.gradient_engine = gradient_engine
        
        self.n_steps = n_steps
        self.baseline = baseline
        self._reference_data = None  # For 'mean' baseline
    
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
        """Get baseline vector for given input."""
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
        Compute Integrated Gradients attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class for attribution
        **kwargs
            baseline : np.ndarray, optional
                Override default baseline for this call
            n_steps : int, optional
                Override default number of steps
            
        Returns
        -------
        np.ndarray
            Attribution scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        
        # Get baseline
        if 'baseline' in kwargs:
            x_baseline = np.asarray(kwargs['baseline']).flatten()
        else:
            x_baseline = self._get_baseline(x)
        
        n_steps = kwargs.get('n_steps', self.n_steps)
        
        # Path interpolation: x' + α(x - x') for α in [0, 1]
        alphas = np.linspace(0.0, 1.0, n_steps, endpoint=True)
        diff = x - x_baseline
        
        # Define forward function
        def forward_fn(x_in):
            return self.model.predict(x_in)
        
        # Accumulate gradients along path
        grads_sum = np.zeros_like(x)
        for alpha in alphas:
            x_interp = x_baseline + alpha * diff
            grad = self.gradient_engine.compute_gradient(
                x_interp, forward_fn, target
            )
            grads_sum += grad
        
        # Integrated Gradients formula
        ig = diff * (grads_sum / len(alphas))
        
        return ig
    
    def get_config(self) -> Dict[str, Any]:
        baseline_config = (
            self.baseline if isinstance(self.baseline, str) 
            else "custom_array"
        )
        return {
            "method": self.name,
            "n_steps": self.n_steps,
            "baseline": baseline_config,
            "gradient_engine": self.gradient_engine.get_config()
        }
