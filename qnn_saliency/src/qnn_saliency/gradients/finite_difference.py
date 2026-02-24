# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Finite-difference gradient computation for hybrid QNNs."""

from typing import Callable, Optional
import numpy as np

from .base import BaseGradientEngine


class FiniteDifferenceGradient(BaseGradientEngine):
    """
    Compute gradients using central finite difference method.
    
    This is suitable for hybrid QNN models where the quantum layer
    is wrapped with PyTorch or other autodiff frameworks, but we want
    to compute input gradients without relying on autodiff.
    
    The gradient is approximated as:
        ∂f/∂x_j ≈ (f(x + δe_j) - f(x - δe_j)) / (2δ)
    
    For forward-only mode (faster but less accurate):
        ∂f/∂x_j ≈ (f(x + δe_j) - f(x)) / δ
    
    Parameters
    ----------
    delta : float
        Perturbation size for finite difference. Default is 1e-3.
    central : bool
        If True, use central difference (more accurate).
        If False, use forward difference (faster). Default is True.
    aggregate : str
        How to aggregate gradients across output dimensions.
        - 'mean_abs': Mean of absolute values (default for saliency)
        - 'sum': Sum across outputs
        - 'target': Use specific target index only
        
    Example
    -------
    >>> engine = FiniteDifferenceGradient(delta=1e-3)
    >>> grad = engine.compute_gradient(x, model.forward, target_idx=0)
    """
    
    def __init__(
        self,
        delta: float = 1e-3,
        central: bool = True,
        aggregate: str = "mean_abs"
    ):
        """
        Initialize finite difference gradient engine.
        
        Parameters
        ----------
        delta : float
            Perturbation size
        central : bool
            Use central (True) or forward (False) difference
        aggregate : str
            Aggregation method: 'mean_abs', 'sum', or 'target'
        """
        self.delta = delta
        self.central = central
        self.aggregate = aggregate
    
    def compute_gradient(
        self,
        x: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        target_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient of model output w.r.t. input features.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        forward_fn : Callable
            Function that takes input and returns model output.
            Should accept shape (1, n_features) or (n_features,) and
            return shape (n_outputs,) or (1, n_outputs).
        target_idx : int, optional
            Index of target output to compute gradient for.
            Only used when aggregate='target'.
            
        Returns
        -------
        np.ndarray
            Gradient of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        n_features = len(x)
        
        # Get base output for forward difference
        if not self.central:
            base_output = self._call_forward(forward_fn, x)
        
        gradients = np.zeros(n_features, dtype=np.float64)
        
        for j in range(n_features):
            if self.central:
                # Central difference: (f(x+δ) - f(x-δ)) / 2δ
                x_plus = x.copy()
                x_plus[j] += self.delta
                x_minus = x.copy()
                x_minus[j] -= self.delta
                
                out_plus = self._call_forward(forward_fn, x_plus)
                out_minus = self._call_forward(forward_fn, x_minus)
                
                grad = (out_plus - out_minus) / (2 * self.delta)
            else:
                # Forward difference: (f(x+δ) - f(x)) / δ
                x_plus = x.copy()
                x_plus[j] += self.delta
                
                out_plus = self._call_forward(forward_fn, x_plus)
                grad = (out_plus - base_output) / self.delta
            
            # Aggregate across output dimensions
            gradients[j] = self._aggregate(grad, target_idx)
        
        return gradients
    
    def _call_forward(
        self,
        forward_fn: Callable,
        x: np.ndarray
    ) -> np.ndarray:
        """Call forward function and ensure output is 1D numpy array."""
        # Try with batch dimension first
        try:
            out = forward_fn(x.reshape(1, -1))
        except (ValueError, TypeError):
            out = forward_fn(x)
        
        # Convert to numpy if needed (e.g., from torch)
        if hasattr(out, 'detach'):
            out = out.detach().cpu().numpy()
        
        return np.asarray(out).flatten()
    
    def _aggregate(
        self,
        grad: np.ndarray,
        target_idx: Optional[int]
    ) -> float:
        """Aggregate gradient across output dimensions."""
        if self.aggregate == "target" and target_idx is not None:
            return grad[target_idx]
        elif self.aggregate == "sum":
            return np.sum(grad)
        else:  # mean_abs (default for saliency)
            return np.mean(np.abs(grad))
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            "method": "FiniteDifferenceGradient",
            "delta": self.delta,
            "central": self.central,
            "aggregate": self.aggregate
        }
