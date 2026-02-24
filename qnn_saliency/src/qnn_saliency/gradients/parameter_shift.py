# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Parameter-shift rule gradient computation for VQCs."""

from typing import Callable, Optional
import numpy as np

from .base import BaseGradientEngine


class ParameterShiftGradient(BaseGradientEngine):
    """
    Compute gradients using the parameter-shift rule.
    
    The parameter-shift rule is exact for gates of the form e^{-iθG/2}
    where G has eigenvalues ±1 (e.g., RX, RY, RZ gates):
    
        ∂f/∂θ = (1/2) * [f(θ + π/2) - f(θ - π/2)]
    
    This is the quantum-native gradient computation method, suitable
    for pure VQC models where inputs are encoded as rotation angles.
    
    Parameters
    ----------
    shift : float
        Shift amount for parameter-shift rule. Default is π/2.
        
    Notes
    -----
    When computing input gradients for data-encoding VQCs, the "parameter"
    being differentiated is actually the input feature encoded as a
    rotation angle in the circuit.
    
    Example
    -------
    >>> engine = ParameterShiftGradient()
    >>> grad = engine.compute_gradient(x, vqc_forward)
    """
    
    def __init__(self, shift: float = np.pi / 2):
        """
        Initialize parameter-shift gradient engine.
        
        Parameters
        ----------
        shift : float
            Shift amount (default: π/2 for standard parameter-shift)
        """
        self.shift = shift
    
    def compute_gradient(
        self,
        x: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        target_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient of VQC output w.r.t. input features.
        
        Uses parameter-shift rule:
            ∂f/∂x_j = (1/2) * [f(x + shift*e_j) - f(x - shift*e_j)]
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,). These should be
            angle-encoded inputs for the VQC.
        forward_fn : Callable
            VQC forward function that takes input and returns expectation.
            Should return a scalar or 1D array.
        target_idx : int, optional
            Index of target output if forward_fn returns multiple values.
            If None and output is multi-dimensional, uses the first.
            
        Returns
        -------
        np.ndarray
            Gradient of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        n_features = len(x)
        
        gradients = np.zeros(n_features, dtype=np.float64)
        
        for j in range(n_features):
            # Shifted inputs
            x_plus = x.copy()
            x_plus[j] += self.shift
            x_minus = x.copy()
            x_minus[j] -= self.shift
            
            # Evaluate VQC at shifted points
            out_plus = self._call_forward(forward_fn, x_plus, target_idx)
            out_minus = self._call_forward(forward_fn, x_minus, target_idx)
            
            # Parameter-shift formula
            gradients[j] = 0.5 * (out_plus - out_minus)
        
        return gradients
    
    def _call_forward(
        self,
        forward_fn: Callable,
        x: np.ndarray,
        target_idx: Optional[int]
    ) -> float:
        """Call forward function and extract scalar output."""
        out = forward_fn(x)
        
        # Convert to numpy if needed
        if hasattr(out, 'detach'):
            out = out.detach().cpu().numpy()
        
        out = np.asarray(out).flatten()
        
        if len(out) == 1:
            return float(out[0])
        elif target_idx is not None:
            return float(out[target_idx])
        else:
            return float(out[0])
    
    def get_config(self) -> dict:
        """Get configuration dictionary."""
        return {
            "method": "ParameterShiftGradient",
            "shift": self.shift
        }
