# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Abstract base class for gradient computation engines."""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np


class BaseGradientEngine(ABC):
    """
    Abstract base class for computing gradients of QNN outputs w.r.t. inputs.
    
    Different QNN architectures require different gradient computation strategies:
    - Pure VQC: Parameter-shift rule with π/2 shifts
    - Hybrid QNN: Finite-difference or automatic differentiation
    
    Subclasses must implement:
        - compute_gradient(x, model, target_class): Gradient computation
    """
    
    @abstractmethod
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
            Function that takes input and returns model output
        target_idx : int, optional
            Index of target output (class) to compute gradient for.
            If None, uses predicted class or first output.
            
        Returns
        -------
        np.ndarray
            Gradient of shape (n_features,)
        """
        pass
    
    def __call__(
        self,
        x: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        target_idx: Optional[int] = None
    ) -> np.ndarray:
        """Alias for compute_gradient()."""
        return self.compute_gradient(x, forward_fn, target_idx)
