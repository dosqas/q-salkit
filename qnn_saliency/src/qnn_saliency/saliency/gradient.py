# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Gradient-based saliency method."""

from typing import Optional, Dict, Any
import numpy as np

from .base import BaseSaliency
from ..models.base import BaseModelWrapper
from ..gradients.base import BaseGradientEngine
from ..gradients.finite_difference import FiniteDifferenceGradient


class GradientSaliency(BaseSaliency):
    """
    Gradient-based saliency computation.
    
    Computes feature importance as the absolute gradient of the model
    output with respect to input features:
    
        saliency_j = |∂f/∂x_j|
    
    This is the simplest and most direct saliency method, measuring
    how sensitive the model output is to small changes in each feature.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    gradient_engine : BaseGradientEngine, optional
        Gradient computation engine. Default uses FiniteDifferenceGradient.
    absolute : bool
        If True, return absolute values of gradients. Default True.
        
    Example
    -------
    >>> from qnn_saliency import GradientSaliency, VQCWrapper
    >>> from qnn_saliency.gradients import ParameterShiftGradient
    >>>
    >>> wrapper = VQCWrapper(predict_fn, theta, n_features=4)
    >>> saliency = GradientSaliency(wrapper, gradient_engine=ParameterShiftGradient())
    >>> scores = saliency.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        gradient_engine: Optional[BaseGradientEngine] = None,
        absolute: bool = True
    ):
        """
        Initialize gradient saliency method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model
        gradient_engine : BaseGradientEngine, optional
            Gradient computation engine
        absolute : bool
            Return absolute gradient values
        """
        super().__init__(model)
        
        if gradient_engine is None:
            self.gradient_engine = FiniteDifferenceGradient()
        else:
            self.gradient_engine = gradient_engine
        
        self.absolute = absolute
    
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute gradient-based attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class (not used in basic gradient saliency)
        **kwargs
            Additional arguments (unused)
            
        Returns
        -------
        np.ndarray
            Saliency scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        
        # Define forward function for gradient engine
        def forward_fn(x_in):
            return self.model.predict(x_in)
        
        gradient = self.gradient_engine.compute_gradient(x, forward_fn, target)
        
        if self.absolute:
            return np.abs(gradient)
        return gradient
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "method": self.name,
            "absolute": self.absolute,
            "gradient_engine": self.gradient_engine.get_config()
        }


class GradientTimesInput(BaseSaliency):
    """
    Gradient × Input saliency method.
    
    Computes feature importance as the product of the gradient and
    the input value:
    
        saliency_j = x_j × ∂f/∂x_j
    
    This captures both the sensitivity (gradient) and the actual
    contribution (input magnitude) of each feature.
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model
    gradient_engine : BaseGradientEngine, optional
        Gradient computation engine
    absolute : bool
        If True, return absolute values. Default True.
        
    Example
    -------
    >>> saliency = GradientTimesInput(wrapper)
    >>> scores = saliency.attribute(X_test[0])
    """
    
    def __init__(
        self,
        model: BaseModelWrapper,
        gradient_engine: Optional[BaseGradientEngine] = None,
        absolute: bool = True
    ):
        super().__init__(model)
        
        if gradient_engine is None:
            self.gradient_engine = FiniteDifferenceGradient()
        else:
            self.gradient_engine = gradient_engine
        
        self.absolute = absolute
    
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Gradient × Input attribution scores.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class (not used)
        **kwargs
            Additional arguments (unused)
            
        Returns
        -------
        np.ndarray
            Saliency scores of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        
        def forward_fn(x_in):
            return self.model.predict(x_in)
        
        gradient = self.gradient_engine.compute_gradient(x, forward_fn, target)
        
        saliency = x * gradient
        
        if self.absolute:
            return np.abs(saliency)
        return saliency
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "absolute": self.absolute,
            "gradient_engine": self.gradient_engine.get_config()
        }
