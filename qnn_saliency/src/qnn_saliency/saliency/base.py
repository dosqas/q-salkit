# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Abstract base class for saliency methods."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..models.base import BaseModelWrapper


class BaseSaliency(ABC):
    """
    Abstract base class for saliency/attribution methods.
    
    Saliency methods compute importance scores for each input feature,
    indicating how much each feature contributes to the model's prediction.
    
    Subclasses must implement:
        - attribute(x, target): Compute attribution scores
    
    Parameters
    ----------
    model : BaseModelWrapper
        Wrapped QNN model to explain
    """
    
    def __init__(self, model: BaseModelWrapper):
        """
        Initialize saliency method.
        
        Parameters
        ----------
        model : BaseModelWrapper
            Wrapped QNN model to compute attributions for
        """
        self.model = model
    
    @abstractmethod
    def attribute(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute attribution scores for input features.
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,)
        target : int, optional
            Target class to compute attribution for.
            If None, uses the predicted class.
        **kwargs
            Method-specific additional arguments
            
        Returns
        -------
        np.ndarray
            Attribution scores of shape (n_features,)
            Higher absolute values indicate more important features.
        """
        pass
    
    def attribute_batch(
        self,
        X: np.ndarray,
        target: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute attribution scores for multiple samples.
        
        Default implementation loops over samples. Subclasses may
        override for more efficient batch processing.
        
        Parameters
        ----------
        X : np.ndarray
            Input samples of shape (n_samples, n_features)
        target : np.ndarray, optional
            Target classes of shape (n_samples,)
        **kwargs
            Method-specific additional arguments
            
        Returns
        -------
        np.ndarray
            Attribution scores of shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        results = []
        
        for i in range(n_samples):
            t = target[i] if target is not None else None
            results.append(self.attribute(X[i], target=t, **kwargs))
        
        return np.array(results)
    
    @property
    def name(self) -> str:
        """Human-readable name of the saliency method."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters of the saliency method.
        
        Useful for reproducibility and logging.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return {"method": self.name}
    
    def __call__(
        self,
        x: np.ndarray,
        target: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Alias for attribute()."""
        return self.attribute(x, target=target, **kwargs)
