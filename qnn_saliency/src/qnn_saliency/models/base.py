# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Abstract base class for QNN model wrappers."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class BaseModelWrapper(ABC):
    """
    Abstract base class for wrapping quantum neural network models.
    
    This wrapper provides a unified interface for different QNN architectures
    (pure VQC, hybrid QNN with PyTorch, etc.) to enable consistent saliency
    computation across model types.
    
    Subclasses must implement:
        - predict(x): Forward pass returning class probabilities
        - predict_class(x): Return predicted class label
        - n_features: Property returning number of input features
    """
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute model prediction (probabilities or raw output).
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Model output. For classification, this should be probabilities
            of shape (n_classes,) or (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def predict_class(self, x: np.ndarray) -> Union[int, np.ndarray]:
        """
        Compute predicted class label(s).
        
        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        int or np.ndarray
            Predicted class label(s)
        """
        pass
    
    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of input features expected by the model."""
        pass
    
    @property
    def n_classes(self) -> Optional[int]:
        """
        Number of output classes (if classification model).
        
        Returns None if not applicable (e.g., regression).
        """
        return None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for predict() to allow model(x) syntax."""
        return self.predict(x)
