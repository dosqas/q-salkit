# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""VQC model wrapper for pure Variational Quantum Classifiers."""

from typing import Optional, Callable, Union
import numpy as np

from .base import BaseModelWrapper


class VQCWrapper(BaseModelWrapper):
    """
    Wrapper for pure Variational Quantum Classifier models.
    
    This wrapper provides a unified interface for VQC models that use
    Qiskit primitives (Estimator/Sampler) for quantum circuit execution.
    
    Parameters
    ----------
    predict_fn : Callable
        Function that takes (theta, X) and returns raw expectation values.
        Typically wraps an Estimator.run() call.
    theta : np.ndarray
        Trained variational parameters of the VQC.
    n_features : int
        Number of input features.
    score_to_proba : Callable, optional
        Function to convert raw scores to probabilities.
        Default is sigmoid: p = 1/(1 + exp(-z))
    threshold : float
        Classification threshold for binary classification. Default is 0.5.
        
    Example
    -------
    >>> # After training a VQC
    >>> def vqc_predict(theta, X):
    ...     # Bind parameters and run estimator
    ...     return estimator.run(circuits, observables, params).result().values
    >>> 
    >>> wrapper = VQCWrapper(
    ...     predict_fn=vqc_predict,
    ...     theta=theta_star,
    ...     n_features=4
    ... )
    >>> proba = wrapper.predict(X_test)
    >>> classes = wrapper.predict_class(X_test)
    """
    
    def __init__(
        self,
        predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        theta: np.ndarray,
        n_features: int,
        score_to_proba: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        threshold: float = 0.5,
        n_classes: int = 2
    ):
        """
        Initialize VQC wrapper.
        
        Parameters
        ----------
        predict_fn : Callable
            Function (theta, X) -> raw_scores
        theta : np.ndarray
            Trained parameters
        n_features : int
            Number of input features
        score_to_proba : Callable, optional
            Score to probability conversion. Default: sigmoid
        threshold : float
            Classification threshold
        n_classes : int
            Number of output classes (default: 2 for binary)
        """
        self._predict_fn = predict_fn
        self._theta = np.asarray(theta, dtype=np.float64)
        self._n_features = n_features
        self._threshold = threshold
        self._n_classes = n_classes
        
        if score_to_proba is None:
            self._score_to_proba = self._default_sigmoid
        else:
            self._score_to_proba = score_to_proba
    
    @staticmethod
    def _default_sigmoid(z: np.ndarray) -> np.ndarray:
        """Default sigmoid: p = 1/(1 + exp(-z))"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """
        Get raw expectation values from VQC.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Raw expectation values
        """
        x = np.atleast_2d(x)
        return self._predict_fn(self._theta, x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples,) for binary classification
            or (n_samples, n_classes) for multi-class
        """
        raw = self.predict_raw(x)
        return self._score_to_proba(raw)
    
    def predict_class(self, x: np.ndarray) -> Union[int, np.ndarray]:
        """
        Predict class labels.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        int or np.ndarray
            Predicted class label(s)
        """
        proba = self.predict(x)
        if self._n_classes == 2:
            labels = (proba >= self._threshold).astype(int)
        else:
            labels = np.argmax(proba, axis=-1)
        
        # Return scalar if single input
        if labels.ndim == 0 or (labels.ndim == 1 and len(labels) == 1):
            return int(labels.flat[0])
        return labels
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self._n_features
    
    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self._n_classes
    
    @property
    def theta(self) -> np.ndarray:
        """Trained variational parameters."""
        return self._theta.copy()
    
    @theta.setter
    def theta(self, value: np.ndarray):
        """Update variational parameters."""
        self._theta = np.asarray(value, dtype=np.float64)
    
    def expectation(self, x: np.ndarray) -> float:
        """
        Compute expectation value for a single input.
        
        Convenience method for saliency computation.
        
        Parameters
        ----------
        x : np.ndarray
            Single input of shape (n_features,)
            
        Returns
        -------
        float
            Raw expectation value
        """
        x = np.atleast_2d(x)
        raw = self._predict_fn(self._theta, x)
        return float(np.asarray(raw).flat[0])


class HybridQNNWrapper(BaseModelWrapper):
    """
    Wrapper for hybrid quantum-classical neural network models.
    
    This wrapper supports models that combine quantum layers (via Qiskit)
    with classical neural network layers (e.g., PyTorch).
    
    Parameters
    ----------
    model : object
        The hybrid model object. Should have a forward/predict method.
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes.
    device : str
        Device for PyTorch tensors ('cpu' or 'cuda').
        
    Example
    -------
    >>> # With a PyTorch hybrid model
    >>> wrapper = HybridQNNWrapper(model, n_features=4, n_classes=3)
    >>> proba = wrapper.predict(X_test)
    """
    
    def __init__(
        self,
        model,
        n_features: int,
        n_classes: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize hybrid QNN wrapper.
        
        Parameters
        ----------
        model : object
            Hybrid model with forward() method
        n_features : int
            Number of input features
        n_classes : int
            Number of output classes
        device : str
            PyTorch device
        """
        self._model = model
        self._n_features = n_features
        self._n_classes = n_classes
        self._device = device
        
        # Try to import torch
        try:
            import torch
            self._torch = torch
            self._has_torch = True
        except ImportError:
            self._has_torch = False
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities via softmax.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples, n_classes)
        """
        x = np.atleast_2d(x)
        
        if self._has_torch:
            inputs = self._torch.tensor(x, dtype=self._torch.float32, device=self._device)
            with self._torch.no_grad():
                logits = self._model(inputs)
                proba = self._torch.softmax(logits, dim=1)
            return proba.cpu().numpy()
        else:
            # Fallback: assume model returns numpy
            logits = self._model(x)
            # Simple softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """
        Get raw logits (before softmax).
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Logits of shape (n_samples, n_classes)
        """
        x = np.atleast_2d(x)
        
        if self._has_torch:
            inputs = self._torch.tensor(x, dtype=self._torch.float32, device=self._device)
            with self._torch.no_grad():
                logits = self._model(inputs)
            return logits.cpu().numpy()
        else:
            return self._model(x)
    
    def predict_class(self, x: np.ndarray) -> Union[int, np.ndarray]:
        """
        Predict class labels.
        
        Parameters
        ----------
        x : np.ndarray
            Input of shape (n_features,) or (n_samples, n_features)
            
        Returns
        -------
        int or np.ndarray
            Predicted class label(s)
        """
        proba = self.predict(x)
        labels = np.argmax(proba, axis=1)
        
        if len(labels) == 1:
            return int(labels[0])
        return labels
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self._n_features
    
    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return self._n_classes
    
    @property
    def model(self):
        """Underlying model object."""
        return self._model
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass returning logits.
        
        Convenience method for gradient computation.
        """
        return self.predict_logits(x)
