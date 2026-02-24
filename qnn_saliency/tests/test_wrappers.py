# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for model wrappers."""

import pytest
import numpy as np

from qnn_saliency.models.wrappers import VQCWrapper


class TestVQCWrapper:
    """Tests for VQCWrapper class."""
    
    @pytest.fixture
    def mock_predict_fn(self):
        """Create a mock prediction function that takes (theta, X)."""
        def predict_fn(theta, x):
            x = np.atleast_2d(x)
            # Simple linear combination using theta as weights
            # For testing, theta is used to scale the output
            s = np.sum(x * theta[:x.shape[1]], axis=1)
            return s  # Return raw scores
        return predict_fn
    
    @pytest.fixture
    def theta(self):
        """Mock trained parameters."""
        return np.array([1.0, -0.5, 0.8, 0.2])
    
    def test_creation(self, mock_predict_fn, theta):
        """Should create wrapper successfully."""
        wrapper = VQCWrapper(
            predict_fn=mock_predict_fn,
            theta=theta,
            n_features=4,
            n_classes=2
        )
        
        assert wrapper.n_features == 4
        assert wrapper.n_classes == 2
    
    def test_predict_raw(self, mock_predict_fn, theta):
        """Should compute raw predictions."""
        wrapper = VQCWrapper(mock_predict_fn, theta, n_features=4)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        raw = wrapper.predict_raw(x)
        
        assert isinstance(raw, np.ndarray)
    
    def test_predict_single_sample(self, mock_predict_fn, theta):
        """Should predict single sample."""
        wrapper = VQCWrapper(mock_predict_fn, theta, n_features=4, n_classes=2)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        proba = wrapper.predict(x)
        
        # Should return probabilities (sigmoid applied)
        assert isinstance(proba, np.ndarray)
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_predict_batch(self, mock_predict_fn, theta):
        """Should predict batch of samples."""
        wrapper = VQCWrapper(mock_predict_fn, theta, n_features=4, n_classes=2)
        X = np.random.randn(5, 4)
        
        proba = wrapper.predict(X)
        
        assert isinstance(proba, np.ndarray)
        assert len(proba) == 5
    
    def test_predict_class_single(self, mock_predict_fn, theta):
        """Should return class for single sample."""
        wrapper = VQCWrapper(mock_predict_fn, theta, n_features=4, n_classes=2)
        x = np.array([1.0, 1.0, 1.0, 1.0])  # Positive sum -> likely class 1
        
        cls = wrapper.predict_class(x)
        
        assert cls in [0, 1]
        assert isinstance(cls, (int, np.integer))
    
    def test_predict_class_batch(self, mock_predict_fn, theta):
        """Should return classes for batch."""
        wrapper = VQCWrapper(mock_predict_fn, theta, n_features=4, n_classes=2)
        X = np.random.randn(5, 4)
        
        classes = wrapper.predict_class(X)
        
        assert len(classes) == 5
        assert all(c in [0, 1] for c in classes)
    
    def test_custom_score_to_proba(self, mock_predict_fn, theta):
        """Should use custom score_to_proba function."""
        # Custom: just clip to [0, 1]
        def custom_proba(z):
            return np.clip(z, 0, 1)
        
        wrapper = VQCWrapper(
            mock_predict_fn, theta, n_features=4,
            score_to_proba=custom_proba
        )
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        proba = wrapper.predict(x)
        
        assert isinstance(proba, np.ndarray)


class TestVQCWrapperWithQiskit:
    """Tests for VQCWrapper with Qiskit integration."""
    
    def test_from_estimator_qnn_placeholder(self):
        """Placeholder test for Qiskit EstimatorQNN integration."""
        # This test documents how to integrate with Qiskit
        # Actual integration requires qiskit-machine-learning
        
        # Example usage (pseudocode):
        # from qiskit_machine_learning.neural_networks import EstimatorQNN
        # from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        # 
        # qnn = EstimatorQNN(...)
        # wrapper = VQCWrapper(
        #     predict_fn=lambda theta, x: qnn.forward(x, theta),
        #     theta=theta_trained,
        #     n_features=qnn.num_inputs,
        #     n_classes=2
        # )
        pass
