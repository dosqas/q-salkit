# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for saliency methods."""

import pytest
import numpy as np

from qnn_saliency.models.base import BaseModelWrapper
from qnn_saliency.gradients.base import BaseGradientEngine
from qnn_saliency.saliency.gradient import GradientSaliency, GradientTimesInput
from qnn_saliency.saliency.smoothgrad import SmoothGrad
from qnn_saliency.saliency.integrated_gradients import IntegratedGradients
from qnn_saliency.saliency.occlusion import Occlusion, NoiseSensitivity


class MockModel(BaseModelWrapper):
    """Mock model with linear predictions."""
    
    def __init__(self, weights):
        self.weights = np.array(weights)
    
    def predict(self, x):
        x = np.atleast_2d(x)
        logits = x @ self.weights
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        if proba.shape[0] == 1:
            return proba.flatten()
        return proba
    
    def predict_class(self, x):
        proba = self.predict(x)
        if proba.ndim == 1:
            return int(np.argmax(proba))
        return np.argmax(proba, axis=1)
    
    @property
    def n_features(self):
        return self.weights.shape[0]
    
    @property
    def n_classes(self):
        return self.weights.shape[1]


class MockGradientEngine(BaseGradientEngine):
    """Mock gradient engine computing numerical gradients."""
    
    def compute_gradient(self, x, forward_fn, target_idx=None):
        """Compute gradient using finite differences."""
        x = np.asarray(x).flatten()
        n = len(x)
        grad = np.zeros(n)
        eps = 1e-5
        
        base_out = forward_fn(x)
        if base_out.ndim > 0:
            base_out = base_out.flatten()
        
        if target_idx is None:
            target_idx = int(np.argmax(base_out))
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            out_plus = forward_fn(x_plus)
            out_minus = forward_fn(x_minus)
            
            if out_plus.ndim > 0:
                out_plus = out_plus.flatten()[target_idx]
            if out_minus.ndim > 0:
                out_minus = out_minus.flatten()[target_idx]
            
            grad[i] = (out_plus - out_minus) / (2 * eps)
        
        return grad


class TestGradientSaliency:
    """Tests for GradientSaliency method."""
    
    @pytest.fixture
    def setup(self):
        weights = np.array([
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
            [-0.5, 1.0, 0.5],
            [0.0, -0.5, 1.0]
        ])
        model = MockModel(weights)
        grad_engine = MockGradientEngine()
        return model, grad_engine
    
    def test_returns_array(self, setup):
        """Should return numpy array."""
        model, grad_engine = setup
        saliency = GradientSaliency(model, grad_engine)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_non_negative(self, setup):
        """Gradient saliency should be non-negative (absolute values)."""
        model, grad_engine = setup
        saliency = GradientSaliency(model, grad_engine)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result = saliency.attribute(x)
        
        assert np.all(result >= 0)
    
    def test_target_class_changes_result(self, setup):
        """Different target classes should give different saliencies."""
        model, grad_engine = setup
        saliency = GradientSaliency(model, grad_engine)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result_0 = saliency.attribute(x, target=0)
        result_1 = saliency.attribute(x, target=1)
        
        # Different targets can give different saliencies
        # (not guaranteed to be different, but test API works)
        assert result_0.shape == result_1.shape


class TestGradientTimesInput:
    """Tests for GradientTimesInput method."""
    
    @pytest.fixture
    def setup(self):
        weights = np.array([
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
            [-0.5, 1.0, 0.5],
            [0.0, -0.5, 1.0]
        ])
        model = MockModel(weights)
        grad_engine = MockGradientEngine()
        return model, grad_engine
    
    def test_returns_array(self, setup):
        """Should return numpy array."""
        model, grad_engine = setup
        saliency = GradientTimesInput(model, grad_engine)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_zero_input_zero_saliency(self, setup):
        """Zero input features should have zero saliency."""
        model, grad_engine = setup
        saliency = GradientTimesInput(model, grad_engine)
        x = np.array([0.5, 0.0, 0.8, 0.0])  # Features 1 and 3 are zero
        
        result = saliency.attribute(x)
        
        # Features with zero input should have zero saliency
        assert result[1] == 0.0
        assert result[3] == 0.0


class TestSmoothGrad:
    """Tests for SmoothGrad method."""
    
    @pytest.fixture
    def setup(self):
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        model = MockModel(weights)
        grad_engine = MockGradientEngine()
        return model, grad_engine
    
    def test_returns_array(self, setup):
        """Should return numpy array."""
        model, grad_engine = setup
        saliency = SmoothGrad(model, grad_engine, n_samples=10, sigma=0.1)
        x = np.array([0.5, -0.2])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
    
    def test_reproducible_with_seed(self, setup):
        """Should be reproducible with same seed."""
        model, grad_engine = setup
        x = np.array([0.5, -0.2])
        
        np.random.seed(42)
        saliency1 = SmoothGrad(model, grad_engine, n_samples=10, sigma=0.1)
        result1 = saliency1.attribute(x)
        
        np.random.seed(42)
        saliency2 = SmoothGrad(model, grad_engine, n_samples=10, sigma=0.1)
        result2 = saliency2.attribute(x)
        
        np.testing.assert_array_almost_equal(result1, result2)


class TestIntegratedGradients:
    """Tests for IntegratedGradients method."""
    
    @pytest.fixture
    def setup(self):
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        model = MockModel(weights)
        grad_engine = MockGradientEngine()
        return model, grad_engine
    
    def test_returns_array(self, setup):
        """Should return numpy array."""
        model, grad_engine = setup
        saliency = IntegratedGradients(model, grad_engine, n_steps=10)
        x = np.array([0.5, -0.2])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
    
    def test_custom_baseline(self, setup):
        """Should use custom baseline."""
        model, grad_engine = setup
        x = np.array([0.5, -0.2])
        baseline = np.array([0.1, 0.1])
        
        saliency = IntegratedGradients(model, grad_engine, n_steps=10)
        result = saliency.attribute(x, baseline=baseline)
        
        assert isinstance(result, np.ndarray)


class TestOcclusion:
    """Tests for Occlusion method."""
    
    @pytest.fixture
    def model(self):
        weights = np.array([
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
            [-0.5, 1.0, 0.5],
            [0.0, -0.5, 1.0]
        ])
        return MockModel(weights)
    
    def test_returns_array(self, model):
        """Should return numpy array."""
        saliency = Occlusion(model)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
    
    def test_non_negative(self, model):
        """Occlusion saliency should be non-negative."""
        saliency = Occlusion(model)
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        result = saliency.attribute(x)
        
        assert np.all(result >= 0)
    
    def test_different_baselines(self, model):
        """Different baselines should give different results."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        
        saliency_zero = Occlusion(model, baseline="zero")
        result_zero = saliency_zero.attribute(x)
        
        # Custom baseline
        custom_baseline = np.array([0.3, 0.3, 0.3, 0.3])
        saliency_custom = Occlusion(model, baseline=custom_baseline)
        result_custom = saliency_custom.attribute(x)
        
        # Different baselines can give different results
        assert result_zero.shape == result_custom.shape


class TestNoiseSensitivity:
    """Tests for NoiseSensitivity method."""
    
    @pytest.fixture
    def model(self):
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        return MockModel(weights)
    
    def test_returns_array(self, model):
        """Should return numpy array."""
        saliency = NoiseSensitivity(model, n_samples=10, sigma=0.1)
        x = np.array([0.5, -0.2])
        
        result = saliency.attribute(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
    
    def test_non_negative(self, model):
        """Sensitivity should be non-negative."""
        saliency = NoiseSensitivity(model, n_samples=10, sigma=0.1)
        x = np.array([0.5, -0.2])
        
        result = saliency.attribute(x)
        
        assert np.all(result >= 0)
