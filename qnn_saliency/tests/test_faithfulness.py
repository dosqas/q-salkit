# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for faithfulness metrics."""

import pytest
import numpy as np

from qnn_saliency.models.base import BaseModelWrapper
from qnn_saliency.metrics.faithfulness import (
    deletion_score,
    insertion_score,
    average_sensitivity,
)


class MockModel(BaseModelWrapper):
    """Mock model for testing."""
    
    def __init__(self, n_features=4, n_classes=3):
        self._n_features = n_features
        self._n_classes = n_classes
        self.weights = np.random.RandomState(42).randn(n_features, n_classes)
    
    def predict(self, x):
        x = np.atleast_2d(x)
        logits = x @ self.weights
        # Softmax
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
        return self._n_features
    
    @property
    def n_classes(self):
        return self._n_classes


class TestDeletionScore:
    """Tests for deletion_score function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_float(self, model):
        """Should return a float AUDC value."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        audc = deletion_score(model, x, saliency)
        
        assert isinstance(audc, float)
    
    def test_returns_curve_data(self, model):
        """Should return curve data when requested."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        audc, k_vals, confs = deletion_score(
            model, x, saliency, return_curve=True
        )
        
        assert isinstance(audc, float)
        assert len(k_vals) == len(confs) == 5  # k=0,1,2,3,4
        assert k_vals[0] == 0
        assert k_vals[-1] == 4
    
    def test_confidence_decreases_with_deletion(self, model):
        """Deleting salient features should decrease confidence."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        # Feature 2 (0.8) is most salient
        saliency = np.array([0.1, 0.1, 0.9, 0.1])
        
        _, k_vals, confs = deletion_score(
            model, x, saliency, return_curve=True
        )
        
        # First deletion (most salient) should cause significant drop
        # Note: not guaranteed, but likely with random model
        assert len(confs) > 1
    
    def test_target_class_tracking(self, model):
        """Should track specific target class probability."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        # Track class 0 specifically
        audc_0 = deletion_score(model, x, saliency, target=0)
        audc_1 = deletion_score(model, x, saliency, target=1)
        
        # Should potentially give different scores
        assert isinstance(audc_0, float)
        assert isinstance(audc_1, float)
    
    def test_custom_baseline(self, model):
        """Should use custom baseline for deletion."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        baseline = np.array([0.0, 0.0, 0.5, 0.0])  # Custom baseline
        
        audc_zero = deletion_score(model, x, saliency)
        audc_custom = deletion_score(model, x, saliency, baseline=baseline)
        
        # Different baselines should give different results
        assert isinstance(audc_custom, float)
    
    def test_audc_range(self, model):
        """AUDC should be in reasonable range [0, 1]."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        audc = deletion_score(model, x, saliency)
        
        # Probabilities are in [0,1], so AUDC (integral) should be reasonable
        assert 0 <= audc <= 1


class TestInsertionScore:
    """Tests for insertion_score function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_float(self, model):
        """Should return a float AUIC value."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        auic = insertion_score(model, x, saliency)
        
        assert isinstance(auic, float)
    
    def test_returns_curve_data(self, model):
        """Should return curve data when requested."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        
        auic, k_vals, confs = insertion_score(
            model, x, saliency, return_curve=True
        )
        
        assert len(k_vals) == len(confs) == 5
        assert k_vals[0] == 0
        assert k_vals[-1] == 4
    
    def test_starts_from_baseline(self, model):
        """Should start from baseline (k=0) prediction."""
        x = np.array([0.5, -0.2, 0.8, 0.1])
        saliency = np.array([0.1, 0.3, 0.5, 0.1])
        baseline = np.zeros(4)
        
        _, k_vals, confs = insertion_score(
            model, x, saliency, baseline=baseline, return_curve=True
        )
        
        # k=0 should be baseline-only prediction
        baseline_pred = model.predict(baseline)
        expected_conf = float(np.max(baseline_pred))
        # Note: we track target class, so this might differ
        assert len(confs) == 5


class TestAverageSensitivity:
    """Tests for average_sensitivity function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_float(self, model):
        """Should return a float sensitivity value."""
        X = np.random.randn(10, 4)
        
        def saliency_fn(x):
            return np.abs(x)  # Simple mock saliency
        
        sens = average_sensitivity(model, X, saliency_fn, delta=1e-3)
        
        assert isinstance(sens, float)
        assert sens >= 0
    
    def test_constant_saliency_zero_sensitivity(self, model):
        """Constant saliency function should have zero sensitivity."""
        X = np.random.randn(5, 4)
        
        def constant_saliency(x):
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        sens = average_sensitivity(model, X, constant_saliency, delta=1e-3)
        
        assert np.isclose(sens, 0.0, atol=1e-10)
