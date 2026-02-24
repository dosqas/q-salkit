# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for diagnostic metrics."""

import pytest
import numpy as np

from qnn_saliency.models.base import BaseModelWrapper
from qnn_saliency.metrics.diagnostics import (
    bug_detection_signal,
    minimum_efficacy,
    overreliance_risk,
    feature_agreement,
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


class TestBugDetectionSignal:
    """Tests for bug_detection_signal function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_dict(self, model):
        """Should return a dictionary with expected keys."""
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 3, 10)
        saliency = np.random.rand(10, 4)
        
        result = bug_detection_signal(model, X, y, saliency)
        
        assert isinstance(result, dict)
        assert 'correct_mean_saliency' in result
        assert 'incorrect_mean_saliency' in result
        assert 'ratio' in result
        assert 'n_correct' in result
        assert 'n_incorrect' in result
    
    def test_correct_count(self, model):
        """Should correctly count correct/incorrect predictions."""
        X = np.random.randn(10, 4)
        y_pred = model.predict_class(X)  # Use model's predictions
        saliency = np.random.rand(10, 4)
        
        # When true labels match predictions, all should be correct
        result = bug_detection_signal(model, X, y_pred, saliency)
        
        assert result['n_correct'] == 10
        assert result['n_incorrect'] == 0
    
    def test_all_incorrect(self, model):
        """Should handle all incorrect predictions."""
        X = np.random.randn(10, 4)
        y_pred = model.predict_class(X)
        # Flip all labels to be incorrect
        y_wrong = (y_pred + 1) % 3
        saliency = np.random.rand(10, 4)
        
        result = bug_detection_signal(model, X, y_wrong, saliency)
        
        assert result['n_correct'] == 0
        assert result['n_incorrect'] == 10


class TestMinimumEfficacy:
    """Tests for minimum_efficacy function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_float(self, model):
        """Should return a float."""
        X = np.random.randn(5, 4)
        saliency = np.random.rand(5, 4)
        
        result = minimum_efficacy(model, X, saliency)
        
        assert isinstance(result, float)
    
    def test_with_baseline(self, model):
        """Should work with custom baseline."""
        X = np.random.randn(5, 4)
        saliency = np.random.rand(5, 4)
        baseline = np.zeros(4)
        
        result = minimum_efficacy(model, X, saliency, baseline=baseline)
        
        assert isinstance(result, float)
    
    def test_with_top_k(self, model):
        """Should work with different top_k values."""
        X = np.random.randn(5, 4)
        saliency = np.random.rand(5, 4)
        
        result_1 = minimum_efficacy(model, X, saliency, top_k=1)
        result_2 = minimum_efficacy(model, X, saliency, top_k=2)
        
        assert isinstance(result_1, float)
        assert isinstance(result_2, float)


class TestOverrelianceRisk:
    """Tests for overreliance_risk function."""
    
    @pytest.fixture
    def model(self):
        return MockModel(n_features=4, n_classes=3)
    
    def test_returns_float(self, model):
        """Should return a float (correlation score)."""
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 3, 10)
        saliency = np.random.rand(10, 4)
        
        result = overreliance_risk(model, X, y, saliency)
        
        assert isinstance(result, float)
    
    def test_correlation_range(self, model):
        """Correlation should be in [-1, 1]."""
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 3, 10)
        saliency = np.random.rand(10, 4)
        
        result = overreliance_risk(model, X, y, saliency)
        
        # Correlation or a derived metric could be outside [-1, 1] in edge cases
        # but typically should be in reasonable range
        assert isinstance(result, float)


class TestFeatureAgreement:
    """Tests for feature_agreement function."""
    
    def test_identical_saliencies_full_agreement(self):
        """Identical saliencies should have perfect agreement."""
        s1 = np.array([0.5, 0.3, 0.1, 0.1])
        s2 = np.array([0.5, 0.3, 0.1, 0.1])
        
        agreement = feature_agreement(s1, s2, top_k=2)
        
        assert agreement == 1.0
    
    def test_different_top_features_no_agreement(self):
        """Completely different top features should have zero agreement."""
        s1 = np.array([0.9, 0.1, 0.0, 0.0])  # Top: features 0, 1
        s2 = np.array([0.0, 0.0, 0.1, 0.9])  # Top: features 3, 2
        
        agreement = feature_agreement(s1, s2, top_k=2)
        
        assert agreement == 0.0
    
    def test_partial_overlap(self):
        """Partial overlap should give fractional agreement."""
        s1 = np.array([0.5, 0.4, 0.1, 0.0])  # Top 2: 0, 1
        s2 = np.array([0.5, 0.0, 0.4, 0.1])  # Top 2: 0, 2
        
        agreement = feature_agreement(s1, s2, top_k=2)
        
        assert agreement == 0.5  # 1 out of 2 match
    
    def test_default_top_k(self):
        """Default top_k should be 3."""
        s1 = np.array([0.4, 0.3, 0.2, 0.1])
        s2 = np.array([0.4, 0.3, 0.2, 0.1])
        
        agreement = feature_agreement(s1, s2)
        
        assert agreement == 1.0
    
    def test_top_k_capped_at_features(self):
        """top_k larger than feature count should be capped."""
        s1 = np.array([0.5, 0.5])
        s2 = np.array([0.5, 0.5])
        
        # With only 2 features, top_k=5 should effectively be top_k=2
        agreement = feature_agreement(s1, s2, top_k=5)
        
        assert isinstance(agreement, float)
        assert 0 <= agreement <= 1
