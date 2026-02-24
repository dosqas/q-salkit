# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for distribution metrics."""

import pytest
import numpy as np

from qnn_saliency.metrics.distribution import (
    saliency_entropy,
    saliency_sparseness,
    hoyer_sparseness,
    saliency_concentration,
    gini_coefficient,
)


class TestSaliencyEntropy:
    """Tests for saliency_entropy function."""
    
    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        saliency = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = saliency_entropy(saliency)
        expected = np.log(4)  # Maximum entropy for 4 elements
        assert np.isclose(entropy, expected, rtol=0.01)
    
    def test_single_feature_zero_entropy(self):
        """Single non-zero feature should have zero entropy."""
        saliency = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = saliency_entropy(saliency)
        assert np.isclose(entropy, 0.0, atol=1e-6)
    
    def test_concentrated_low_entropy(self):
        """Concentrated distribution should have lower entropy."""
        concentrated = np.array([0.9, 0.05, 0.03, 0.02])
        spread = np.array([0.3, 0.25, 0.25, 0.2])
        
        entropy_concentrated = saliency_entropy(concentrated)
        entropy_spread = saliency_entropy(spread)
        
        assert entropy_concentrated < entropy_spread
    
    def test_handles_all_zeros(self):
        """All zeros should return maximum entropy."""
        saliency = np.zeros(4)
        entropy = saliency_entropy(saliency)
        assert np.isclose(entropy, np.log(4), rtol=0.01)
    
    def test_handles_negative_values(self):
        """Should work with absolute values."""
        saliency = np.array([-0.5, 0.3, -0.1, 0.1])
        entropy = saliency_entropy(saliency)
        assert 0 <= entropy <= np.log(4)


class TestSaliencySparseness:
    """Tests for saliency_sparseness function (L1/L2 ratio)."""
    
    def test_single_feature_minimum_sparseness(self):
        """Single non-zero feature should have minimum L1/L2 = 1."""
        saliency = np.array([1.0, 0.0, 0.0, 0.0])
        sparseness = saliency_sparseness(saliency)
        assert np.isclose(sparseness, 1.0, atol=1e-6)
    
    def test_uniform_maximum_sparseness(self):
        """Uniform distribution should have maximum L1/L2 = sqrt(n)."""
        saliency = np.array([1.0, 1.0, 1.0, 1.0])
        sparseness = saliency_sparseness(saliency)
        expected = 2.0  # sqrt(4)
        assert np.isclose(sparseness, expected, rtol=0.01)
    
    def test_concentrated_lower_sparseness(self):
        """More concentrated = lower L1/L2 ratio."""
        concentrated = np.array([0.9, 0.05, 0.03, 0.02])
        spread = np.array([0.3, 0.25, 0.25, 0.2])
        
        sparse_concentrated = saliency_sparseness(concentrated)
        sparse_spread = saliency_sparseness(spread)
        
        assert sparse_concentrated < sparse_spread
    
    def test_batch_input(self):
        """Should handle batch of saliency maps."""
        saliency = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0]
        ])
        sparseness = saliency_sparseness(saliency)
        assert isinstance(sparseness, float)


class TestHoyerSparseness:
    """Tests for hoyer_sparseness function (normalized)."""
    
    def test_single_feature_max_sparseness(self):
        """Single non-zero feature should have sparseness = 1."""
        saliency = np.array([1.0, 0.0, 0.0, 0.0])
        sparseness = hoyer_sparseness(saliency)
        assert np.isclose(sparseness, 1.0, atol=0.01)
    
    def test_uniform_zero_sparseness(self):
        """Uniform distribution should have sparseness = 0."""
        saliency = np.array([1.0, 1.0, 1.0, 1.0])
        sparseness = hoyer_sparseness(saliency)
        assert np.isclose(sparseness, 0.0, atol=0.01)
    
    def test_range_zero_to_one(self):
        """Sparseness should always be in [0, 1]."""
        for _ in range(10):
            saliency = np.random.rand(10)
            sparseness = hoyer_sparseness(saliency)
            assert 0 <= sparseness <= 1


class TestSaliencyConcentration:
    """Tests for saliency_concentration function."""
    
    def test_all_in_top_k(self):
        """If top-k has all saliency, concentration = 1."""
        saliency = np.array([1.0, 0.0, 0.0, 0.0])
        concentration = saliency_concentration(saliency, top_k=1)
        assert np.isclose(concentration, 1.0)
    
    def test_uniform_concentration(self):
        """Uniform distribution, top-1 should have 1/n concentration."""
        saliency = np.array([0.25, 0.25, 0.25, 0.25])
        concentration = saliency_concentration(saliency, top_k=1)
        assert np.isclose(concentration, 0.25)
    
    def test_top_k_increases_concentration(self):
        """Higher k should increase concentration."""
        saliency = np.array([0.4, 0.3, 0.2, 0.1])
        
        conc_1 = saliency_concentration(saliency, top_k=1)
        conc_2 = saliency_concentration(saliency, top_k=2)
        conc_3 = saliency_concentration(saliency, top_k=3)
        
        assert conc_1 < conc_2 < conc_3


class TestGiniCoefficient:
    """Tests for gini_coefficient function."""
    
    def test_perfect_equality_zero_gini(self):
        """Uniform distribution should have Gini = 0."""
        saliency = np.array([1.0, 1.0, 1.0, 1.0])
        gini = gini_coefficient(saliency)
        assert np.isclose(gini, 0.0, atol=0.05)
    
    def test_perfect_inequality_max_gini(self):
        """Single non-zero feature should have high Gini."""
        saliency = np.array([1.0, 0.0, 0.0, 0.0])
        gini = gini_coefficient(saliency)
        assert gini > 0.7  # Should be close to 1
    
    def test_range_zero_to_one(self):
        """Gini should always be in [0, 1]."""
        for _ in range(10):
            saliency = np.abs(np.random.rand(10))
            gini = gini_coefficient(saliency)
            assert 0 <= gini <= 1
