# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""pytest configuration for qnn_saliency tests."""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_saliency():
    """Sample saliency values for testing."""
    return np.array([0.5, 0.3, 0.15, 0.05])


@pytest.fixture
def uniform_saliency():
    """Uniform saliency values for testing."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def sparse_saliency():
    """Sparse saliency values (one dominant feature)."""
    return np.array([0.95, 0.02, 0.02, 0.01])
