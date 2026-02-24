# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""
QNN Saliency - Interpretability Toolkit for Quantum Neural Networks
====================================================================

This package provides saliency methods and faithfulness metrics
for understanding Quantum Neural Network predictions.

Modules
-------
models
    Model wrappers for VQC and Hybrid QNN architectures
gradients
    Gradient computation engines (parameter-shift, finite-difference)
saliency
    Saliency methods (Gradient, SmoothGrad, Integrated Gradients, Occlusion)
metrics
    Faithfulness and diagnostic metrics

Example
-------
>>> from qnn_saliency import GradientSaliency, VQCWrapper
>>> from qnn_saliency.metrics import deletion_score, saliency_entropy
>>>
>>> # Wrap your trained VQC
>>> model = VQCWrapper(predict_fn, theta=theta_star, n_features=4)
>>>
>>> # Compute saliency
>>> saliency = GradientSaliency(model)
>>> scores = saliency.attribute(X_test[0])
>>>
>>> # Evaluate faithfulness
>>> deletion = deletion_score(model, X_test[0], scores)
"""

__version__ = "0.1.0"

# Model wrappers
from .models import BaseModelWrapper, VQCWrapper, HybridQNNWrapper

# Gradient engines
from .gradients import (
    BaseGradientEngine,
    FiniteDifferenceGradient,
    ParameterShiftGradient,
)

# Saliency methods
from .saliency import (
    BaseSaliency,
    GradientSaliency,
    GradientTimesInput,
    SmoothGrad,
    IntegratedGradients,
    Occlusion,
    NoiseSensitivity,
)

# Convenience imports from metrics
from .metrics import (
    deletion_score,
    insertion_score,
    saliency_entropy,
    saliency_sparseness,
    hoyer_sparseness,
    saliency_concentration,
    gini_coefficient,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "BaseModelWrapper",
    "VQCWrapper",
    "HybridQNNWrapper",
    # Gradients
    "BaseGradientEngine",
    "FiniteDifferenceGradient",
    "ParameterShiftGradient",
    # Saliency
    "BaseSaliency",
    "GradientSaliency",
    "GradientTimesInput",
    "SmoothGrad",
    "IntegratedGradients",
    "Occlusion",
    "NoiseSensitivity",
    # Metrics (common)
    "deletion_score",
    "insertion_score",
    "saliency_entropy",
    "saliency_sparseness",
    "hoyer_sparseness",
    "saliency_concentration",
    "gini_coefficient",
]
