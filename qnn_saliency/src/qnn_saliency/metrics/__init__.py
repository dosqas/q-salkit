# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Metrics module - Faithfulness and diagnostic metrics for saliency evaluation."""

from .faithfulness import (
    deletion_score,
    insertion_score,
    average_sensitivity,
)
from .distribution import (
    saliency_entropy,
    saliency_sparseness,
    hoyer_sparseness,
    saliency_concentration,
    gini_coefficient,
)
from .diagnostics import (
    bug_detection_signal,
    minimum_efficacy,
    overreliance_risk,
    feature_agreement,
)

__all__ = [
    # Faithfulness
    "deletion_score",
    "insertion_score",
    "average_sensitivity",
    # Distribution
    "saliency_entropy",
    "saliency_sparseness",
    "hoyer_sparseness",
    "saliency_concentration",
    "gini_coefficient",
    # Diagnostics
    "bug_detection_signal",
    "minimum_efficacy",
    "overreliance_risk",
    "feature_agreement",
]
