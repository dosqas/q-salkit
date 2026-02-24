# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Utility functions for QNN Saliency."""

from .validation import (
    ValidationError,
    validate_array,
    validate_positive,
    validate_range,
    validate_same_shape,
    validate_feature_count,
    validate_model,
    ensure_2d,
    ensure_1d,
)

__all__ = [
    "ValidationError",
    "validate_array",
    "validate_positive",
    "validate_range",
    "validate_same_shape",
    "validate_feature_count",
    "validate_model",
    "ensure_2d",
    "ensure_1d",
]
