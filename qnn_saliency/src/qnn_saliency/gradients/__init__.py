# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Gradients module - Gradient computation engines for QNNs."""

from .base import BaseGradientEngine
from .finite_difference import FiniteDifferenceGradient
from .parameter_shift import ParameterShiftGradient

__all__ = [
    "BaseGradientEngine",
    "FiniteDifferenceGradient",
    "ParameterShiftGradient",
]
