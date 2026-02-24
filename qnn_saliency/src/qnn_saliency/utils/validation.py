# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""Input validation utilities for QNN Saliency."""

from typing import Optional, Any, Union, Sequence
import numpy as np


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_array(
    x: Any,
    name: str = "input",
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[tuple] = None,
    min_size: Optional[int] = None,
    dtype: np.dtype = np.float64,
    allow_none: bool = False
) -> Optional[np.ndarray]:
    """
    Validate and convert input to numpy array.
    
    Parameters
    ----------
    x : Any
        Input to validate
    name : str
        Name for error messages
    expected_ndim : int, optional
        Expected number of dimensions
    expected_shape : tuple, optional
        Expected shape (use -1 for any dimension)
    min_size : int, optional
        Minimum total size
    dtype : np.dtype
        Target dtype for conversion
    allow_none : bool
        If True, None input returns None
        
    Returns
    -------
    np.ndarray or None
        Validated array
        
    Raises
    ------
    ValidationError
        If validation fails
    """
    if x is None:
        if allow_none:
            return None
        raise ValidationError(f"{name} cannot be None")
    
    try:
        arr = np.asarray(x, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} cannot be converted to array: {e}")
    
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValidationError(
            f"{name} must be {expected_ndim}D, got {arr.ndim}D with shape {arr.shape}"
        )
    
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValidationError(
                    f"{name} shape mismatch at dimension {i}: "
                    f"expected {expected}, got {actual}"
                )
    
    if min_size is not None and arr.size < min_size:
        raise ValidationError(
            f"{name} must have at least {min_size} elements, got {arr.size}"
        )
    
    return arr


def validate_positive(
    value: Union[int, float],
    name: str,
    allow_zero: bool = False
) -> Union[int, float]:
    """Validate that value is positive (or non-negative)."""
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive: bool = True
) -> Union[int, float]:
    """Validate that value is within range."""
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValidationError(f"{name} must be > {min_val}, got {value}")
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValidationError(f"{name} must be < {max_val}, got {value}")
    
    return value


def validate_same_shape(
    arr1: np.ndarray,
    arr2: np.ndarray,
    name1: str = "array1",
    name2: str = "array2"
) -> None:
    """Validate that two arrays have the same shape."""
    if arr1.shape != arr2.shape:
        raise ValidationError(
            f"{name1} and {name2} must have same shape: "
            f"{arr1.shape} != {arr2.shape}"
        )


def validate_feature_count(
    x: np.ndarray,
    expected: int,
    name: str = "input"
) -> None:
    """Validate that input has expected number of features."""
    actual = x.shape[-1] if x.ndim > 0 else 1
    if actual != expected:
        raise ValidationError(
            f"{name} has {actual} features, expected {expected}"
        )


def validate_model(model: Any, name: str = "model") -> None:
    """Validate that model has required interface."""
    required_methods = ['predict', 'predict_class', 'n_features']
    
    for method in required_methods:
        if not hasattr(model, method):
            raise ValidationError(
                f"{name} missing required attribute/method: {method}"
            )


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (add batch dimension if needed)."""
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def ensure_1d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 1D (flatten if needed)."""
    return x.flatten()
