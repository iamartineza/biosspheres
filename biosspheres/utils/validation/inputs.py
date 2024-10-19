import numpy as np
from scipy import sparse
import warnings


def integer_validation(integer: int, name: str) -> None:
    """To check if integer (named name) is an integer type."""
    if not isinstance(integer, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer or numpy integer type,"
            f"got {type(integer).__name__}"
        )
    pass


def big_l_validation(big_l: int, name: str) -> None:
    """
    To check if big_l (named name) represents the degree of a spherical
    harmonic.
    """
    integer_validation(big_l, name)
    if big_l < 0:
        raise ValueError(f"{name} must be non-negative, got {big_l}")
    if big_l > 3000:
        warnings.warn(
            f"{name} is {big_l}, there might be issues with "
            f"{name} being too big.",
            UserWarning,
        )
    pass


def n_validation(n: int, name: str) -> None:
    """Checks if n (named name) is a positive integer"""
    integer_validation(n, name)
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {n}")
    pass


def float_validation(fl: float, name: str) -> None:
    """To check if fl (named name) is a float type."""
    if not isinstance(fl, (float, np.floating)):
        raise TypeError(
            f"{name} must be a float or numpy float type,"
            f"got {type(fl).__name__}"
        )
    if not (np.isfinite(fl)):
        raise ValueError(f"{name} must be a finite number, got {fl}")
    pass


def radius_validation(r: float, name: str) -> None:
    """Checks if r is a positive float"""
    float_validation(r, name)
    if r <= 0.0:
        raise ValueError(f"{name} must be positive, got {r}")
    pass


def pi_validation(pi: float, name: str) -> None:
    """Checks if pi is a float different from 0"""
    float_validation(pi, name)
    if np.abs(pi) == np.abs(0.0):
        raise ValueError(f"{name} must be different from 0, got {pi}")
    pass


def bool_validation(b: bool, name: str) -> None:
    if not isinstance(b, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a boolean or numpy boolean type, got "
            f"{type(b).__name__}"
        )
    pass


def numpy_array_validation(array: np.ndarray, name: str) -> None:
    """To check if array (named name) is a numpy array."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be an array, got {type(array).__name__}")
    pass


def finite_values_in_array(array: np.ndarray, name: str) -> None:
    """To check if array (named name) has only finite values."""
    if not (np.all(np.isfinite(array))):
        raise ValueError(f"All elements in {name} must be a finite number")
    pass


def dimensions_array_validation(array: np.ndarray, name: str, dim: int) -> None:
    """To check if array (named name) is a numpy array of one dimension."""
    if array.ndim != dim:
        raise ValueError(
            f"Expected '{name}' to be '{dim}'-dimensional, but got {array.ndim}"
            f"dimensions."
        )
    pass


def float_array_validation(array: np.ndarray, name: str) -> None:
    """To check if array (named name) is a numpy array of floats."""
    if not issubclass(array.dtype.type, np.floating):
        raise TypeError(
            f"{name} must be an array of floats, got array with dtype"
            f"{array.dtype}"
        )
    pass


def full_float_array_validation(array: np.ndarray, name: str, dim: int) -> None:
    """To check if array (named name) is a dimD array with finite and
    real values."""
    numpy_array_validation(array, name)
    dimensions_array_validation(array, name, dim)
    float_array_validation(array, name)
    finite_values_in_array(array, name)
    pass


def trigonometric_arrays_validation(
    array: np.ndarray, name: str, dim: int
) -> None:
    """
    To check if array (named name) is an array that corresponds with a
    trigonometric function bounded between -1 and 1.
    """
    full_float_array_validation(array, name, 1)
    if not np.all(array >= -1.0):
        raise ValueError(
            f"All elements in {name} must be greater or equal to -1"
        )
    if not np.all(array <= 1.0):
        raise ValueError(
            f"All elements in {name} must be greater or equal to 1"
        )
    pass


def radii_validation(radii: np.ndarray, name: str) -> None:
    """Extension of radius_validation for an array of 1D"""
    full_float_array_validation(radii, name, 1)
    if radii.size == 0:
        raise ValueError(f"{name} array must not be empty")
    if not np.all(radii > 0.0):
        raise ValueError(f"All elements in {name} must be positive numbers")
    pass


def pii_validation(pii: np.ndarray, name: str) -> None:
    """Extension of pii_validation for an array of 1D"""
    full_float_array_validation(pii, name, 1)
    if pii.size == 0:
        raise ValueError(f"{name} array must not be empty")
    if np.any(np.abs(pii) == np.abs(0.0)):
        raise ValueError(f"All elements in {name} must be non-zero")
    pass


def one_dimensional_array_length_check(array, name, length):
    if len(array) != length:
        raise ValueError(f"{name} array has to be of length {length}")
    pass


def square_array_check(array: np.ndarray, name: str) -> None:
    """Checks if the provided array is a square array"""
    rows, cols = array.shape
    if rows != cols:
        raise ValueError(
            f"{name} is not square: {rows} rows vs {cols} columns."
        )
    pass


def same_shape_check(
    array1: np.ndarray, name1: str, array2: np.ndarray, name2: str
) -> None:
    """Checks if the provided arrays are of the same shape"""
    if array1.shape != array2.shape:
        raise ValueError(
            f"{name1} has different shape than {name2}: "
            f"{array1.shape} vs {array2.shape}."
        )
    pass


def same_type_check(
    array1: np.ndarray, name1: str, array2: np.ndarray, name2: str
) -> None:
    """Checks if the provided arrays are of the same type"""
    if array1.dtype != array2.dtype:
        raise ValueError(
            f"{name1} has different type than {name2}: "
            f"{array1.shape} vs {array2.shape}."
        )
    pass


def two_dimensional_array_check(array: np.ndarray, name: str) -> None:
    """All checks for a two-dimensional array."""
    numpy_array_validation(array, name)
    dimensions_array_validation(array, name, 2)
    finite_values_in_array(array, name)
    pass


def is_scipy_linear_op(
    linear_op: sparse.linalg.LinearOperator, name: str
) -> None:
    """Checks if the provided object is a scipy linear operator."""
    if not isinstance(linear_op, sparse.linalg.LinearOperator):
        raise TypeError(
            f"{name} should be a scipy.sparse.linalg.LinearOperator, "
            f"but received an object of type {type(linear_op).__name__}."
        )
    pass


def is_scipy_sparse_array(array: sparse.sparray, name: str) -> None:
    """Checks if the provided object is a scipy sparse array."""
    if not isinstance(array, sparse.sparray):
        raise TypeError(
            f"{name} should be a scipy sparse array, but received an"
            f"object of type {type(array).__name__}."
        )
    pass
