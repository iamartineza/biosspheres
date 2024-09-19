import numpy as np
import warnings


def big_l_validation(big_l: int, name: str) -> None:
    if not isinstance(big_l, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer or numpy integer type,"
            f"got {type(big_l).__name__}"
        )
    if big_l < 0:
        raise ValueError(f"{name} must be non-negative, got {big_l}")
    if big_l > 3000:
        warnings.warn(
            f"{name} is {big_l}, there might be issues with "
            f"{name} being too big.",
            UserWarning,
        )
    pass


def radius_validation(r: float, name: str) -> None:
    if not isinstance(r, (float, np.floating)):
        raise TypeError(
            f"{name} must be a float or numpy float type), got "
            f"{type(r).__name__}"
        )
    if not (np.isfinite(r)):
        raise ValueError(f"{name} must be a finite number, got {r}")
    if r <= 0.0:
        raise ValueError(f"{name} must be positive, got {r}")
    pass


def bool_validation(b: bool, name: str) -> None:
    if not isinstance(b, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a boolean or numpy boolean type, got "
            f"{type(b).__name__}"
        )
    pass


def radii_validation(radii: np.ndarray, name: str) -> None:
    if not isinstance(radii, np.ndarray):
        raise TypeError(f"{name} must be an array, got {type(radii).__name__}")
    if not issubclass(radii.dtype.type, np.floating):
        raise TypeError(
            f"{name} must be an array of floats, got array with dtype"
            f"{radii.dtype}"
        )
    if radii.size == 0:
        raise ValueError(f"{name} array must not be empty")
    if radii.ndim != 1:
        raise ValueError(
            f"Expected '{name}' to be one-dimensional, but got {radii.ndim}"
            f"dimensions."
        )
    if not (np.all(np.isfinite(radii))):
        raise ValueError(f"All elements in {name} must be a finite number")
    if not np.all(radii > 0.0):
        raise ValueError(f"All elements in {name} must be positive numbers")
    pass


def pi_validation(pi: float, name: str) -> None:
    if not isinstance(pi, (float, np.floating)):
        raise TypeError(
            f"{name} must be a float or numpy float type), got "
            f"{type(pi).__name__}"
        )
    if not (np.isfinite(pi)):
        raise ValueError(f"{name} must be a finite number, got {pi}")
    if np.abs(pi) == np.abs(0.0):
        raise ValueError(f"{name} must be different from 0, got {pi}")
    pass


def pii_validation(pii: np.ndarray, name: str) -> None:
    if not isinstance(pii, np.ndarray):
        raise TypeError(f"{name} must be an array, got {type(pii).__name__}")

    if not issubclass(pii.dtype.type, np.floating):
        raise TypeError(
            f"{name} must be an array of floats, got array with dtype"
            f"{pii.dtype}"
        )
    if pii.size == 0:
        raise ValueError(f"{name} array must not be empty")
    if pii.ndim != 1:
        raise ValueError(
            f"Expected '{name}' to be one-dimensional, but got {pii.ndim}"
            f"dimensions."
        )
    if not (np.all(np.isfinite(pii))):
        raise ValueError(f"All elements in {name} must be a finite number")
    if np.any(np.abs(pii) == np.abs(0.0)):
        raise ValueError(f"All elements in {name} must be non-zero")
    pass


def n_validation(n: int, name: str) -> None:
    if not isinstance(n, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer or numpy integer type,"
            f"got {type(n).__name__}"
        )
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {n}")
    pass


def two_dimensional_array_check(array: np.ndarray, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"{name} should be a numpy array, but got {type(array).__name__}."
        )
    if array.ndim != 2:
        raise ValueError(f"{name} should be 2D, but got {array.ndim}D.")
    pass


def square_array_check(array: np.ndarray, name: str) -> None:
    rows, cols = array.shape
    if rows != cols:
        raise ValueError(
            f"{name} is not square: {rows} rows vs {cols} columns."
        )
    pass
