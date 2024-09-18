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
    if not isinstance(r, (float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be a float or numpy float type), got "
            f"{type(r).__name__}"
        )
    if not (np.isfinite(r)):
        raise ValueError(f"{name} must be a finite number, got {r}")
    if not (np.isfinite(r**2)):
        raise ValueError(f"{name}**2 must be a finite number, got {r}")
    if r <= 0:
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
        try:
            # Attempt to convert to NumPy array
            radii = np.asarray(radii)
            warnings.warn(
                f"{name} was not a numpy array. "
                "It is recommended to use numpy arrays for better performance.",
                UserWarning,
            )
        except Exception as e:
            raise TypeError(
                f"{name} must be an array, got {type(radii).__name__}"
            )

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
    if not (np.all(np.isfinite(radii**2))):
        raise ValueError(
            f"All elements to the power of two in {name} must be a finite"
            f"number"
        )
    if not np.all(radii > 0):
        raise ValueError(f"All elements in {name} must be positive numbers")
    pass
