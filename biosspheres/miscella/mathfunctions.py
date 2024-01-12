import numpy as np


def cte(
        c: float
) -> float:
    return c


def linear_function_3d(
        coefficients: np.ndarray,
        vector: np.ndarray
) -> float:
    return np.dot(coefficients, vector)


def plane_wave(
        a: float,
        k: np.ndarray,
        r: np.ndarray,
        p: np.ndarray
) -> complex:
    return a*np.exp(np.dot(k, (r + p)) * 1j)


def point_source(
        r: np.ndarray,
        p: np.ndarray,
        sigma_e: float
) -> float:
    """
    Returns 1. / (4 * pi * sigma_e * ||r - p||_2)
    
    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    r : numpy array
        of length 3 in the cartesian coordinate system. Position vector
        to be evaluated.
    p : numpy array
        of length 3 in the cartesian coordinate system. Position vector
        of the point source.
    sigma_e : float
        > 0.

    Returns
    -------
    float
    
    """
    return 1. / (4 * np.pi * sigma_e * np.linalg.norm(r - p))


def laplace_almost_fundamental_function(
        r_1: np.ndarray, r_2: np.ndarray,
        p_1: np.ndarray, p_2: np.ndarray
) -> float:
    """
    Evaluating in two DIFFERENT points.
    
    Parameters
    ----------
    r_1
    r_2
    p_1
    p_2

    Returns
    -------
    value : float
    
    """
    value = 1. / np.linalg.norm(p_1 + r_1 - (p_2 + r_2))
    return value
