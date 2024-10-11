import numpy as np


def cte(c: float):
    """
    Constant function.

    Parameters
    ----------
    c: float
        constant parameter.

    Returns
    -------
    constant_function: Callable of one parameter

    """

    def constant_function(x: np.ndarray):
        return c

    return constant_function


def linear_function(coefficients: np.ndarray):
    """
    Linear function.

    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    coefficients: np.ndarray
        of floats, parameters of the linear function.

    Returns
    -------
    linear: Callable of one parameter
    """

    def linear(x: np.ndarray):
        assert len(coefficients) == len(
            x
        ), "Vector is not of the correct length"
        return np.dot(coefficients, x)

    return linear


def plane_wave(
    a: float, k: np.ndarray, r: np.ndarray, p: np.ndarray
) -> complex:
    """
    Plane wave function, input in three dimensions.

    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    a: float
        amplitude.
    k: np.ndarray
        wave vector.
    r: np.ndarray
        variable.
    p: np.ndarray

    Returns
    -------
    complex
        result of the plane wave function given the parameters and the
        vector variable.
    """
    return a * np.exp(np.dot(k, (r + p)) * 1j)


def callable_plane_wave(a: float, k: np.ndarray, p: np.ndarray):
    def plane_wave_function(x: np.ndarray):
        assert len(p) == len(x), "Vector is not of the correct length"
        return a * np.exp(np.dot(k, (x + p)) * 1j)

    return plane_wave_function


def point_source(r: np.ndarray, p: np.ndarray, sigma_e: float) -> float:
    """
    Returns 1. / (4 * pi * sigma_e * ||r - p||_2), where

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
    return 1.0 / (4 * np.pi * sigma_e * np.linalg.norm(r - p))


def laplace_almost_fundamental_function(
    r_1: np.ndarray, r_2: np.ndarray, p_1: np.ndarray, p_2: np.ndarray
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
    value = 1.0 / np.linalg.norm(p_1 + r_1 - (p_2 + r_2))
    return value
