"""
Tests for the spherical harmonic expansions of a point source and of a
plane wave in biosspheres.miscella.harmonicex: the truncated expansion
approximates the function on the sphere in the L^2 norm.

"""

import numpy as np
import pytest
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.mathfunctions as mathfunctions


def relative_error(
    grid: np.ndarray,
    coefficients: np.ndarray,
    big_l: int,
    total_weights: np.ndarray,
    cos_theta: np.ndarray,
) -> float:
    """
    Relative L^2 error on the sphere between the values grid and the
    expansion with coefficients of degrees 0 to big_l, computed with
    the quadrature of weights total_weights.

    Parameters
    ----------
    grid : np.ndarray
        values of the function at the quadrature points.
    coefficients : np.ndarray
        coefficients of the azimuthal expansion.
    big_l : int
        >= 0, max degree of the truncated expansion.
    total_weights : np.ndarray
        quadrature weights.
    cos_theta : np.ndarray
        cosine of the polar angle of the quadrature points.

    Returns
    -------
    float

    """
    unique_cos, inverse = np.unique(cos_theta, return_inverse=True)
    legendre = np.asarray(
        [pyshtools.legendre.PlON(big_l, c) for c in unique_cos]
    )
    expansion = (legendre @ coefficients[0 : big_l + 1])[inverse]
    return np.sqrt(
        np.sum(np.abs(grid - expansion) ** 2 * total_weights)
        / np.sum(np.abs(grid) ** 2 * total_weights)
    )


@pytest.mark.parametrize(
    "big_l, distance, tolerance",
    [
        (20, 50.0, 4e-14),
        (46, 20.0, 8e-14),
        (80, 15.0, 6e-14),
        (172, 12.0, 2e-13),
    ],
)
def test_point_source_expansion(
    big_l: int, distance: float, tolerance: float
) -> None:
    radius = 10.0
    sigma_e = 5.0
    p_0 = np.asarray([0.0, 0.0, distance])
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(2 * big_l)
    )
    grid = np.asarray(
        [
            mathfunctions.point_source(radius * pre_vector[:, ii], p_0, sigma_e)
            for ii in np.arange(0, final_length)
        ]
    )
    coefficients = harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        big_l, radius, distance, sigma_e, 1.0
    )
    assert (
        relative_error(
            grid, coefficients, big_l, total_weights, pre_vector[2, :]
        )
        < tolerance
    )
    pass


@pytest.mark.parametrize(
    "big_l, k, tolerance",
    [
        (16, 1.0, 4e-14),
        (34, 11.0, 9e-13),
        (52, 21.0, 2e-13),
        (66, 31.0, 2e-13),
        (80, 41.0, 3e-13),
    ],
)
def test_plane_wave_expansion(big_l: int, k: float, tolerance: float) -> None:
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l + 10)
    )
    wave_vector = np.asarray([0.0, 0.0, k])
    grid = np.asarray(
        [
            mathfunctions.plane_wave(
                1.0, wave_vector, pre_vector[:, ii], np.zeros(3)
            )
            for ii in np.arange(0, final_length)
        ]
    )
    coefficients = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        big_l, 1.0, 0.0, k, 1.0, True
    )
    assert (
        relative_error(
            grid, coefficients, big_l, total_weights, pre_vector[2, :]
        )
        < tolerance
    )
    pass
