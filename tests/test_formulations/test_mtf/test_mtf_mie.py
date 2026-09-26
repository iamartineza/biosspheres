import numpy as np
import pytest
from scipy import special
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.selfinteractions as helmholtzself
import biosspheres.miscella.harmonicex as harmonicex


def mie_traces(
    big_l: int, r: float, k0: float, k1: float, s0: float, s1: float
) -> np.ndarray:
    eles = np.arange(0, big_l + 1)
    j0 = special.spherical_jn(eles, k0 * r)
    j0p = special.spherical_jn(eles, k0 * r, derivative=True)
    h0 = j0 + 1j * special.spherical_yn(eles, k0 * r)
    h0p = j0p + 1j * special.spherical_yn(eles, k0 * r, derivative=True)
    j1 = special.spherical_jn(eles, k1 * r)
    j1p = special.spherical_jn(eles, k1 * r, derivative=True)
    alpha = 2 * np.sqrt(np.pi) * 1j**eles * np.sqrt(2 * eles + 1)
    determinant = -h0 * s1 * k1 * j1p + j1 * s0 * k0 * h0p
    c = alpha * (j0 * s1 * k1 * j1p - j1 * s0 * k0 * j0p) / determinant
    beta = alpha * (-h0 * s0 * k0 * j0p + s0 * k0 * h0p * j0) / determinant
    return np.concatenate((c * h0, -k0 * c * h0p, beta * j1, k1 * beta * j1p))


@pytest.mark.parametrize(
    "big_l, r, k0, k1, s0, s1, tol",
    [
        (20, 1.3, 2.0, 3.1, 1.0, 1.7, 5e-14),
        (30, 0.8, 5.0, 1.5, 2.0, 0.6, 1e-13),
        (40, 2.0, 7.0, 9.0, 1.0, 1.0, 1e-12),
    ],
)
def test_mtf_one_sphere_matches_mie_series(
    big_l: int, r: float, k0: float, k1: float, s0: float, s1: float, tol
) -> None:
    pi = s1 / s0
    b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        big_l, r, 0.0, k0, 1.0, azimuthal=True
    )
    b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
        big_l, r, 0.0, k0, 1.0, azimuthal=True
    )
    matrix = mtf.mtf_1_matrix(
        r,
        pi,
        helmholtzself.a_0j_matrix(big_l, r, k0, azimuthal=True),
        helmholtzself.a_j_matrix(big_l, r, k1, azimuthal=True),
    )
    solution = np.linalg.solve(
        matrix, righthands.b_vector_1_sphere_mtf(r, 1.0 / pi, b_d, b_n)
    )
    exact = mie_traces(big_l, r, k0, k1, s0, s1)
    assert np.max(np.abs(solution - exact)) / np.max(np.abs(exact)) < tol
    pass
