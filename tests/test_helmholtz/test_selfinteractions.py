"""
Tests for biosspheres.helmholtz.selfinteractions.

The analytic references assume the kernel
exp(i k |x - y|) / (4 pi |x - y|) and spherical harmonics orthonormal
on the unit sphere. With them, for a sphere of radius r and rho < r,
the single and double layer potentials of Y_l,0 at rho x are
i k r**2 j_l(k rho) h_l(k r) Y_l,0(x) and
i k**2 r**2 j_l(k rho) h_l'(k r) Y_l,0(x), respectively.

"""

import numpy as np
import pytest
from scipy import special
import biosspheres.helmholtz.selfinteractions as selfinteractions
import biosspheres.miscella.extensions as extensions
import biosspheres.quadratures.sphere as quadratures

CONFIGURATIONS = [
    (10, 0.3, 1.3),
    (10, 1.3, 2.0),
    (15, 2.5, 7.0),
    (25, 1.0, 11.0),
]


def spherical_functions(
    big_l: int, r: float, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spherical Bessel and Hankel functions of first kind, and their
    derivatives, evaluated at k r for degrees 0 to big_l.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.

    Returns
    -------
    j_l : np.ndarray
    j_l_d : np.ndarray
    h_l : np.ndarray
    h_l_d : np.ndarray

    """
    eles = np.arange(0, big_l + 1)
    j_l = special.spherical_jn(eles, k * r)
    j_l_d = special.spherical_jn(eles, k * r, derivative=True)
    h_l = j_l + 1j * special.spherical_yn(eles, k * r)
    h_l_d = j_l_d + 1j * special.spherical_yn(eles, k * r, derivative=True)
    return j_l, j_l_d, h_l, h_l_d


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("big_l, r, k", CONFIGURATIONS)
def test_self_interactions_closed_forms(big_l: int, r: float, k: float) -> None:
    j_l, j_l_d, h_l, h_l_d = spherical_functions(big_l, r, k)
    v = 1j * k * r**4 * j_l * h_l
    k_1 = 0.5j * k**2 * r**4 * (j_l * h_l_d + j_l_d * h_l)
    w = -1j * k**3 * r**4 * j_l_d * h_l_d
    assert (
        relative_error(selfinteractions.v_jj_azimuthal_symmetry(big_l, r, k), v)
        < 2e-14
    )
    assert (
        relative_error(
            selfinteractions.k_1_jj_azimuthal_symmetry(big_l, r, k), k_1
        )
        < 2e-14
    )
    assert (
        relative_error(
            selfinteractions.k_0_jj_azimuthal_symmetry(big_l, r, k), -k_1
        )
        < 2e-14
    )
    assert (
        relative_error(selfinteractions.w_jj_azimuthal_symmetry(big_l, r, k), w)
        < 2e-14
    )
    l2_1 = 2 * np.arange(0, big_l + 1) + 1
    assert (
        relative_error(
            selfinteractions.bio_jj(
                big_l, r, k, selfinteractions.v_jj_azimuthal_symmetry
            ),
            np.repeat(v, l2_1),
        )
        < 2e-14
    )
    pass


@pytest.mark.parametrize("big_l, r, k", CONFIGURATIONS)
def test_layer_potentials_by_quadrature(big_l: int, r: float, k: float) -> None:
    rho = 0.5 * r
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(2 * big_l + 40)
    )
    y = r * pre_vector
    weights = r**2 * total_weights
    x_hat = np.asarray([[0.3, -0.4, np.sqrt(0.75)], [0.6, 0.0, -0.8]]).T
    x = rho * x_hat
    difference = x[:, :, np.newaxis] - y[:, np.newaxis, :]
    distance = np.linalg.norm(difference, axis=0)
    kernel = np.exp(1j * k * distance) / (4.0 * np.pi * distance)
    normal_derivative = (
        kernel
        * (1j * k - 1.0 / distance)
        * np.sum(-difference * pre_vector[:, np.newaxis, :], axis=0)
        / distance
    )
    eles = np.arange(0, big_l + 1)
    j_l, j_l_d, h_l, h_l_d = spherical_functions(big_l, r, k)
    j_l_rho = special.spherical_jn(eles, k * rho)
    normalization = np.sqrt((2 * eles + 1) / (4.0 * np.pi))
    y_l_0 = normalization[:, np.newaxis] * special.eval_legendre(
        eles[:, np.newaxis], pre_vector[2, np.newaxis, :]
    )
    y_l_0_x = normalization[:, np.newaxis] * special.eval_legendre(
        eles[:, np.newaxis], x_hat[2, np.newaxis, :]
    )
    single = (y_l_0 * weights) @ kernel.T
    double = (y_l_0 * weights) @ normal_derivative.T
    single_reference = (1j * k * r**2 * j_l_rho * h_l)[:, np.newaxis] * y_l_0_x
    double_reference = (1j * k**2 * r**2 * j_l_rho * h_l_d)[
        :, np.newaxis
    ] * y_l_0_x
    assert relative_error(single, single_reference) < 2e-12
    assert relative_error(double, double_reference) < 2e-12
    pass


@pytest.mark.parametrize("big_l, r, k", CONFIGURATIONS)
@pytest.mark.parametrize("azimuthal", [True, False])
@pytest.mark.parametrize(
    "linear_operator, matrix",
    [
        (selfinteractions.a_0j_linear_operator, selfinteractions.a_0j_matrix),
        (selfinteractions.a_j_linear_operator, selfinteractions.a_j_matrix),
    ],
)
def test_big_a_linear_operator_matches_matrix(
    big_l: int,
    r: float,
    k: float,
    azimuthal: bool,
    linear_operator,
    matrix,
) -> None:
    num = big_l + 1 if azimuthal else (big_l + 1) ** 2
    rng = np.random.default_rng(0)
    b = rng.random(2 * num) + 1j * rng.random(2 * num)
    operator = linear_operator(big_l, r, k, azimuthal)
    dense = matrix(big_l, r, k, azimuthal)
    assert relative_error(operator.matvec(b), dense @ b) < 1e-14
    assert relative_error(operator.rmatvec(b), np.conj(dense.T) @ b) < 1e-14
    pass


@pytest.mark.parametrize("big_l, r, k", CONFIGURATIONS)
@pytest.mark.parametrize(
    "linear_operator",
    [
        selfinteractions.a_0j_linear_operator,
        selfinteractions.a_j_linear_operator,
    ],
)
def test_big_a_azimuthal_matches_general(
    big_l: int, r: float, k: float, linear_operator
) -> None:
    num = big_l + 1
    rng = np.random.default_rng(1)
    b = rng.random(2 * num) + 1j * rng.random(2 * num)

    def extend(u: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                extensions.azimuthal_trace_to_general_with_zeros(
                    big_l, u[0:num]
                ),
                extensions.azimuthal_trace_to_general_with_zeros(
                    big_l, u[num : 2 * num]
                ),
            )
        )

    azimuthal = linear_operator(big_l, r, k, True)
    general = linear_operator(big_l, r, k, False)
    assert (
        relative_error(extend(azimuthal.matvec(b)), general.matvec(extend(b)))
        < 1e-14
    )
    pass
