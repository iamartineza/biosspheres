import numpy as np
import pytest
from scipy import special
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.reconstructions as reconstructions
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.crossinteractions as helmholtzcross
import biosspheres.helmholtz.selfinteractions as helmholtzself


def solve_mtf(
    big_l: int, radii: np.ndarray, centers: list, kii: np.ndarray, sigmas
) -> np.ndarray:
    n = len(radii)
    pii = sigmas[1:] / sigmas[0]
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    b = righthands.b_vector_n_spheres_mtf_plane_wave(
        n, big_l, centers, 0.0, kii[0], 1.0, radii, x_dia, mass_n_two
    )
    a_0_self, a_n = helmholtzself.a_0_a_n_sparse_matrices(
        n, big_l, radii, kii, azimuthal=False
    )
    cross = helmholtzcross.all_cross_interactions_n_spheres_2d(
        n, big_l, 2 * big_l + 10, kii[0], radii, centers
    )
    matrix = mtf.mtf_n_matrix(cross, a_0_self, a_n, x_dia, x_dia_inv)
    return np.linalg.solve(matrix, b)


def random_points(center: np.ndarray, r_min: float, r_max: float, number):
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(number, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    return center + directions * rng.uniform(r_min, r_max, (number, 1))


@pytest.mark.parametrize(
    "big_l, r, k0, k1, s0, s1",
    [(20, 1.3, 2.0, 3.1, 1.0, 1.7), (25, 0.8, 5.0, 1.5, 2.0, 0.6)],
)
def test_one_sphere_field_matches_mie_series(big_l, r, k0, k1, s0, s1) -> None:
    kii = np.asarray([k0, k1])
    coefficients = solve_mtf(
        big_l, np.asarray([r]), [np.zeros(3)], kii, np.asarray([s0, s1])
    )
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
    for r_min, r_max, amplitude, radial_function, k in [
        (1.05 * r, 3.0 * r, c, "h", k0),
        (0.05 * r, 0.95 * r, beta, "j", k1),
    ]:
        points = random_points(np.zeros(3), r_min, r_max, 200)
        rho = np.linalg.norm(points, axis=1)
        radial = special.spherical_jn(eles[:, np.newaxis], k * rho)
        if radial_function == "h":
            radial = radial + 1j * special.spherical_yn(
                eles[:, np.newaxis], k * rho
            )
        y_l0 = np.sqrt((2 * eles[:, np.newaxis] + 1) / (4 * np.pi)) * (
            special.eval_legendre(eles[:, np.newaxis], points[:, 2] / rho)
        )
        exact = np.sum(amplitude[:, np.newaxis] * radial * y_l0, axis=0)
        u = reconstructions.rf_helmholtz_n_spheres(
            points, 1, np.asarray([r]), [np.zeros(3)], kii, big_l, coefficients
        )
        assert np.max(np.abs(u - exact)) / np.max(np.abs(exact)) < 1e-12
    pass


def test_phantom_spheres_field_is_the_incident_wave() -> None:
    big_l = 14
    radii = np.asarray([1.0, 0.7, 0.9])
    centers = [
        np.zeros(3),
        np.asarray([2.5, 0.0, 0.5]),
        np.asarray([-0.4, 2.2, -1.0]),
    ]
    kii = np.full(4, 2.0)
    coefficients = solve_mtf(big_l, radii, centers, kii, np.ones(4))
    outside = np.asarray([[0.0, 0.0, 4.0], [5.0, 1.0, -1.0], [1.2, 1.2, 0.0]])
    u_outside = reconstructions.rf_helmholtz_n_spheres(
        outside, 3, radii, centers, kii, big_l, coefficients
    )
    assert np.max(np.abs(u_outside)) < 1e-10
    inside = np.concatenate(
        [random_points(c, 0.1 * r, 0.9 * r, 30) for c, r in zip(centers, radii)]
        + [np.asarray(centers)]
    )
    u_inside = reconstructions.rf_helmholtz_n_spheres(
        inside, 3, radii, centers, kii, big_l, coefficients
    )
    incident = np.exp(1j * kii[0] * inside[:, 2])
    assert np.max(np.abs(u_inside - incident)) < 1e-10
    pass


def test_total_field_is_continuous_across_the_spheres() -> None:
    big_l = 12
    radii = np.asarray([1.0, 0.7, 0.9, 0.6])
    centers = [
        np.zeros(3),
        np.asarray([2.5, 0.0, 0.5]),
        np.asarray([-0.4, 2.2, -1.0]),
        np.asarray([0.3, -1.1, 2.0]),
    ]
    kii = np.asarray([2.0, 3.0, 1.5, 2.5, 4.0])
    coefficients = solve_mtf(big_l, radii, centers, kii, np.ones(5))
    rng = np.random.default_rng(2)
    for c, r in zip(centers, radii):
        directions = rng.normal(size=(20, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        inner = c + (1.0 - 1e-7) * r * directions
        outer = c + (1.0 + 1e-7) * r * directions
        u_inner = reconstructions.rf_helmholtz_n_spheres(
            inner, 4, radii, centers, kii, big_l, coefficients
        )
        u_outer = reconstructions.rf_helmholtz_n_spheres(
            outer, 4, radii, centers, kii, big_l, coefficients
        ) + np.exp(1j * kii[0] * outer[:, 2])
        assert np.max(np.abs(u_inner - u_outer)) < 5e-4
    pass
