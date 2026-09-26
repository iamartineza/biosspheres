import numpy as np
import pytest
import scipy.special as special
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.crossinteractions as helmholtzcross
import biosspheres.helmholtz.selfinteractions as helmholtzself
import biosspheres.laplace.crossinteractions as laplacecross
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.miscella.extensions as extensions
import biosspheres.miscella.harmonicex as harmonicex


def one_sphere_errors(r, b_d, b_n, a_0, a_1) -> dict:
    num = len(b_d)
    b = righthands.b_vector_1_sphere_mtf(r, 1.0, b_d, b_n)
    solution = np.linalg.solve(mtf.mtf_1_matrix(r, 1.0, a_0, a_1), b)
    exterior = solution[0 : 2 * num]
    interior = solution[2 * num : 4 * num]
    return {
        "exterior_trace": np.linalg.norm(exterior),
        "interior_vs_phi_e": np.linalg.norm(
            np.concatenate((b_d, -b_n)) - interior
        ),
        "calderon_0": np.linalg.norm(2 * a_0 @ exterior - r**2 * exterior),
        "calderon_1": np.linalg.norm(2 * a_1 @ interior - r**2 * interior),
        "jump_dirichlet": np.linalg.norm(
            solution[0:num] - solution[2 * num : 3 * num] + b_d
        ),
        "jump_neumann": np.linalg.norm(
            solution[num : 2 * num] + b_n + solution[3 * num : 4 * num]
        ),
    }


def n_spheres_errors(
    big_l, big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv, mass_n_two, b, one
) -> dict:
    num = (big_l + 1) ** 2
    half = len(b) // 2
    matrix = mtf.mtf_n_matrix(big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv)
    solution = np.linalg.solve(matrix, b)
    exterior = solution[0:half]
    interior = solution[half:]
    one_sphere = np.concatenate(
        [
            extensions.azimuthal_trace_to_general_with_zeros(
                big_l, one[i * (big_l + 1) : (i + 1) * (big_l + 1)]
            )
            for i in range(4)
        ]
    )
    first_sphere = np.concatenate(
        (solution[0 : 2 * num], solution[half : half + 2 * num])
    )
    return {
        "calderon_0": np.linalg.norm(
            2.0 * (big_a_0_cross @ exterior + a_0_self.dot(exterior))
            - mass_n_two * exterior
        ),
        "calderon_n": np.linalg.norm(
            2.0 * a_n.dot(interior) - mass_n_two * interior
        ),
        "jump": np.linalg.norm(
            -exterior * x_dia + mass_n_two * interior - b[half:]
        ),
        "analytic_first_sphere": np.linalg.norm(one_sphere - first_sphere)
        / np.linalg.norm(one_sphere),
    }


@pytest.fixture(scope="module")
def laplace_one_sphere() -> dict:
    max_l = 50
    r = 1.3
    b_d = harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        max_l, r, 20.0, 1.0, 1.0
    )
    b_n = harmonicex.point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        max_l, r, 20.0, 1.0, 1.0
    )
    return one_sphere_errors(
        r,
        b_d,
        b_n,
        laplaceself.a_0j_matrix(max_l, r, azimuthal=True),
        laplaceself.a_j_matrix(max_l, r, azimuthal=True),
    )


@pytest.fixture(scope="module")
def helmholtz_one_sphere() -> dict:
    max_l = 20
    r = 1.3
    k0 = 2.0
    b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        max_l, r, 5.0, k0, 1.0, azimuthal=True
    )
    b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
        max_l, r, 5.0, k0, 1.0, azimuthal=True
    )
    return one_sphere_errors(
        r,
        b_d,
        b_n,
        helmholtzself.a_0j_matrix(max_l, r, k0, azimuthal=True),
        helmholtzself.a_j_matrix(max_l, r, k0, azimuthal=True),
    )


CENTERS = [
    np.asarray([0.0, 0.0, 0.0]),
    np.asarray([5.0, 0.0, 0.0]),
    np.asarray([-6.0, 0.0, 0.0]),
]
RADII = np.asarray([1.15, 1.2, 1.3])


@pytest.fixture(scope="module")
def laplace_three_spheres() -> dict:
    n = 3
    big_l = 10
    sigmas = np.asarray([1.0, 0.25, 1.0, 1.0])
    pii = sigmas[1:] / sigmas[0]
    p0 = np.asarray([0.0, 0.0, 20.0])
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, RADII, pii, azimuthal=False
    )
    mass_n_two = mass.n_two_j_blocks(big_l, RADII, azimuthal=False)
    b = righthands.b_vector_n_spheres_mtf_point_source(
        n, big_l, CENTERS, p0, RADII, sigmas[0], x_dia, mass_n_two
    )
    r = RADII[0]
    b_d = harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        big_l, r, 20.0, sigmas[0], 1.0
    )
    b_n = harmonicex.point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        big_l, r, 20.0, sigmas[0], 1.0
    )
    one = np.linalg.solve(
        mtf.mtf_1_matrix(
            r,
            pii[0],
            laplaceself.a_0j_matrix(big_l, r, azimuthal=True),
            laplaceself.a_j_matrix(big_l, r, azimuthal=True),
        ),
        righthands.b_vector_1_sphere_mtf(r, 1.0 / pii[0], b_d, b_n),
    )
    a_0_self, a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, RADII, azimuthal=False
    )
    return n_spheres_errors(
        big_l,
        laplacecross.all_cross_interactions_n_spheres_v2d(
            n, big_l, 2 * big_l + 5, RADII, CENTERS
        ),
        a_0_self,
        a_n,
        x_dia,
        x_dia_inv,
        mass_n_two,
        b,
        one,
    )


@pytest.fixture(scope="module")
def helmholtz_three_spheres() -> dict:
    n = 3
    big_l = 10
    sigmas = np.asarray([1.0, 1.25, 1.0, 1.0])
    kii = np.asarray([2.0, 2.5, 2.0, 2.0])
    pii = sigmas[1:] / sigmas[0]
    p_z = 5.0
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, RADII, pii, azimuthal=False
    )
    mass_n_two = mass.n_two_j_blocks(big_l, RADII, azimuthal=False)
    b = righthands.b_vector_n_spheres_mtf_plane_wave(
        n, big_l, CENTERS, p_z, kii[0], 1.0, RADII, x_dia, mass_n_two
    )
    eles = np.arange(0, big_l + 1)
    j_l = special.spherical_jn(eles, RADII[:, np.newaxis] * kii[0])
    j_lp = special.spherical_jn(
        eles, RADII[:, np.newaxis] * kii[0], derivative=True
    )
    r = RADII[0]
    b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        big_l, r, p_z, kii[0], 1.0, azimuthal=True
    )
    b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
        big_l, r, p_z, kii[0], 1.0, azimuthal=True
    )
    one = np.linalg.solve(
        mtf.mtf_1_matrix(
            r,
            pii[0],
            helmholtzself.a_0j_matrix(big_l, r, kii[0], azimuthal=True),
            helmholtzself.a_j_matrix(big_l, r, kii[1], azimuthal=True),
        ),
        righthands.b_vector_1_sphere_mtf(r, 1.0 / pii[0], b_d, b_n),
    )
    a_0_self, a_n = helmholtzself.a_0_a_n_sparse_matrices(
        n, big_l, RADII, kii, azimuthal=False
    )
    return n_spheres_errors(
        big_l,
        helmholtzcross.all_cross_interactions_n_spheres_from_v_2d(
            n, big_l, 2 * big_l + 5, kii[0], RADII, CENTERS, j_l, j_lp
        ),
        a_0_self,
        a_n,
        x_dia,
        x_dia_inv,
        mass_n_two,
        b,
        one,
    )


@pytest.mark.parametrize(
    "name, tol",
    [
        ("exterior_trace", 2e-17),
        ("interior_vs_phi_e", 2e-17),
        ("calderon_0", 2e-17),
        ("calderon_1", 5e-17),
        ("jump_dirichlet", 2e-17),
        ("jump_neumann", 1e-17),
    ],
)
def test_phantom_laplace_one_sphere(
    laplace_one_sphere: dict, name: str, tol: float
) -> None:
    assert laplace_one_sphere[name] < tol
    pass


@pytest.mark.parametrize(
    "name, tol",
    [
        ("exterior_trace", 5e-13),
        ("interior_vs_phi_e", 2e-13),
        ("calderon_0", 5e-13),
        ("calderon_1", 5e-13),
        ("jump_dirichlet", 2e-13),
        ("jump_neumann", 5e-13),
    ],
)
def test_phantom_helmholtz_one_sphere(
    helmholtz_one_sphere: dict, name: str, tol: float
) -> None:
    assert helmholtz_one_sphere[name] < tol
    pass


@pytest.mark.parametrize(
    "name, tol",
    [
        ("calderon_0", 5e-15),
        ("calderon_n", 5e-16),
        ("jump", 5e-15),
        ("analytic_first_sphere", 2e-13),
    ],
)
def test_phantom_laplace_three_spheres(
    laplace_three_spheres: dict, name: str, tol: float
) -> None:
    assert laplace_three_spheres[name] < tol
    pass


@pytest.mark.parametrize(
    "name, tol",
    [
        ("calderon_0", 2e-12),
        ("calderon_n", 2e-12),
        ("jump", 2e-12),
        ("analytic_first_sphere", 1e-13),
    ],
)
def test_phantom_helmholtz_three_spheres(
    helmholtz_three_spheres: dict, name: str, tol: float
) -> None:
    assert helmholtz_three_spheres[name] < tol
    pass
