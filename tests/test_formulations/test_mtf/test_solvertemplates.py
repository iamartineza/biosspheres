import numpy as np
import pytest
import biosspheres.formulations.mtf.solvertemplates as solver
import biosspheres.laplace.selfinteractions as selfin
import biosspheres.miscella.extensions as extensions
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.spherearrangements as pos

BIG_L = 10
R = 1.3
DISTANCE = 5.0


@pytest.mark.parametrize(
    "sigma_e, sigma_i, tol", [(1.0, 1.0, 1e-14), (5.0, 0.455, 5e-14)]
)
def test_one_sphere_calderon_and_jumps(
    sigma_e: float, sigma_i: float, tol: float
) -> None:
    num = BIG_L + 1
    solution = (
        solver.mtf_laplace_one_sphere_point_source_azimuthal_direct_solver(
            BIG_L, R, sigma_e, sigma_i, DISTANCE, 1.0
        )
    )
    b_d = harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        BIG_L, R, DISTANCE, sigma_e, 1.0
    )
    b_n = harmonicex.point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        BIG_L, R, DISTANCE, sigma_e, 1.0
    )
    exterior = solution[0 : 2 * num]
    interior = solution[2 * num : 4 * num]
    a_0 = selfin.a_0j_matrix(BIG_L, R, azimuthal=True)
    a_1 = selfin.a_j_matrix(BIG_L, R, azimuthal=True)
    scale = np.linalg.norm(np.concatenate((b_d, b_n)))
    errors = [
        np.linalg.norm(2 * a_0 @ exterior - R**2 * exterior),
        np.linalg.norm(2 * a_1 @ interior - R**2 * interior),
        np.linalg.norm(solution[0:num] - solution[2 * num : 3 * num] + b_d),
        np.linalg.norm(
            sigma_e * (solution[num : 2 * num] + b_n)
            + sigma_i * solution[3 * num : 4 * num]
        ),
    ]
    assert np.max(errors) / scale < tol
    pass


@pytest.mark.parametrize(
    "sigma_e, sigma_i, tol", [(1.0, 1.0, 5e-14), (5.0, 0.455, 1e-14)]
)
def test_one_sphere_vs_n_spheres_direct_solver(
    sigma_e: float, sigma_i: float, tol: float
) -> None:
    num = BIG_L + 1
    solution = (
        solver.mtf_laplace_one_sphere_point_source_azimuthal_direct_solver(
            BIG_L, R, sigma_e, sigma_i, DISTANCE, 1.0
        )
    )
    general = np.concatenate(
        [
            extensions.azimuthal_trace_to_general_with_zeros(
                BIG_L, solution[i * num : (i + 1) * num]
            )
            for i in range(4)
        ]
    )
    solution_n = solver.mtf_laplace_n_spheres_point_source_direct_solver(
        1,
        BIG_L,
        2 * BIG_L + 5,
        np.asarray([R]),
        [np.zeros(3)],
        np.asarray([sigma_e, sigma_i]),
        np.asarray([0.0, 0.0, DISTANCE]),
    )
    error = np.linalg.norm(general - solution_n) / np.linalg.norm(general)
    assert error < tol
    pass


@pytest.mark.parametrize("sigma_e, sigma_i", [(0.0, 0.75), (1.75, 0.0)])
def test_one_sphere_invalid_sigma(sigma_e: float, sigma_i: float) -> None:
    with pytest.raises(ValueError):
        solver.mtf_laplace_one_sphere_point_source_azimuthal_direct_solver(
            BIG_L, R, sigma_e, sigma_i, DISTANCE, 1.0
        )
    pass


@pytest.mark.parametrize(
    "big_l, big_l_c, tol",
    [
        (5, 25, 2e-9),
        pytest.param(15, 55, 2e-9, marks=pytest.mark.slow),
    ],
)
def test_n_spheres_direct_vs_indirect(
    big_l: int, big_l_c: int, tol: float
) -> None:
    n = 8
    r = 0.875
    radii = np.ones(n) * r
    center_positions = pos.cube_vertex_positions(2, r, 1.15)
    sigmas = np.ones(n + 1) * 0.75
    sigmas[0] = 1.75
    p0 = np.ones(3) * -5.0
    direct = solver.mtf_laplace_n_spheres_point_source_direct_solver(
        n, big_l, big_l_c, radii, center_positions, sigmas, p0
    )
    indirect = solver.mtf_laplace_n_spheres_point_source_indirect_solver(
        n, big_l, big_l_c, radii, center_positions, sigmas, p0, 1e-10
    )
    error = np.linalg.norm(direct - indirect) / np.linalg.norm(direct)
    assert error < tol
    pass
