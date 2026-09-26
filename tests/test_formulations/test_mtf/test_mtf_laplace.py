import numpy as np
import pytest
import scipy.sparse.linalg
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.crossinteractions as cross
import biosspheres.laplace.selfinteractions as selfin
import biosspheres.miscella.extensions as extensions

BIG_L = 5
R = 2.1
PI = 3.0


def gmres(operator, b: np.ndarray) -> np.ndarray:
    solution, info = scipy.sparse.linalg.gmres(
        operator, b, rtol=1e-13, restart=len(b)
    )
    assert info == 0
    return solution


def to_general(x: np.ndarray, blocks: int) -> np.ndarray:
    num = BIG_L + 1
    return np.concatenate(
        [
            extensions.azimuthal_trace_to_general_with_zeros(
                BIG_L, x[i * num : (i + 1) * num]
            )
            for i in range(blocks)
        ]
    )


def mtf_1_operator(azimuthal: bool):
    return mtf.mtf_1_linear_operator(
        selfin.a_0j_linear_operator(BIG_L, R, azimuthal),
        selfin.a_j_linear_operator(BIG_L, R, azimuthal),
        mtf.x_j_diagonal(BIG_L, R, PI, azimuthal),
        mtf.x_j_diagonal_inv(BIG_L, R, PI, azimuthal),
    )


@pytest.mark.parametrize("azimuthal, tol", [(True, 1e-13), (False, 1e-13)])
def test_mtf_1_linear_operator_vs_matrix(azimuthal: bool, tol: float) -> None:
    num = BIG_L + 1 if azimuthal else (BIG_L + 1) ** 2
    b = np.random.default_rng(0).random(4 * num)
    matrix = mtf.mtf_1_matrix(
        R,
        PI,
        selfin.a_0j_matrix(BIG_L, R, azimuthal),
        selfin.a_j_matrix(BIG_L, R, azimuthal),
    )
    error = np.linalg.norm(
        np.linalg.solve(matrix, b) - gmres(mtf_1_operator(azimuthal), b)
    )
    assert error < tol
    pass


def test_mtf_1_azimuthal_vs_general() -> None:
    b = np.random.default_rng(0).random(4 * (BIG_L + 1))
    solution = gmres(mtf_1_operator(False), to_general(b, 4))
    solution_azimuthal = to_general(gmres(mtf_1_operator(True), b), 4)
    assert np.linalg.norm(solution - solution_azimuthal) < 5e-14
    pass


@pytest.mark.parametrize("azimuthal, tol", [(True, 2e-14), (False, 5e-14)])
def test_mtf_1_reduced_linear_operator_vs_matrix(
    azimuthal: bool, tol: float
) -> None:
    num = BIG_L + 1 if azimuthal else (BIG_L + 1) ** 2
    b = np.random.default_rng(0).random(2 * num)
    operator = mtf.mtf_1_reduced_linear_operator(BIG_L, R, PI, azimuthal)
    matrix = mtf.mtf_1_reduced_matrix_laplace(
        PI, selfin.a_0j_matrix(BIG_L, R, azimuthal)
    )
    error = np.linalg.norm(np.linalg.solve(matrix, b) - gmres(operator, b))
    assert error < tol
    pass


def test_mtf_1_reduced_azimuthal_vs_general() -> None:
    b = np.random.default_rng(0).random(2 * (BIG_L + 1))
    solution = gmres(
        mtf.mtf_1_reduced_linear_operator(BIG_L, R, PI, False),
        to_general(b, 2),
    )
    solution_azimuthal = to_general(
        gmres(mtf.mtf_1_reduced_linear_operator(BIG_L, R, PI, True), b), 2
    )
    assert np.linalg.norm(solution - solution_azimuthal) < 2e-15
    pass


@pytest.mark.parametrize("azimuthal, tol", [(True, 1e-13), (False, 1e-13)])
def test_mtf_1_reduced_vs_not(azimuthal: bool, tol: float) -> None:
    num = BIG_L + 1 if azimuthal else (BIG_L + 1) ** 2
    b = np.random.default_rng(0).random(4 * num)
    solution = gmres(mtf_1_operator(azimuthal), b)
    a_1 = selfin.a_j_linear_operator(BIG_L, R, azimuthal)
    x_j = mtf.x_j_diagonal(BIG_L, R, PI, azimuthal)
    sol_red_1 = gmres(
        mtf.mtf_1_reduced_linear_operator(BIG_L, R, PI, azimuthal),
        b[0 : 2 * num] + 2.0 * a_1.matvec(b[2 * num : 4 * num]) / x_j,
    )
    sol_red_2 = 2.0 * a_1.matvec(
        (b[2 * num : 4 * num] + x_j * sol_red_1) / R**4
    )
    error = np.linalg.norm(np.concatenate((sol_red_1, sol_red_2)) - solution)
    assert error < tol
    pass


def n_spheres_system() -> tuple:
    n = 3
    big_l = 5
    radii = np.ones(n) * 1.112
    center_positions = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([-7.0, -3.0, -2.0]),
        np.asarray([3.0, 5.0, 7.0]),
    ]
    sigmas = np.asarray([3.0, 0.4, 1.7, 0.75])
    pii = sigmas[1:] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    big_a_0_cross = cross.all_cross_interactions_n_spheres_v2d(
        n, big_l, 10, radii, center_positions
    )
    a_0_self, a_n = selfin.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False
    )
    b = righthands.b_vector_n_spheres_mtf_point_source(
        n,
        big_l,
        center_positions,
        np.asarray([5.0, -5.0, 5.0]),
        radii,
        sigmas[0],
        x_dia,
        mass_n_two,
    )
    reduced = selfin.reduced_a_sparse_matrix(
        n, big_l, radii, pii, azimuthal=False
    )
    return (
        big_a_0_cross,
        a_0_self,
        a_n,
        x_dia,
        x_dia_inv,
        mass_n_two,
        b,
        reduced,
    )


def test_mtf_n_matrix_vs_linear_operator() -> None:
    big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv, _, b, _ = n_spheres_system()
    matrix = mtf.mtf_n_matrix(big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv)
    operator = mtf.mtf_n_linear_operator_v1(
        big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv
    )
    error = np.linalg.norm(np.linalg.solve(matrix, b) - gmres(operator, b))
    assert error < 2e-13
    pass


def test_mtf_n_reduced_vs_not() -> None:
    big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv, mass_n_two, b, reduced = (
        n_spheres_system()
    )
    half = len(b) // 2
    matrix = mtf.mtf_n_matrix(big_a_0_cross, a_0_self, a_n, x_dia, x_dia_inv)
    solution = np.linalg.solve(matrix, b)
    sol_red_1 = np.linalg.solve(
        mtf.mtf_n_reduced_matrix(big_a_0_cross, reduced),
        b[0:half] + 2.0 * a_n.dot(b[half:]) / x_dia,
    )
    sol_red_2 = 2.0 * a_n.dot((b[half:] + x_dia * sol_red_1) / mass_n_two**2)
    error = np.linalg.norm(np.concatenate((sol_red_1, sol_red_2)) - solution)
    assert error < 2e-15
    pass
