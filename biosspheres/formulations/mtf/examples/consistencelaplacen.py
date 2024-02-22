import numpy as np
import scipy.sparse.linalg
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.laplace.crossinteractions as laplacecross


def testing_mtf_linear_operators_and_matrices_n_spheres() -> None:
    n = 3
    big_l = 5
    big_l_c = 10
    radii = np.ones(n) * 1.112
    center_positions = [
        np.asarray([0.0, 0.0, 0]),
        np.asarray([-7.0, -3.0, -2.0]),
        np.asarray([3.0, 5.0, 7.0]),
    ]
    sigmas = np.ones(n + 1) * 0.75
    sigmas[0] = 3.0
    sigmas[1] = 0.4
    sigmas[2] = 1.7
    p0 = np.asarray([5.0, -5.0, 5.0])
    tolerance = 10 ** (-13)
    num = (big_l + 1) ** 2

    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions
    )
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False
    )

    b = righthands.b_vector_n_spheres_mtf_point_source(
        n, big_l, center_positions, p0, radii, sigmas[0], x_dia, mass_n_two
    )
    # Direct solver
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    solution_direct = np.linalg.solve(matrix, b)
    del matrix

    # Iterative solver
    linear_operator = mtf.mtf_n_linear_operator_v1(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    solution_indirect, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tolerance,
        restart=(4 * num) ** 3,
        callback=callback_function,
        callback_type="pr_norm",
    )

    print(
        "\nRunning "
        "testing_mtf_linear_operators_and_matrices_laplace_n_spheres"
    )
    print("Difference between solutions.")
    print(np.linalg.norm(solution_direct - solution_indirect))
    pass


def testing_mtf_reduced_vs_not_n_spheres() -> None:
    n = 3
    big_l = 5
    big_l_c = 10
    radii = np.ones(n) * 1.112
    center_positions = [
        np.asarray([0.0, 0.0, 0]),
        np.asarray([-7.0, -3.0, -2.0]),
        np.asarray([3.0, 5.0, 7.0]),
    ]
    sigmas = np.ones(n + 1) * 0.75
    sigmas[0] = 3.0
    sigmas[1] = 0.4
    sigmas[2] = 1.7
    p0 = np.asarray([5.0, -5.0, 5.0])

    num = (big_l + 1) ** 2

    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions
    )
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False
    )

    b = righthands.b_vector_n_spheres_mtf_point_source(
        n, big_l, center_positions, p0, radii, sigmas[0], x_dia, mass_n_two
    )
    # Not reduced system
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    solution_not_reduced = np.linalg.solve(matrix, b)

    # Reduced system
    sparse_reduced = laplaceself.reduced_a_sparse_matrix(
        n, big_l, radii, pii, azimuthal=False
    )
    matrix = mtf.mtf_n_reduced_matrix(big_a_0_cross, sparse_reduced)
    b_red_1 = (
        b[0 : 2 * n * num]
        + 2.0 * sparse_big_a_n.dot(b[2 * n * num : 4 * n * num]) / x_dia
    )
    sol_red_1 = np.linalg.solve(matrix, b_red_1)
    sol_red_2 = (
        b[2 * n * num : 4 * n * num] + x_dia * sol_red_1
    ) / mass_n_two**2
    sol_red_2[:] = 2.0 * sparse_big_a_n.dot(sol_red_2[:])
    solution_reduced = np.concatenate((sol_red_1, sol_red_2))
    print("\nRunning function testing_mtf_reduced_vs_not_n_spheres")
    print("- The following should be zero or near.")
    print(np.linalg.norm(solution_not_reduced - solution_reduced))
    pass


if __name__ == "__main__":
    testing_mtf_reduced_vs_not_n_spheres()
    testing_mtf_linear_operators_and_matrices_n_spheres()
