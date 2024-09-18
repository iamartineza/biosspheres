import numpy as np
import scipy.sparse.linalg
import scipy.special as special
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.selfinteractions as self
import biosspheres.helmholtz.crossinteractions as cross


def testing_mtf_linear_operators_and_matrices_n_spheres() -> None:
    n = 3
    big_l = 10
    big_l_c = 2 * big_l + 5
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
    kii = np.ones(n + 1) * 0.75
    kii[0] = 2.0
    kii[1] = 0.4
    kii[2] = 1.7

    p_z = 5.0
    a = 1.0
    tolerance = 10 ** (-9)
    num = (big_l + 1) ** 2

    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )

    j_l = np.empty((n, big_l + 1))
    j_lp = np.empty((n, big_l + 1))

    for i in np.arange(0, n):
        j_l[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0]
        )
        j_lp[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0], derivative=True
        )
        pass

    big_a_0_cross = cross.all_cross_interactions_n_spheres_from_v_2d(
        n,
        big_l,
        big_l_c,
        kii[0],
        radii,
        center_positions,
        j_l,
        j_lp,
    )
    sparse_big_a_0_self, sparse_big_a_n = self.a_0_a_n_sparse_matrices(
        n, big_l, radii, kii, azimuthal=False
    )

    b = righthands.b_vector_n_spheres_mtf_plane_wave(
        n, big_l, center_positions, p_z, kii[0], a, radii, x_dia, mass_n_two
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
    print("Difference between solutions.")
    print(np.linalg.norm(solution_direct - solution_indirect))
    pass


def testing_mtf_reduced_vs_not_n_spheres() -> None:
    n = 3
    big_l = 0
    big_l_c = 2 * big_l + 5
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
    kii = np.ones(n + 1) * 0.75
    kii[0] = 2.0
    kii[1] = 0.4
    kii[2] = 1.7

    p_z = 5.0
    a = 1.0
    tolerance = 10 ** (-9)

    num = (big_l + 1) ** 2

    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )

    j_l = np.empty((n, big_l + 1))
    j_lp = np.empty((n, big_l + 1))

    for i in np.arange(0, n):
        j_l[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0]
        )
        j_lp[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0], derivative=True
        )
        pass

    big_a_0_cross = cross.all_cross_interactions_n_spheres_from_v_2d(
        n,
        big_l,
        big_l_c,
        kii[0],
        radii,
        center_positions,
        j_l,
        j_lp,
    )
    sparse_big_a_0_self, sparse_big_a_n = self.a_0_a_n_sparse_matrices(
        n, big_l, radii, kii, azimuthal=False
    )

    b = righthands.b_vector_n_spheres_mtf_plane_wave(
        n, big_l, center_positions, p_z, kii[0], a, radii, x_dia, mass_n_two
    )

    # Not reduced system
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    solution_not_reduced = np.linalg.solve(matrix, b)

    # Reduced system
    sparse_reduced = self.reduced_a_sparse_matrix(
        n, big_l, radii, kii, pii, azimuthal=False
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
    testing_mtf_linear_operators_and_matrices_n_spheres()
    testing_mtf_reduced_vs_not_n_spheres()
