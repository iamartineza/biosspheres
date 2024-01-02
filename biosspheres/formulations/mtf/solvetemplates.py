import numpy as np
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.laplace.crossinteractions as crossin
import biosspheres.miscella.harmonicex as harmonicex


def mtf_laplace_one_sphere_point_source_azimuthal_direct_solver(
        big_l: int,
        r: float,
        sigma_e: float,
        sigma_i: float,
        distance: float,
        intensity: float,
) -> np.ndarray:
    pi = sigma_i / sigma_e
    
    # Build of phi_e.
    b_d = (harmonicex.
            point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
                big_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
                big_l, r, distance, sigma_e, intensity))
    b = righthands.b_vector_1_sphere_mtf(r, 1. / pi, b_d, b_n)
    
    # Build of mtf matrix
    a_0 = laplaceself.a_0j_matrix(big_l, r, azimuthal=True)
    a_1 = laplaceself.a_j_matrix(big_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution = np.linalg.solve(matrix, b)
    
    return solution


def mtf_laplace_n_spheres_point_source_direct_solver(
        n: int,
        big_l: int,
        big_l_c: int,
        radii: np.ndarray,
        center_positions,
        sigmas: np.ndarray,
        p0: np.ndarray,
) -> np.ndarray:

    # Build of phi_e.
    pii = sigmas[1:len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False)
    b = righthands.b_vector_n_spheres_mtf_point_source(n, big_l,
                                                       center_positions, p0,
                                                       radii, sigmas[0], x_dia,
                                                       mass_n_two)
    # Build of mtf matrix
    big_a_0_cross = crossin.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions)
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False)
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv)

    solution = np.linalg.solve(matrix, b)

    return solution
