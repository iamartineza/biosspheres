import numpy as np
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands.mtf as righthands
import biosspheres.laplace.selfinteractions as laplace
import biosspheres.miscella.harmonicex as harmonicex


def mtf_laplace_solve_one_sphere_point_source_azimuthal(
        max_l: int,
        r: float,
        sigma_e: float,
        sigma_i: float,
        distance: float,
        intensity: float,
) -> np.ndarray:
    pi = sigma_e / sigma_i
    
    # Build of phi_e.
    b_d = (harmonicex.
            point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
                max_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
                max_l, r, distance, sigma_e, intensity))
    b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
    
    # Build of mtf matrix
    a_0 = laplace.a_0j_matrix(max_l, r, azimuthal=True)
    a_1 = laplace.a_j_matrix(max_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution = np.linalg.solve(matrix, b)
    
    return solution
