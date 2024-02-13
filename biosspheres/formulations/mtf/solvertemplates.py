import numpy as np
import scipy.sparse.linalg
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.laplace.crossinteractions as laplacecross
import biosspheres.miscella.harmonicex as harmonicex


def mtf_laplace_one_sphere_point_source_azimuthal_direct_solver(
    big_l: int,
    r: float,
    sigma_e: float,
    sigma_i: float,
    distance: float,
    intensity: float,
) -> np.ndarray:
    """
    Returns the solution of the system
    M x = b,
    where M is the matrix of the MTF system for one sphere, and b is the
    right hand corresponding to an external excitation of the function
    biossphere.miscella.mathfunctions.point_source(
        r, [0, 0, distance], sigma_e).

    Notes
    -----
    Write details about what it is being solved.


    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    sigma_e : float
        > 0, parameter of the exterior medium.
    sigma_i : float
        > 0, parameter of the interior medium.
    distance : float
        > 0, distance of the center of the sphere to the point source.
    intensity : float
        > 0, intensity of the point source.

    Returns
    -------
    solution : np.ndarray
        of floats. Length 4 * (big_l + 1).

    """
    pi = sigma_i / sigma_e

    # Build of phi_e.
    b_d = harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        big_l, r, distance, sigma_e, intensity
    )
    b_n = harmonicex.point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        big_l, r, distance, sigma_e, intensity
    )
    b = righthands.b_vector_1_sphere_mtf(r, 1.0 / pi, b_d, b_n)

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
    center_positions: list[np.ndarray],
    sigmas: np.ndarray,
    p0: np.ndarray,
) -> np.ndarray:
    """

    Parameters
    ----------
    n : int
        > 0, number of spheres.
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.
    radii : np.ndarray
        Array with the radii of the spheres.
    center_positions : list[np.ndarray]
        List or arrays with the center position of the spheres
    sigmas : np.ndarray
        Array with the parameters of each medium.
    p0 : np.ndarray
        Length 3. Indicates the position of the point source.

    Returns
    -------
    solution : np.ndarray
        of floats. Length 4 * n * (big_l + 1)**2.

    """
    # Build of the right hand side.
    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    b = righthands.b_vector_n_spheres_mtf_point_source(
        n, big_l, center_positions, p0, radii, sigmas[0], x_dia, mass_n_two
    )
    # Build of mtf matrix
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions
    )
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False
    )
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    del big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, mass_n_two, x_dia
    del x_dia_inv, pii

    solution = np.linalg.solve(matrix, b)

    return solution


def mtf_laplace_n_spheres_point_source_indirect_solver(
    n: int,
    big_l: int,
    big_l_c: int,
    radii: np.ndarray,
    center_positions: list[np.ndarray],
    sigmas: np.ndarray,
    p0: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """

    Parameters
    ----------
    n : int
        > 0, number of spheres.
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.
    radii : np.ndarray
        Array with the radii of the spheres.
    center_positions : list[np.ndarray]
        List or arrays with the center position of the spheres
    sigmas : np.ndarray
        Array with the parameters of each medium.
    p0 : np.ndarray
        Length 3. Indicates the position of the point source.
    tolerance : float
        Tolerance for scipy.sparse.linalg.gmres routine.

    Returns
    -------
    solution : np.ndarray
        of floats. Length 4 * n * (big_l + 1)**2.

    """
    # Build of the right hand side.
    pii = sigmas[1 : len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    b = righthands.b_vector_n_spheres_mtf_point_source(
        n, big_l, center_positions, p0, radii, sigmas[0], x_dia, mass_n_two
    )
    # Build of mtf matrix
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions
    )
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False
    )

    linear_operator = mtf.mtf_n_linear_operator_v1(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    del big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, mass_n_two, x_dia
    del x_dia_inv, pii

    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b, tol=tolerance, restart=(4 * (big_l + 1) ** 2) ** 3
    )

    return solution
