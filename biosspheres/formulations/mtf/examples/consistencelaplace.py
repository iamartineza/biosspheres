import numpy as np
import scipy.sparse.linalg
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.laplace.crossinteractions as laplacecross
import biosspheres.miscella.extensions as extensions


def testing_mtf_linear_operators_and_matrices_one_sphere(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace operators.
    Testing that the MTF linear operators and the MTF matrix do the same
    matrix for one sphere and azimuthal symmetry

    Parameters
    ----------
    big_l : int
        >= 0, max degree. Default 5
    r : float
        > 0, radius. Default 2.5
    pi : float
        > 0, adimensional parameter. Default 3.
    
    Returns
    -------
    None

    """
    num = (big_l + 1)
    b = np.random.random((4 * num))
    
    a_0 = laplaceself.a_0j_linear_operator(big_l, r)
    a_1 = laplaceself.a_j_linear_operator(big_l, r)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    a_0 = laplaceself.a_0j_matrix(big_l, r)
    a_1 = laplaceself.a_j_matrix(big_l, r, True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                               tol=10**(-13),
                                               restart=(4 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    print(
        '\nRunning function testing_mtf_linear_operators_and_matrices_laplace')
    print('- MTF azimuthal')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    num = (big_l + 1)**2
    b = np.random.random((4 * num))
    
    a_0 = laplaceself.a_0j_linear_operator(big_l, r, False)
    a_1 = laplaceself.a_j_linear_operator(big_l, r, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    a_0 = laplaceself.a_0j_matrix(big_l, r, False)
    a_1 = laplaceself.a_j_matrix(big_l, r, False)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    
    norms = []
    
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                               tol=10**(-13),
                                               restart=(4 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    print('- MTF')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_linear_operators_and_matrices_n_spheres(
) -> None:
    n = 3
    big_l = 5
    big_l_c = 10
    radii = np.ones(n) * 1.112
    center_positions = [np.asarray([0., 0., 0]), np.asarray([-7., -3., -2.]),
                        np.asarray([3., 5., 7.])]
    sigmas = np.ones(n+1) * 0.75
    sigmas[0] = 3.
    sigmas[1] = 0.4
    sigmas[2] = 1.7
    p0 = np.asarray([5., -5., 5.])
    tolerance = 10**(-13)
    num = (big_l + 1)**2
    
    pii = sigmas[1:len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False)
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions)
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False)
    
    b = righthands.b_vector_n_spheres_mtf_point_source(n, big_l,
                                                       center_positions, p0,
                                                       radii, sigmas[0], x_dia,
                                                       mass_n_two)
    # Direct solver
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv)
    
    solution_direct = np.linalg.solve(matrix, b)
    del matrix
    
    # Iterative solver
    linear_operator = mtf.mtf_n_linear_operator_v1(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    solution_indirect, info = scipy.sparse.linalg.gmres(
        linear_operator, b,
        tol=tolerance,
        restart=(4 * num)**3,
        callback=callback_function,
        callback_type='pr_norm')
    
    print('\nRunning '
          'testing_mtf_linear_operators_and_matrices_laplace_n_spheres')
    print('Difference between solutions.')
    print(np.linalg.norm(solution_direct - solution_indirect))
    pass


def testing_mtf_azimuthal_and_no_azimuthal(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace operators.
    
    Notes
    -----
    For a right hand with azimuthal symmetry when extended by 0 to explicitly
    compute the result without considering the symmetry, the result should be
    the same with the azimuthal and the not azimuthal versions.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.

    Returns
    -------
    None

    """
    num = (big_l + 1)
    b = np.random.random((4 * num))
    b_2_1 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[0:num])
    b_2_2 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[num:2 * num])
    b_2_3 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[2 * num:3 * num])
    b_2_4 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[3 * num:4 * num])
    b2_12 = np.concatenate((b_2_1, b_2_2))
    b2_34 = np.concatenate((b_2_3, b_2_4))
    b2 = np.concatenate((b2_12, b2_34))
    
    a_0 = laplaceself.a_0j_linear_operator(big_l, r, False)
    a_1 = laplaceself.a_j_linear_operator(big_l, r, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    num = (big_l + 1)**2
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b2,
                                               tol=10**(-13),
                                               restart=(4 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    num = (big_l + 1)
    
    a_0 = laplaceself.a_0j_linear_operator(big_l, r)
    a_1 = laplaceself.a_j_linear_operator(big_l, r)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    norms = []
    
    solution2, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                                tol=10**(-13),
                                                restart=(4 * num)**3,
                                                callback=callback_function,
                                                callback_type='pr_norm')
    solution2_1 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[0:num])
    solution2_2 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[num:2 * num])
    solution2_3 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[2 * num:3 * num])
    solution2_4 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[3 * num:4 * num])
    solution2_12 = np.concatenate((solution2_1, solution2_2))
    solution2_34 = np.concatenate((solution2_3, solution2_4))
    solution2 = np.concatenate((solution2_12, solution2_34))
    print('\nRunning function testing_mtf_azimuthal_and_no_azimuthal')
    print('- The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_reduced_linear_operators_and_matrices(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Testing that the formulations reduced linear operators and the A matrix for one
    sphere represent the same

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.
        
    Returns
    -------
    None

    """
    num = (big_l + 1)
    b = np.random.random((2 * num))
    
    linear_operator = (
        mtf.mtf_1_reduced_linear_operator(
            big_l, r, pi))
    a_0 = laplaceself.a_0j_matrix(big_l, r, True)
    matrix = mtf.mtf_1_reduced_matrix_laplace(pi, a_0)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                               tol=10**(-13),
                                               restart=(2 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    print(
        '\nRunning function testing_mtf_reduced_linear_operators_and_matrices')
    print('- MTF reduced azimuthal')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    num = (big_l + 1)**2
    b = np.random.random((2 * num))
    
    linear_operator = (
        mtf.mtf_1_reduced_linear_operator(big_l, r, pi, False))
    a_0 = laplaceself.a_0j_matrix(big_l, r, False)
    matrix = mtf.mtf_1_reduced_matrix_laplace(pi, a_0)
    
    norms = []
    
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                               tol=10**(-13),
                                               restart=(2 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    print('- MTF reduced')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_reduced_azimuthal_and_no_azimuthal(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace.

    Notes
    -----
    For a right hand with azimuthal symmetry when extended by 0 to explicitly
    compute the result without considering the symmetry, the result should be
    the same with the azimuthal and the not azimuthal versions.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.
        
    Returns
    -------
    None

    """
    num = (big_l + 1)
    b = np.random.random((2 * num))
    b_2_1 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[0:num])
    b_2_2 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[num:2 * num])
    b2 = np.concatenate((b_2_1, b_2_2))
    
    linear_operator = mtf.mtf_1_reduced_linear_operator(
        big_l, r, pi, False)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    num = (big_l + 1)**2
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b2,
                                               tol=10**(-13),
                                               restart=(2 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    num = (big_l + 1)
    
    linear_operator = (
        mtf.mtf_1_reduced_linear_operator(
            big_l, r, pi))
    
    norms = []
    
    solution2, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                                tol=10**(-13),
                                                restart=(2 * num)**3,
                                                callback=callback_function,
                                                callback_type='pr_norm')
    solution2_1 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[0:num])
    solution2_2 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[num:2 * num])
    solution2 = np.concatenate((solution2_1, solution2_2))
    print('\nRunning function testing_mtf_reduced_azimuthal_and_no_azimuthal')
    print('- The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_reduced_vs_not_one_sphere(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace

    Notes
    -----
    The formulations reduced system came from making the steps to obtain the Schur's
    complement, thus, having the correct right hand sides the result needs to
    be the same with or without taking the reduced system.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.
    
    Returns
    -------
    None

    """
    num = (big_l + 1)
    b = np.random.random((4 * num))
    
    linear_operator_red = (
        mtf.mtf_1_reduced_linear_operator(
            big_l, r, pi))
    a_0 = laplaceself.a_0j_linear_operator(big_l, r)
    a_1 = laplaceself.a_j_linear_operator(big_l, r)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    num = (big_l + 1)
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                               tol=10**(-13),
                                               restart=(4 * num)**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    
    norms = []
    
    a_1_lin_op = laplaceself.a_j_linear_operator(big_l, r)
    b_red_1 = b[0:2 * num] + 2. * (a_1_lin_op.matvec(b[2 * num:4 * num])
                                   / mtf.x_j_diagonal(big_l, r, pi, True))
    sol_red_1, info = scipy.sparse.linalg.gmres(
        linear_operator_red, b_red_1,
        tol=10**(-13),
        restart=(2 * num)**3,
        callback=callback_function,
        callback_type='pr_norm')
    sol_red_2 = (
            b[2 * num:4 * num] +
            mtf.x_j_diagonal(big_l, r, pi, True) * sol_red_1)
    sol_red_2[:] = sol_red_2[:] / (r**4)
    sol_red_2[:] = 2. * a_1_lin_op.matvec(sol_red_2[:])
    solution_red_1 = np.concatenate((sol_red_1, sol_red_2))
    print('\nRunning function testing_mtf_reduced_vs_not_one_sphere')
    print('- The following should be zero or near.')
    print(np.linalg.norm(solution_red_1 - solution))
    pass


def testing_mtf_reduced_vs_not_n_spheres() -> None:
    n = 3
    big_l = 5
    big_l_c = 10
    radii = np.ones(n) * 1.112
    center_positions = [np.asarray([0., 0., 0]), np.asarray([-7., -3., -2.]),
                        np.asarray([3., 5., 7.])]
    sigmas = np.ones(n + 1) * 0.75
    sigmas[0] = 3.
    sigmas[1] = 0.4
    sigmas[2] = 1.7
    p0 = np.asarray([5., -5., 5.])
    
    num = (big_l + 1)**2
    
    pii = sigmas[1:len(sigmas)] / sigmas[0]
    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)
    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False)
    big_a_0_cross = laplacecross.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, radii, center_positions)
    sparse_big_a_0_self, sparse_big_a_n = laplaceself.a_0_a_n_sparse_matrices(
        n, big_l, radii, azimuthal=False)
    
    b = righthands.b_vector_n_spheres_mtf_point_source(n, big_l,
                                                       center_positions, p0,
                                                       radii, sigmas[0], x_dia,
                                                       mass_n_two)
    # Not reduced system
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv)
    
    solution_not_reduced = np.linalg.solve(matrix, b)
    
    # Reduced system
    sparse_reduced = laplaceself.reduced_a_sparse_matrix(
        n, big_l, radii, pii, azimuthal=False)
    matrix = mtf.mtf_n_reduced_matrix(big_a_0_cross, sparse_reduced)
    b_red_1 = b[0:2*n*num] + 2. * sparse_big_a_n.dot(b[2 * n*num:4 * n*num]) / x_dia
    sol_red_1 = np.linalg.solve(matrix, b_red_1)
    sol_red_2 = (b[2 * n*num:4 * n*num] + x_dia * sol_red_1) / mass_n_two**2
    sol_red_2[:] = 2. * sparse_big_a_n.dot(sol_red_2[:])
    solution_reduced = np.concatenate((sol_red_1, sol_red_2))
    print('\nRunning function testing_mtf_reduced_vs_not_n_spheres')
    print('- The following should be zero or near.')
    print(np.linalg.norm(solution_not_reduced - solution_reduced))
    pass


if __name__ == '__main__':
    testing_mtf_reduced_vs_not_n_spheres()
    testing_mtf_linear_operators_and_matrices_one_sphere()
    testing_mtf_linear_operators_and_matrices_n_spheres()
    testing_mtf_azimuthal_and_no_azimuthal()
    testing_mtf_reduced_linear_operators_and_matrices()
    testing_mtf_reduced_azimuthal_and_no_azimuthal()
    testing_mtf_reduced_vs_not_one_sphere()
