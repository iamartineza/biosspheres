import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplace
import biosspheres.laplace.drawing as draw
import biosspheres.miscella.auxindexes as auxindexes
import biosspheres.miscella.extensions as extensions
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.mathfunctions as mathfunctions
import biosspheres.quadratures.sphere as quadratures


def testing_mtf_linear_operators_and_matrices_laplace(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace operators.
    Testing that the MTF linear operators and the MTF matrix do the same matrix
    for one sphere and azimuthal symmetry.

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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r)
    a_1 = laplace.a_j_linear_operator(big_l, r)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    a_0 = laplace.a_0j_matrix(big_l, r)
    a_1 = laplace.a_j_matrix(big_l, r, True)
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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r, False)
    a_1 = laplace.a_j_linear_operator(big_l, r, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    a_0 = laplace.a_0j_matrix(big_l, r, False)
    a_1 = laplace.a_j_matrix(big_l, r, False)
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


def testing_mtf_linear_operators_and_matrices_laplace(
        big_l: int = 5, r: float = 2.1, pi: float = 3.
) -> None:
    """
    Test for Laplace operators.
    Testing that the MTF linear operators and the MTF matrix do the same matrix
    for one sphere and azimuthal symmetry

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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r)
    a_1 = laplace.a_j_linear_operator(big_l, r)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    
    a_0 = laplace.a_0j_matrix(big_l, r)
    a_1 = laplace.a_j_matrix(big_l, r, True)
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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r, False)
    a_1 = laplace.a_j_linear_operator(big_l, r, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)
    a_0 = laplace.a_0j_matrix(big_l, r, False)
    a_1 = laplace.a_j_matrix(big_l, r, False)
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


def testing_mtf_azimuthal_and_no_azimuthal_laplace(
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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r, False)
    a_1 = laplace.a_j_linear_operator(big_l, r, False)
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
    
    a_0 = laplace.a_0j_linear_operator(big_l, r)
    a_1 = laplace.a_j_linear_operator(big_l, r)
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


def testing_mtf_reduced_linear_operators_and_matrices_laplace(
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
    a_0 = laplace.a_0j_matrix(big_l, r, True)
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
    a_0 = laplace.a_0j_matrix(big_l, r, False)
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


def testing_mtf_reduced_azimuthal_and_no_azimuthal_laplace(
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


def testing_mtf_reduced_vs_not_laplace(
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
    a_0 = laplace.a_0j_linear_operator(big_l, r)
    a_1 = laplace.a_j_linear_operator(big_l, r)
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
    
    a_1_lin_op = laplace.a_j_linear_operator(big_l, r)
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
    print('\nRunning function testing_mtf_reduced_vs_not')
    print('- The following should be zero or near.')
    print(np.linalg.norm(solution_red_1 - solution))
    pass


def phantom_1_point_source_azimuthal(
        max_l: int = 50,
        r: float = 1.3,
        distance: float = 20.,
        intensity: float = 1.,
        resolution: int = 10,
        horizontal: float = 10.,
        vertical: float = 10.,
) -> None:
    print('\nPhantom spheres experiments for the MTF,')
    print('- One sphere, point source, azimuthal symmetry.')
    pi = 1.
    sigma_e = 1.
    num = max_l + 1
    
    # --- Build of phi_e.
    b_d = (harmonicex.
        point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
            max_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
        point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
            max_l, r, distance, sigma_e, intensity))
    b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
    
    a_0 = laplace.a_0j_matrix(max_l, r, azimuthal=True)
    a_1 = laplace.a_j_matrix(max_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution2 = np.linalg.solve(matrix, b)
    
    print('--- Checking of errors.')
    dirichlet_ex = solution2[0:num]
    neumann_ex = solution2[num:2 * num]
    dirichlet_in = solution2[2 * num:3 * num]
    neumann_in = solution2[3 * num:4 * num]
    exterior_u = solution2[0:2 * num]
    print('---- Norm of the exterior trace (should be near zero).')
    print(np.linalg.norm(exterior_u))
    print('---- Norm of the difference between $\\gamma^{01} \\phi_e^L$')
    print('and the interior trace (absolute error):')
    print(np.linalg.norm(
        np.concatenate((b_d, -b_n)) - solution2[2 * num:4 * num]))
    print('---- Discrete Calderon errors:')
    print(np.linalg.norm(2 * np.matmul(a_0, solution2[0:2 * num])
                         - r**2 * solution2[0:2 * num]))
    print(
        np.linalg.norm(2 * np.matmul(a_1, solution2[2 * num:4 * num])
                       - r**2 * solution2[2 * num:4 * num]))
    print('---- Jump errors.')
    print('----- Dirichlet trace:')
    jump_dirichlet = np.linalg.norm(dirichlet_ex - dirichlet_in + b_d)
    print(jump_dirichlet)
    print('----- Neumann trace:')
    jump_neumann = \
        np.linalg.norm((neumann_ex + b_n) + neumann_in)
    print(jump_neumann)
    print('----- Total jump error:')
    print(np.sqrt(jump_dirichlet**2 + jump_neumann**2))
    
    print('--- For plotting the convergence when the degree is increasing.')
    solutions = np.zeros((4 * num, max_l))
    errores = np.zeros((4 * num, max_l))
    for el in range(0, max_l):
        now_num = el + 1
        b_d = (harmonicex.
            point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
                el, r, distance, sigma_e, intensity))
        b_n = harmonicex. \
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
                el, r, distance, sigma_e, intensity)
        b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
        a_0 = laplace.a_0j_matrix(el, r, azimuthal=True)
        a_1 = laplace.a_j_matrix(el, r, azimuthal=True)
        matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
        solution = np.linalg.solve(matrix, b)
        solutions[0:now_num, el] = solution[0:now_num]
        solutions[num:num + now_num, el] = solution[now_num:2 * now_num]
        solutions[2 * num:2 * num + now_num, el] = solution[
                                                   2 * now_num:3 * now_num]
        solutions[3 * num:3 * num + now_num, el] = solution[
                                                   3 * now_num:4 * now_num]
        errores[:, el] = solutions[:, el] - solution2
    y1 = np.linalg.norm(errores[0:num], axis=0)
    y2 = np.linalg.norm(errores[num:2 * num], axis=0)
    y3 = np.linalg.norm(errores[2 * num:3 * num], axis=0)
    y4 = np.linalg.norm(errores[3 * num:4 * num], axis=0)
    plt.figure()
    plt.semilogy(
        y1, marker='p', linestyle='dashed',
        label='$||\\gamma_D^{01}u_0 - u_{D,01}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y2, marker='*', linestyle='dashed',
        label='$||\\gamma_N^{01}u_0 - u_{N,01}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y3, marker='x', linestyle='dashed',
        label='$||\\gamma_D^{1}u_1 - u_{D,1}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y4, marker='.', linestyle='dashed',
        label='$||\\gamma_N^{1}u_1 - u_{N,1}^{L}||_{L^2(\\Gamma_1)}$')
    plt.xlabel('$L$')
    plt.ylabel('Error')
    plt.legend()
    print('--- Pictures of the phantom sphere.')
    
    big_l_c = 100
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(max_l)
    quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics = \
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            max_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    eles = np.arange(0, max_l + 1)
    l_square_plus_l = (eles + 1) * eles
    
    vector = pre_vector * r
    
    surface_field_d0 = np.sum(
        solution2[0:num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                   l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_d0_max = np.max(surface_field_d0)
    surface_field_d0_min = np.min(surface_field_d0)
    surface_field_d0 = (surface_field_d0 - surface_field_d0_min) \
                       / (surface_field_d0_max - surface_field_d0_min)
    surface_field_n0 = np.sum(
        solution2[num:2 * num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                         l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_n0_max = np.max(surface_field_n0)
    surface_field_n0_min = np.min(surface_field_n0)
    surface_field_n0 = (surface_field_n0 - surface_field_n0_min) \
                       / (surface_field_n0_max - surface_field_n0_min)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_d0))
    ax_1.set_xlabel('$x \\ [\\mu m]$')
    ax_1.set_ylabel('$y \\ [\\mu m]$')
    ax_1.set_zlabel('$z \\ [\\mu m]$')
    ax_1.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax_1,
        label='[V]'
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_n0))
    ax.set_xlabel('$x \\ [\\mu m]$')
    ax.set_ylabel('$y \\ [\\mu m]$')
    ax.set_zlabel('$z \\ [\\mu m]$')
    ax.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax,
        label='[V $/ \\mu m$ ]'
    )
    
    def point_source(x: np.ndarray) -> float:
        return 0.
    
    center = np.asarray([0., 0., 0.])
    inter_horizontal = resolution
    inter_vertical = resolution
    
    p = np.array([0., 0., distance])
    
    r = 1.
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    temp = np.zeros_like(solution2[2 * (max_l + 1):4 * (max_l + 1)])
    temp[:] = solution2[2 * (max_l + 1):4 * (max_l + 1)]
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = 0.
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.colorbar(label='[V]')
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    
    def point_source(x: np.ndarray) -> float:
        return mathfunctions.point_source(x, p, sigma_e)
    
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = temp[:]
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = 0.
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.colorbar(label='[V]')
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    
    pass


def non_phantom_1_point_source_z_alignment_distance_convergence(
        max_l: int = 50,
        r: float = 1.3,
        sigma_e: float = 5.,
        sigma_i: float = 0.455,
        distance: float = 20.,
        intensity: float = 1.,
        resolution: int = 10,
        horizontal: float = 10.,
        vertical: float = 10.,
) -> None:
    print(
        '1 sphere, non phantom, z-alignment, point source external function.')
    print('Fixed distance, convergence in degree.')
    pi = sigma_i / sigma_e
    num = max_l + 1
    
    b_d = (harmonicex.
    point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
    point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b_max = righthands.b_vector_1_sphere_mtf(r, 1./pi, b_d, b_n)
    
    a_0 = laplace.a_0j_matrix(max_l, r, azimuthal=True)
    a_1 = laplace.a_j_matrix(max_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution2 = np.linalg.solve(matrix, b_max)
    
    print('--- Checking of errors.')
    dirichlet_ex = solution2[0:num]
    neumann_ex = solution2[num:2 * num]
    dirichlet_in = solution2[2 * num:3 * num]
    neumann_in = solution2[3 * num:4 * num]
    print('---- Discrete Calderon errors:')
    print(np.linalg.norm(2 * np.matmul(a_0, solution2[0:2 * num])
                         - r**2 * solution2[0:2 * num]))
    print(
        np.linalg.norm(2 * np.matmul(a_1, solution2[2 * num:4 * num])
                       - r**2 * solution2[2 * num:4 * num]))
    print('---- Jump errors.')
    print('----- Dirichlet trace:')
    jump_dirichlet = np.linalg.norm(dirichlet_ex - dirichlet_in + b_d)
    print(jump_dirichlet)
    print('----- Neumann trace:')
    jump_neumann = \
        np.linalg.norm(sigma_e * (neumann_ex + b_n) + sigma_i * neumann_in)
    print(jump_neumann)
    print('----- Total jump error:')
    print(np.sqrt(jump_dirichlet**2 + jump_neumann**2))
    
    solutions = np.zeros((4 * num, max_l))
    errores = np.zeros((4 * num, max_l))
    for el in np.arange(0, max_l):
        now_num = el + 1
        b_d = (harmonicex.
        point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity))
        b_n = harmonicex. \
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity)
        b = righthands.b_vector_1_sphere_mtf(r, 1./pi, b_d, b_n)
        a_0 = laplace.a_0j_matrix(el, r, azimuthal=True)
        a_1 = laplace.a_j_matrix(el, r, azimuthal=True)
        matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
        solution = np.linalg.solve(matrix, b)
        solutions[0:now_num, el] = solution[0:now_num]
        solutions[num:num + now_num, el] = solution[now_num:2 * now_num]
        solutions[2 * num:2 * num + now_num, el] = solution[
                                                   2 * now_num:3 * now_num]
        solutions[3 * num:3 * num + now_num, el] = solution[
                                                   3 * now_num:4 * now_num]
        errores[:, el] = solutions[:, el] - solution2
    y1 = np.linalg.norm(errores[0:num], axis=0) / np.linalg.norm(dirichlet_ex)
    y2 = np.linalg.norm(errores[num:2 * num], axis=0) / np.linalg.norm(
        neumann_ex)
    y3 = np.linalg.norm(errores[2 * num:3 * num], axis=0) / np.linalg.norm(
        dirichlet_in)
    y4 = np.linalg.norm(errores[3 * num:4 * num], axis=0) / np.linalg.norm(
        neumann_in)
    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.semilogy(
        y1, marker='p', linestyle='dashed',
        label='$RE2\\left(\\gamma_D^{01}u_0, u_{D,01}^{L}\\right)_1$')
    plt.semilogy(
        y2, marker='*', linestyle='dashed',
        label='$RE2\\left(\\gamma_N^{01}u_0, u_{N,01}^{L}\\right)_1$')
    plt.semilogy(
        y3, marker='x', linestyle='dashed',
        label='$RE2\\left(\\gamma_D^{1}u_0, u_{D,1}^{L}\\right)_1$')
    plt.semilogy(
        y4, marker='.', linestyle='dashed',
        label='$RE2\\left(\\gamma_N^{1}u_0, u_{N,1}^{L}\\right)_1$')
    plt.xlabel('$L$')
    plt.ylabel('Error')
    plt.legend(edgecolor='white')
    plt.rcParams.update({'font.size': 20})
    center = np.asarray([0., 0., 0.])
    inter_horizontal = resolution
    inter_vertical = resolution
    center_positions = [center]
    radius = np.asarray([r])
    
    
    num_big = num**2
    aux_drawing = np.zeros(4 * num_big)
    aux_drawing[
    0:num_big] = extensions.azimuthal_trace_to_general_with_zeros(
        max_l, solution2[0:num])
    aux_drawing[
    num_big:2 * num_big] = extensions.azimuthal_trace_to_general_with_zeros(
        max_l, solution2[num:2 * num])
    
    
    big_l_c = 100
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(
        max_l)
    quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics = \
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            max_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    eles = np.arange(0, max_l + 1)
    l_square_plus_l = (eles + 1) * eles
    
    vector = pre_vector * radius
    
    surface_field_d0 = np.sum(
        solution2[0:num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                   l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_d0_max = np.max(surface_field_d0)
    surface_field_d0_min = np.min(surface_field_d0)
    surface_field_d0 = (surface_field_d0 - surface_field_d0_min) \
                       / (surface_field_d0_max - surface_field_d0_min)
    surface_field_n0 = np.sum(
        solution2[num:2 * num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                         l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_n0_max = np.max(surface_field_n0)
    surface_field_n0_min = np.min(surface_field_n0)
    surface_field_n0 = (surface_field_n0 - surface_field_n0_min) \
                       / (surface_field_n0_max - surface_field_n0_min)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_d0))
    ax_1.set_xlabel('$x \\ [\\mu m]$')
    ax_1.set_ylabel('$y \\ [\\mu m]$')
    ax_1.set_zlabel('$z \\ [\\mu m]$')
    ax_1.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax_1,
        label='[V]'
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_n0))
    ax.set_xlabel('$x \\ [\\mu m]$')
    ax.set_ylabel('$y \\ [\\mu m]$')
    ax.set_zlabel('$z \\ [\\mu m]$')
    ax.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax,
        label='[V $/ \\mu m$ ]'
    )
    
    def point_source(x: np.ndarray) -> float:
        return 0.
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm(halfrange=0.00045)
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar()
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',  # vmin=0., vmax=0.0032,
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',  # vmin=0., vmax=0.0032,
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    aux_drawing[2 * num_big:3 * num_big] = \
        extensions.azimuthal_trace_to_general_with_zeros(
            max_l, solution2[2 * num:3 * num] + b_max[0:num])
    aux_drawing[3 * num_big:4 * num_big] = \
        extensions.azimuthal_trace_to_general_with_zeros(
            max_l, solution2[3 * num:4 * num] - b_max[num:2 * num])
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar()
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    pass


if __name__ == '__main__':
    testing_mtf_linear_operators_and_matrices_laplace()
    testing_mtf_linear_operators_and_matrices_laplace()
    testing_mtf_azimuthal_and_no_azimuthal_laplace()
    testing_mtf_reduced_linear_operators_and_matrices_laplace()
    testing_mtf_reduced_azimuthal_and_no_azimuthal_laplace()
    testing_mtf_reduced_vs_not_laplace()
    phantom_1_point_source_azimuthal(resolution=100)
    plt.show()
    non_phantom_1_point_source_z_alignment_distance_convergence(resolution=100)
    plt.show()
