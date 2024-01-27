import numpy as np
import scipy.sparse.linalg
import biosspheres.laplace.selfinteractions as selfinteractions
import biosspheres.miscella.extensions as extensions


def testing_big_a_linear_operators_and_matrices() -> None:
    """
    Testing that the A linear operators and the A matrix for one sphere
    represent the same.
    
    Notes
    -----
    A linear system with a random right hand side is solved with the
    linear operators and the matrices, thus the solution must be almost
    the same.
    
    Returns
    -------
    None
    
    """
    big_l = 5
    r = 2.5
    num = big_l + 1
    b = np.random.random((2 * num))
    
    linear_operator = selfinteractions.a_0j_linear_operator(big_l, r)
    matrix = selfinteractions.a_0j_matrix(big_l, r)
    
    norms = []
    
    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.
    
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b,
        tol=10**(-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    
    print('\nRunning function testing_big_a_linear_operators_and_matrices')
    print('- A_0 azimuthal')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    linear_operator = selfinteractions.a_j_linear_operator(big_l, r)
    matrix = selfinteractions.a_j_matrix(big_l, r, True)
    
    norms = []
    
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b,
        tol=10**(-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    
    print('- A_j azimuthal')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    num = (big_l + 1)**2
    b = np.random.random((2 * num))
    
    linear_operator = selfinteractions.a_0j_linear_operator(big_l, r, False)
    matrix = selfinteractions.a_0j_matrix(big_l, r, False)
    
    norms = []
    
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b,
        tol=10**(-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    
    print('- A_0j')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    linear_operator = selfinteractions.a_j_linear_operator(big_l, r, False)
    matrix = selfinteractions.a_j_matrix(big_l, r, False)
    
    norms = []
    
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b,
        tol=10**(-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type='pr_norm')
    solution2 = np.linalg.solve(matrix, b)
    
    print('- A_j')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_big_a_azimuthal_and_no_azimuthal() -> None:
    """
    Notes
    -----
    For a right hand with azimuthal symmetry when extended by 0 to
    explicitly compute the result without considering the symmetry, the
    result should be the same with the azimuthal and the not azimuthal
    versions.
    
    Returns
    -------
    None

    """
    big_l = 5
    r = 2.3
    
    num = (big_l + 1)
    b = np.random.random((2 * num))
    b_2_1 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[0:num])
    b_2_2 = extensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[num:2 * num])
    b2 = np.concatenate((b_2_1, b_2_2))
    
    linear_operator = selfinteractions.a_0j_linear_operator(big_l, r, False)
    
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
    
    linear_operator = selfinteractions.a_0j_linear_operator(big_l, r)
    
    norms = []
    
    solution2, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                                tol=10**(-13),
                                                restart=(2 * num)**3,
                                                callback=callback_function,
                                                callback_type='pr_norm')
    solution2_1 = extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[0:num])
    solution2_2 = extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[num:2 * num])
    solution2 = np.concatenate((solution2_1, solution2_2))
    print('\nRunning function testing_big_a_azimuthal_and_no_azimuthal')
    print('- A_0j')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    
    linear_operator = selfinteractions.a_j_linear_operator(big_l, r, False)
    
    norms = []
    
    num = (big_l + 1)**2
    solution, info = scipy.sparse.linalg.gmres(linear_operator, b2,
                                               tol=10**(-13),
                                               restart=2 * num**3,
                                               callback=callback_function,
                                               callback_type='pr_norm')
    num = (big_l + 1)
    
    linear_operator = selfinteractions.a_j_linear_operator(big_l, r)
    
    norms = []
    
    solution2, info = scipy.sparse.linalg.gmres(linear_operator, b,
                                                tol=10**(-13),
                                                restart=2 * num**3,
                                                callback=callback_function,
                                                callback_type='pr_norm')
    solution2_1 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[0:num])
    solution2_2 = \
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, solution2[num:2 * num])
    solution2 = np.concatenate((solution2_1, solution2_2))
    print('- A_j')
    print('  The following should be zero or near.')
    print(np.linalg.norm(solution2 - solution))
    pass


if __name__ == '__main__':
    testing_big_a_linear_operators_and_matrices()
    testing_big_a_azimuthal_and_no_azimuthal()
