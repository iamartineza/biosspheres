import numpy as np
import scipy.sparse.linalg
import biosspheres.helmholtz.selfinteractions as helmholtz
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.miscella.extensions as righthandsextensions


def testing_mtf_linear_operators_and_matrices_helmholtz(
    big_l: int = 10,
    r: float = 1.3,
    k0: float = 2.0,
    k1: float = 1.5,
    pi: float = 3.0,
    tole: float = 10.0 ** (-5),
) -> None:
    """
    Test for Helmholtz.
    Testing that the MTF linear operators and the MTF matrix do the same
    matrix for one sphere and azimuthal symmetry

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
        wave number of the exterior.
    k1 : float
        wave number of the interior.
    pi: float
        adimensional parameter
    tole: float
        tolerance for gmres

    Returns
    -------
    None

    """
    num = big_l + 1
    b = np.random.random((4 * num)) + 1j * np.random.random((4 * num))

    a_0 = helmholtz.a_0j_linear_operator(big_l, r, k0, True)
    a_1 = helmholtz.a_j_linear_operator(big_l, r, k1, True)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)

    a_0 = helmholtz.a_0j_matrix(big_l, r, k0, True)
    a_1 = helmholtz.a_j_matrix(big_l, r, k1, True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(2 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2 = np.linalg.solve(matrix, b)
    print(
        "\nRunning function "
        "testing_mtf_linear_operators_and_matrices_helmholtz"
    )
    print("- MTF azimuthal")
    print("  The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))

    num = (big_l + 1) ** 2
    b = np.random.random((4 * num))

    a_0 = helmholtz.a_0j_linear_operator(big_l, r, k0, False)
    a_1 = helmholtz.a_j_linear_operator(big_l, r, k1, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)

    a_0 = helmholtz.a_0j_matrix(big_l, r, k0, False)
    a_1 = helmholtz.a_j_matrix(big_l, r, k1, False)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)

    norms = []

    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(2 * num) ** 3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2 = np.linalg.solve(matrix, b)
    print("- MTF")
    print("  The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_azimuthal_and_no_azimuthal_helmholtz(
    big_l: int = 10,
    r: float = 1.3,
    k0: float = 2.0,
    k1: float = 1.5,
    pi: float = 3.0,
    tole: float = 10.0 ** (-5),
) -> None:
    """
    Test for Helmholtz.

    Notes
    -----
    For a right hand with azimuthal symmetry when extended by 0 to
    explicitly compute the result without considering the symmetry, the
    result should be the same with the azimuthal and the not azimuthal
    versions.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
        wave number of the exterior.
    k1 : float
        wave number of the interior.
    pi: float
        adimensional parameter
    tole: float
        tolerance for gmres

    Returns
    -------
    None

    """
    num = big_l + 1
    b = np.random.random((4 * num)) + 1j * np.random.random((4 * num))
    b_2_1 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[0:num]
    )
    b_2_2 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[num : 2 * num]
    )
    b_2_3 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[2 * num : 3 * num]
    )
    b_2_4 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[3 * num : 4 * num]
    )
    b2_12 = np.concatenate((b_2_1, b_2_2))
    b2_34 = np.concatenate((b_2_3, b_2_4))
    b2 = np.concatenate((b2_12, b2_34))

    a_0 = helmholtz.a_0j_linear_operator(big_l, r, k0, False)
    a_1 = helmholtz.a_j_linear_operator(big_l, r, k1, False)
    x_j = mtf.x_j_diagonal(big_l, r, pi, False)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, False)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    num = (big_l + 1) ** 2
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b2,
        tol=tole,
        restart=(4 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    num = big_l + 1

    a_0 = helmholtz.a_0j_linear_operator(big_l, r, k0, True)
    a_1 = helmholtz.a_j_linear_operator(big_l, r, k1, True)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)

    norms = []

    solution2, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(4 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2_1 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[0:num]
    )
    solution2_2 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[num : 2 * num]
    )
    solution2_3 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[2 * num : 3 * num]
    )
    solution2_4 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[3 * num : 4 * num]
    )
    solution2_12 = np.concatenate((solution2_1, solution2_2))
    solution2_34 = np.concatenate((solution2_3, solution2_4))
    solution2 = np.concatenate((solution2_12, solution2_34))
    print("\nRunning function testing_mtf_azimuthal_and_no_azimuthal_helmholtz")
    print("- The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_reduced_linear_operators_and_matrices_helmholtz(
    big_l: int = 10,
    r: float = 1.3,
    k0: float = 2.0,
    k1: float = 1.5,
    pi: float = 3.0,
    tole: float = 10.0 ** (-5),
) -> None:
    """
    Test for Helmholtz
    Testing that the formulations reduced linear operators and the A
    matrix for one sphere represent the same

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
        wave number of the exterior.
    k1 : float
        wave number of the interior.
    pi: float
        adimensional parameter
    tole: float
        tolerance for gmres

    Returns
    -------
    None

    """
    num = big_l + 1
    b = np.random.random((2 * num)) + 1j * np.random.random((2 * num))

    linear_operator = mtf.mtf_1_reduced_linear_operator_helmholtz(
        big_l, r, k0, k1, pi, True
    )
    matrix = mtf.mtf_1_reduced_matrix_helmholtz(big_l, r, k0, k1, pi, True)

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(2 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2 = np.linalg.solve(matrix, b)
    print(
        "\nRunning function testing_mtf_reduced_linear_operators_and_matrices"
    )
    print("- MTF reduced azimuthal")
    print("  The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))

    num = (big_l + 1) ** 2
    b = np.random.random((2 * num)) + 1j * np.random.random((2 * num))

    linear_operator = mtf.mtf_1_reduced_linear_operator_helmholtz(
        big_l, r, k0, k1, pi, False
    )
    matrix = mtf.mtf_1_reduced_matrix_helmholtz(big_l, r, k0, k1, pi, False)

    norms = []

    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(2 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2 = np.linalg.solve(matrix, b)
    print("- MTF reduced")
    print("  The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))

    pass


def testing_mtf_reduced_azimuthal_and_no_azimuthal_helmholtz(
    big_l: int = 10,
    r: float = 1.3,
    k0: float = 2.0,
    k1: float = 1.5,
    pi: float = 3.0,
    tole: float = 10.0 ** (-3),
) -> None:
    """
    Test for Helmholtz.

    Notes
    -----
    For a right hand with azimuthal symmetry when extended by 0 to
    explicitly compute the result without considering the symmetry, the
    result should be the same with the azimuthal and the not azimuthal
    versions.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
        wave number of the exterior.
    k1 : float
        wave number of the interior.
    pi: float
        adimensional parameter
    tole: float
        tolerance for gmres

    Returns
    -------
    None

    """
    num = big_l + 1
    b = np.random.random((2 * num)) + 1j * np.random.random((2 * num))
    b_2_1 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[0:num]
    )
    b_2_2 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, b[num : 2 * num]
    )
    b2 = np.concatenate((b_2_1, b_2_2))

    linear_operator = mtf.mtf_1_reduced_linear_operator_helmholtz(
        big_l, r, k0, k1, pi, False
    )

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    num = (big_l + 1) ** 2
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b2,
        tol=tole,
        restart=(2 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    num = big_l + 1

    linear_operator = mtf.mtf_1_reduced_linear_operator_helmholtz(
        big_l, r, k0, k1, pi, True
    )

    norms = []

    solution2, info_2 = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(2 * num) ** 2,
        callback=callback_function,
        callback_type="pr_norm",
    )
    solution2_1 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[0:num]
    )
    solution2_2 = righthandsextensions.azimuthal_trace_to_general_with_zeros(
        big_l, solution2[num : 2 * num]
    )
    solution2 = np.concatenate((solution2_1, solution2_2))
    print("\nRunning function testing_mtf_reduced_azimuthal_and_no_azimuthal")
    print("- The following should be zero or near.")
    print(np.linalg.norm(solution2 - solution))
    pass


def testing_mtf_reduced_vs_not_helmholtz(
    big_l: int = 10,
    r: float = 1.3,
    k0: float = 2.0,
    k1: float = 1.5,
    pi: float = 3.0,
    tole: float = 2.0 * 10.0 ** (-4),
) -> None:
    """
    Test for Helmholtz.

    Notes
    -----
    The formulations reduced system came from making the steps to obtain
    the Schur's complement, thus, having the correct right hand sides
    the result needs to be the same with or without taking the reduced
    system.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
        wave number of the exterior.
    k1 : float
        wave number of the interior.
    pi: float
        adimensional parameter
    tole: float
        tolerance for gmres

    Returns
    -------
    None

    """
    num = big_l + 1
    b = np.random.random((4 * num)) + 1j * np.random.random((4 * num))

    linear_operator_red = mtf.mtf_1_reduced_linear_operator_helmholtz(
        big_l, r, k0, k1, pi, True
    )
    a_0 = helmholtz.a_0j_linear_operator(big_l, r, k0, True)
    a_1 = helmholtz.a_j_linear_operator(big_l, r, k1, True)
    x_j = mtf.x_j_diagonal(big_l, r, pi, True)
    x_j_inv = mtf.x_j_diagonal_inv(big_l, r, pi, True)
    linear_operator = mtf.mtf_1_linear_operator(a_0, a_1, x_j, x_j_inv)

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    num = big_l + 1
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=tole,
        restart=(4 * num) ** 3,
        callback=callback_function,
        callback_type="pr_norm",
    )

    norms = []

    a_1_lin_op = helmholtz.a_j_linear_operator(big_l, r, k1)
    b_red_1 = b[0 : 2 * num] + 2.0 * (
        a_1_lin_op.matvec(b[2 * num : 4 * num])
        / mtf.x_j_diagonal(big_l, r, pi, True)
    )
    sol_red_1, info_2 = scipy.sparse.linalg.gmres(
        linear_operator_red,
        b_red_1,
        tol=tole,
        restart=(2 * num) ** 3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    sol_red_2 = (
        b[2 * num : 4 * num] + mtf.x_j_diagonal(big_l, r, pi, True) * sol_red_1
    )
    sol_red_2[:] = sol_red_2[:] / (r**4)
    sol_red_2[:] = 2.0 * a_1_lin_op.matvec(sol_red_2[:])
    solution_red_1 = np.concatenate((sol_red_1, sol_red_2))
    print("\nRunning function testing_mtf_reduced_vs_not")
    print("- The following should be zero or near.")
    print(np.linalg.norm(solution_red_1 - solution))
    pass


if __name__ == "__main__":
    print("\nHelmholtz. (Caution, convergence is harder to achieve)")
    testing_mtf_linear_operators_and_matrices_helmholtz()
    testing_mtf_azimuthal_and_no_azimuthal_helmholtz()
    testing_mtf_reduced_linear_operators_and_matrices_helmholtz()
    testing_mtf_reduced_azimuthal_and_no_azimuthal_helmholtz()
    testing_mtf_reduced_vs_not_helmholtz()
