import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sparse
import biosspheres.laplace.selfinteractions as selfinteractions
import biosspheres.formulations.massmatrices as mass


def mass_matrix_inv_preconditioning_big_as(
    big_l: int = 10, r: float = 2.5
) -> None:
    num = big_l + 1
    b = np.random.random((2 * num))

    matrix = selfinteractions.a_0j_matrix(big_l, r, azimuthal=True)
    solution_direct = np.linalg.solve(matrix, b)

    norms = []

    def callback_function(pr_norm):
        norms.append(pr_norm)
        return 0.0

    linear_operator = selfinteractions.a_0j_linear_operator(
        big_l, r, azimuthal=True
    )

    solution_iter_without, info_iter_without = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_without = len(norms)
    print("\nRunning mass_matrix_inv_preconditioning_big_as")
    print("Azimuthal routines:")
    print("A_0j")
    print("- Without mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_without)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_without))

    norms = []
    mass_matrix = mass.two_j_blocks(big_l, r, azimuthal=True)
    mass_matrix = sparse.dia_array(
        (mass_matrix[np.newaxis, :], 0),
        shape=(len(mass_matrix), len(mass_matrix)),
    )
    solution_iter_with, info_iter_with = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        M=mass_matrix,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_with = len(norms)
    print("- With mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_with)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_with))

    matrix = selfinteractions.a_j_matrix(big_l, r, True)
    solution_direct = np.linalg.solve(matrix, b)
    linear_operator = selfinteractions.a_j_linear_operator(
        big_l, r, azimuthal=True
    )

    norms = []
    solution_iter_without, info_iter_without = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_without = len(norms)
    print("A_j")
    print("- Without mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_without)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_without))

    norms = []
    mass_matrix = mass.two_j_blocks(big_l, r, azimuthal=True)
    mass_matrix = sparse.dia_array(
        (mass_matrix[np.newaxis, :], 0),
        shape=(len(mass_matrix), len(mass_matrix)),
    )
    solution_iter_with, info_iter_with = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        M=mass_matrix,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_with = len(norms)
    print("- With mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_with)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_with))

    num = (big_l + 1) ** 2
    b = np.random.random((2 * num))

    matrix = selfinteractions.a_0j_matrix(big_l, r, azimuthal=False)
    solution_direct = np.linalg.solve(matrix, b)

    norms = []

    linear_operator = selfinteractions.a_0j_linear_operator(
        big_l, r, azimuthal=False
    )

    solution_iter_without, info_iter_without = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_without = len(norms)
    print("NOT Azimuthal routines:")
    print("A_0j")
    print("- Without mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_without)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_without))

    norms = []
    mass_matrix = mass.two_j_blocks(big_l, r, azimuthal=False)
    mass_matrix = sparse.dia_array(
        (mass_matrix[np.newaxis, :], 0),
        shape=(len(mass_matrix), len(mass_matrix)),
    )
    solution_iter_with, info_iter_with = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        M=mass_matrix,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_with = len(norms)
    print("- With mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_with)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_with))

    matrix = selfinteractions.a_j_matrix(big_l, r, False)
    solution_direct = np.linalg.solve(matrix, b)
    linear_operator = selfinteractions.a_j_linear_operator(
        big_l, r, azimuthal=False
    )

    norms = []
    solution_iter_without, info_iter_without = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_without = len(norms)
    print("A_j")
    print("- Without mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_without)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_without))

    norms = []
    mass_matrix = mass.two_j_blocks(big_l, r, azimuthal=False)
    mass_matrix = sparse.dia_array(
        (mass_matrix[np.newaxis, :], 0),
        shape=(len(mass_matrix), len(mass_matrix)),
    )
    solution_iter_with, info_iter_with = scipy.sparse.linalg.gmres(
        linear_operator,
        b,
        M=mass_matrix,
        tol=10 ** (-13),
        restart=2 * num**3,
        callback=callback_function,
        callback_type="pr_norm",
    )
    norms_length_with = len(norms)
    print("- With mass matrix inv preconditioning:")
    print("-- Iteration number:")
    print(norms_length_with)
    print("-- Difference with direct solution:")
    print(np.linalg.norm(solution_direct - solution_iter_with))
    pass


if __name__ == "__main__":
    mass_matrix_inv_preconditioning_big_as()
