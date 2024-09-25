import numpy as np
import scipy.sparse as sparse
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.utils.validation.inputs as valin


def mtf_coupling_sigma_times_tau_part_sparse_array(
    big_l: int, n: int, radii: np.ndarray, sigmas: np.ndarray, tau: float
):
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")
    valin.radii_validation(radii, "radii")
    valin.pii_validation(sigmas, "sigmas")
    valin.radius_validation(tau, "tau")

    num = (big_l + 1) ** 2
    sigma_matrix = np.zeros((n * num))
    rows_sigma_matrix = np.zeros((n * num), dtype=int)
    columns_sigma_matrix = np.zeros((n * num), dtype=int)

    rango = np.arange(0, num)

    number_sigmas = 0
    for counter in np.arange(0, n):
        counter_times_2 = counter * 2

        sigma_matrix[number_sigmas : (number_sigmas + num)] = (
            radii[counter] ** 2 * sigmas[counter + 1]
        )
        rows_sigma_matrix[number_sigmas : (number_sigmas + num)] = (
            rango + num * counter
        )
        columns_sigma_matrix[number_sigmas : (number_sigmas + num)] = (
            num * (counter_times_2 + 1) + rango
        )
        number_sigmas += num
    sigma_matrix *= tau
    sparse_sigma_matrix = sparse.csc_array(
        (sigma_matrix, (rows_sigma_matrix, columns_sigma_matrix)),
        shape=(n * num, 2 * n * num),
    )
    return sparse_sigma_matrix


def mtf_coupling_c_m_part_sparse_array_one_sphere(
    big_l: int, radius: float, c_m: float
):
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.radius_validation(radius, "radius")

    j_block_diag = mass.j_block(big_l, radius, azimuthal=False)
    c_m_matrix = sparse.dia_array(
        (j_block_diag * c_m, np.array([0])),
        shape=(len(j_block_diag), len(j_block_diag)),
    )
    return c_m_matrix


def mtf_coupling_iden_parts_sparse_array(big_l: int, n: int, radii: np.ndarray):

    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")
    valin.radii_validation(radii, "radii")

    num = (big_l + 1) ** 2
    rango = np.arange(0, num)

    iden_2n_n = np.zeros((n * num))
    rows_iden = np.zeros((n * num), dtype=int)
    columns_iden = np.zeros((n * num), dtype=int)

    number_iden = 0
    for counter in np.arange(0, n):
        counter_times_2 = counter * 2

        iden_2n_n[number_iden : (number_iden + num)] = radii**2 * 0.5
        rows_iden[number_iden : (number_iden + num)] = (
            rango + num * counter_times_2
        )
        columns_iden[number_iden : (number_iden + num)] = num * counter + rango
        number_iden += num
    sparse_iden_2n_n = sparse.csc_array(
        (iden_2n_n, (rows_iden, columns_iden)), shape=(2 * n * num, n * num)
    )
    return sparse_iden_2n_n


def mtf_coupled_one_sphere_matrix_version(
    big_l: int,
    radius: float,
    pi: float,
    sigmas: np.ndarray,
    c_m: float,
    tau: float,
) -> np.ndarray:
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.radius_validation(radius, "radius")
    valin.pi_validation(pi, "pi")
    valin.pii_validation(sigmas, "sigmas")
    valin.pi_validation(c_m, "c_m")
    valin.radius_validation(tau, "tau")

    a_0 = laplaceself.a_0j_matrix(big_l, radius, azimuthal=False)
    a_1 = laplaceself.a_j_matrix(big_l, radius, azimuthal=False)
    mtf_matrix = mtf.mtf_1_matrix(radius, pi, a_0, a_1)
    del a_0
    del a_1

    sigma_matrix = mtf_coupling_sigma_times_tau_part_sparse_array(
        big_l, 1, np.asarray([radius]), sigmas, tau
    ).toarray()

    c_m_matrix = mtf_coupling_c_m_part_sparse_array_one_sphere(
        big_l, radius, c_m
    ).toarray()

    half_iden_2n_n = mtf_coupling_iden_parts_sparse_array(
        big_l, 1, np.asarray([radius])
    )
    aux_matrix = np.concatenate(
        (half_iden_2n_n.toarray(), (-half_iden_2n_n).toarray()), axis=0
    )
    del half_iden_2n_n
    mtf_matrix = np.concatenate((mtf_matrix, aux_matrix), axis=1)
    sigma_matrix = np.concatenate(
        (np.zeros(np.shape(sigma_matrix)), sigma_matrix), axis=1
    )
    aux_matrix = np.concatenate((sigma_matrix, c_m_matrix), axis=1)
    mtf_matrix = np.concatenate((mtf_matrix, aux_matrix), axis=0)

    return mtf_matrix
