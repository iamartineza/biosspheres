import numpy as np
import scipy.linalg as splinalg
import biosspheres.formulations.mtf.timecouplings.linearoperators as tclo


def tau_times_medium(initial_time: float, final_time: float, number_steps: int):
    tau = (final_time - initial_time) / number_steps
    times = tau * np.arange(0, number_steps + 1)
    medium_times = np.zeros(len(times) - 1)
    indexes = np.arange(0, number_steps)
    medium_times[indexes] = times[indexes + 1] + times[indexes]
    medium_times *= 0.5
    return tau, times, medium_times


def mtf_time_coupling_one_sphere(
    big_l: int,
    radius: float,
    sigmas: np.ndarray,
    c_m: float,
    initial_time: float,
    final_time: float,
    number_steps: int,
    initial_conditions: np.ndarray,
    b_phi_part_time_function,
    i_current,
    ode_next_sol,
):
    pi = sigmas[1] / sigmas[0]

    tau, times, medium_times = tau_times_medium(
        initial_time, final_time, number_steps
    )
    del times

    mtf_matrix = tclo.mtf_coupled_one_sphere_matrix_version(
        big_l, radius, pi, sigmas, c_m, tau
    )
    lu, piv = splinalg.lu_factor(mtf_matrix)
    del mtf_matrix
    half_iden_2n_n = tclo.mtf_coupling_iden_parts_sparse_array(
        big_l, 1, np.asarray([radius])
    )
    c_m_matrix = tclo.mtf_coupling_c_m_part_sparse_array_one_sphere(
        big_l, radius, c_m
    )

    num = (big_l + 1) ** 2
    number_of_extra_unknowns = len(initial_conditions[:, 0]) - 1

    solutions = np.empty(
        (number_steps + 1, (5 + number_of_extra_unknowns) * num)
    )
    solutions[0, :] = 0.0
    solutions[0, 4 * num : 5 * num] = initial_conditions[0, :]
    solutions[0, 5 * num : (5 + number_of_extra_unknowns) * num] = (
        initial_conditions[1 : 1 + number_of_extra_unknowns, :]
    )

    # Predictor
    wes_0 = solutions[0, 4 * num : 5 * num]
    qu_0 = solutions[0, 5 * num : (5 + number_of_extra_unknowns) * num]

    b_i_current = i_current(wes_0, qu_0)

    b = np.empty((5 * num))
    aux = half_iden_2n_n.dot(wes_0)
    aux = np.concatenate((-aux, aux), axis=0)
    b[0 : 4 * num] = b_phi_part_time_function(medium_times[0]) + aux
    b[4 * num : 5 * num] = -tau * b_i_current + c_m_matrix.dot(wes_0)

    wes_1 = splinalg.lu_solve((lu, piv), b)[4 * num : 5 * num]
    qu_1 = ode_next_sol(qu_0, qu_0, wes_0)

    # Corrector
    w_hat = (3.0 * wes_1 - wes_0) / 2.0
    q_hat = (3.0 * qu_1 - qu_0) / 2.0

    b_i_current = i_current(w_hat, q_hat)
    aux = half_iden_2n_n.dot(wes_0)
    aux = np.concatenate((-aux, aux), axis=0)
    b[0 : 4 * num] = b_phi_part_time_function(medium_times[0]) + aux
    b[4 * num : 5 * num] = -tau * b_i_current + c_m_matrix.dot(wes_0)

    solutions[1, 0 : 5 * num] = splinalg.lu_solve((lu, piv), b)
    solutions[1, 5 * num : (5 + number_of_extra_unknowns) * num] = ode_next_sol(
        qu_0, q_hat, w_hat
    )
    del wes_0
    del wes_1
    del w_hat
    del q_hat

    v_hat = np.empty((num))
    q_hat = np.empty((number_of_extra_unknowns * num))

    # The other steps
    for i in np.arange(1, number_steps):  # computing step i+1
        v_hat[:] = (
            3.0 * solutions[i, 4 * num : 5 * num]
            - solutions[i - 1, 4 * num : 5 * num]
        ) * 0.5
        q_hat[:] = (
            3.0 * solutions[i, 5 * num : (5 + number_of_extra_unknowns) * num]
            - solutions[i - 1, 5 * num : (5 + number_of_extra_unknowns) * num]
        ) * 0.5

        b_i_current = i_current(v_hat, q_hat)
        aux = half_iden_2n_n.dot(solutions[i, 4 * num : 5 * num])
        aux = np.concatenate((-aux, aux), axis=0)
        b[0 : 4 * num] = b_phi_part_time_function(medium_times[i]) + aux
        b[4 * num : 5 * num] = -tau * b_i_current + c_m_matrix.dot(
            solutions[i, 4 * num : 5 * num]
        )

        solutions[i + 1, 0 : 5 * num] = splinalg.lu_solve((lu, piv), b)
        solutions[i + 1, 5 * num : (5 + number_of_extra_unknowns) * num] = (
            ode_next_sol(
                solutions[i, 5 * num : (5 + number_of_extra_unknowns) * num],
                q_hat,
                v_hat,
            )
        )
    return solutions
