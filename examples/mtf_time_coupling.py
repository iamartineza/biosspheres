import numpy as np
import biosspheres.formulations.mtf.timecouplings.solvertemplates as solve
import biosspheres.formulations.mtf.timecouplings.righthands as righthands
import biosspheres.miscella.forcouplings.currents as currents
import biosspheres.miscella.forcouplings.oderesolutions as oderesolutions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes


def one_sphere_fitzhughnagumo(
    big_l: int = 10,
    big_l_c: int = 30,
    radius: float = 1.0,
    sigmas: np.ndarray = np.asarray([1.5, 0.7]),
    cte: float = 3.0,
    c_m: float = 1.1,
    parameter_theta: float = 0.5,
    parameter_b: float = 0.7,
    parameter_a: float = 0.2,
    initial_time: float = 0.0,
    final_time: float = 10.0,
    number_steps: int = 10**3,
):
    num = (big_l + 1) ** 2

    initial_conditions = np.zeros((2, num))

    b_phi_part_time_function = righthands.phi_part_of_b_cte_space_and_time(
        big_l, 1, np.asarray([radius]), cte
    )

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )
    eles = np.arange(0, big_l + 1)
    l_square_plus_l = (eles + 1) * eles

    i_current = currents.i_fitzhughnagumo_one_sphere_function_creation(
        big_l,
        spherical_harmonics,
        weights,
        pre_vector[2, :, 0],
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
        l_square_plus_l,
        eles,
    )

    tau, times, medium_times = solve.tau_times_medium(
        initial_time, final_time, number_steps
    )

    ode_next_sol = (
        oderesolutions.semi_implicit_fitzhughnagumo_one_sphere_next_creation(
            radius, tau, parameter_theta, parameter_b, parameter_a
        )
    )
    solutions = solve.mtf_time_coupling_one_sphere(
        big_l,
        radius,
        sigmas,
        c_m,
        initial_time,
        final_time,
        number_steps,
        initial_conditions,
        b_phi_part_time_function,
        i_current,
        ode_next_sol,
    )

    return


if __name__ == "__main__":
    one_sphere_fitzhughnagumo()
