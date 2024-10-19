import numpy as np
import biosspheres.formulations.mtf.timecouplings.solvertemplates as solve
import biosspheres.formulations.mtf.timecouplings.righthands as righthands
import biosspheres.miscella.forcouplings.currents as currents
import biosspheres.miscella.forcouplings.oderesolutions as oderesolutions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes
import matplotlib.pyplot as plt


def one_sphere_fitzhughnagumo(
    big_l: int = 10,
    big_l_c: int = 30,
    radius: float = 1.0,
    sigmas: np.ndarray = np.asarray([1.5, 0.7]),
    cte: float = 10.0,
    c_m: float = 1.1,
    parameter_theta: float = 0.27,
    parameter_b: float = 0.27,
    parameter_a: float = 0.21,
    initial_time: float = 0.0,
    final_time: float = 5.0,
    number_steps: int = 10**3,
):
    # Auxiliary parameter
    num = (big_l + 1) ** 2

    # Set up of initial condition
    initial_conditions = np.zeros((2, num))  # Initial condition equal to zero

    # Set up of phi for the right hand side
    b_phi_part_time_function = righthands.phi_part_of_b_cte_space_and_time(
        big_l, 1, np.asarray([radius]), cte
    )

    # Set up of the quadrature
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c
    )

    # Set up of the time steps
    tau, times, medium_times = solve.tau_times_medium(
        initial_time, final_time, number_steps
    )

    # Set up of the non-linear current
    i_current = currents.i_fitzhughnagumo_one_sphere_function_creation(big_l,
                                                                       big_l_c,
                                                                       1.)

    # Set up of the semi-implicit scheme for the "gating" variables (or similar)
    ode_next_sol = (
        oderesolutions.semi_implicit_fitzhughnagumo_one_sphere_next_step(tau,
                                                                         parameter_theta,
                                                                         parameter_b,
                                                                         parameter_a)
    )

    # Solve
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

    # Plottings

    # Auxiliary arrays
    eles = np.arange(0, big_l + 1)
    l_square_plus_l = (eles + 1) * eles

    plt.figure()
    little_partial_spherical = np.sqrt((2 * eles + 1))
    plt.plot(
        times,
        np.sum(
            solutions[:, (4 * num + l_square_plus_l[0 : big_l + 1])]
            * little_partial_spherical[0 : big_l + 1],
            axis=1,
        )
        / (2.0 * np.sqrt(np.pi)),
        label="Numerical solution",
        marker=".",
    )
    plt.title("v")
    plt.figure()
    little_partial_spherical = np.sqrt((2 * eles + 1))
    plt.plot(
        times,
        np.sum(
            solutions[:, (5 * num + l_square_plus_l[0 : big_l + 1])]
            * little_partial_spherical[0 : big_l + 1],
            axis=1,
        )
        / (2.0 * np.sqrt(np.pi)),
        label="Numerical solution",
        marker=".",
    )
    plt.title("g")
    return


if __name__ == "__main__":
    one_sphere_fitzhughnagumo()
    plt.show()
