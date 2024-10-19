from typing import Callable
import numpy as np
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes
import biosspheres.utils.validation.inputs as valin


def semi_implicit_fitzhughnagumo_one_sphere_next_step(
    tau: float,
    parameter_theta: float,
    parameter_b: float,
    parameter_a: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    tau : float
        > 0 . Length of the time step.
    parameter_theta : float
    parameter_b : float
    parameter_a : float

    Returns
    -------
    next_g : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    """

    # Input validation
    valin.radius_validation(tau, "tau")
    valin.float_validation(parameter_theta, "parameter_theta")
    valin.float_validation(parameter_b, "parameter_b")
    valin.float_validation(parameter_a, "parameter_a")

    def next_g(
        g_previous: np.ndarray, g_hat: np.ndarray, v_hat: np.ndarray
    ) -> np.ndarray:
        assert len(g_previous) == len(
            g_hat
        ), "Inputs need to have the same length"
        assert len(g_previous) == len(
            v_hat
        ), "Inputs need to have the same length"
        next_step = parameter_theta * v_hat - parameter_b * g_hat
        next_step[0] += parameter_a * 2.0 * np.sqrt(np.pi)
        next_step[:] *= tau
        next_step[:] += g_previous[:]
        return next_step

    return next_g


def semi_implicit_fitzhughnagumo_n_spheres_next_creation(
    big_l: int,
    n: int,
    tau: float,
    parameters_theta: np.ndarray,
    parameters_b: np.ndarray,
    parameters_a: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Extension for n spheres of
    semi_implicit_fitzhughnagumo_one_sphere_next_creation

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    n : int
        > 0, number of spheres.
    tau : float
        > 0 . Length of the time step.
    parameters_theta : np.ndarray
    parameters_b : np.ndarray
    parameters_a : np.ndarray

    Returns
    -------
    next_g : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

    See Also
    --------
    semi_implicit_fitzhughnagumo_one_sphere_next_creation

    """
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")
    valin.radius_validation(tau, "tau")
    valin.full_float_array_validation(parameters_theta, "parameters_theta", 1)
    valin.full_float_array_validation(parameters_b, "parameters_b", 1)
    valin.full_float_array_validation(parameters_a, "parameters_a", 1)
    valin.one_dimensional_array_length_check(
        parameters_theta, "parameters_theta", n
    )
    valin.one_dimensional_array_length_check(parameters_b, "parameters_b", n)
    valin.one_dimensional_array_length_check(parameters_a, "parameters_a", n)

    num = (big_l + 1) ** 2

    def next_g(
        g_previous: np.ndarray, g_hat: np.ndarray, v_hat: np.ndarray
    ) -> np.ndarray:
        assert len(g_previous) == len(
            g_hat
        ), "Inputs need to have the same length"
        assert len(g_previous) == len(
            v_hat
        ), "Inputs need to have the same length"
        next_step = np.empty(np.shape(g_hat))
        for counter in np.arange(0, n):
            next_step[counter * num : ((counter + 1) * num)] = (
                parameters_theta[counter] * v_hat
                - parameters_b[counter] * g_hat
            )
            next_step[counter * num] += (
                parameters_a[counter]
                * 2.0
                * np.sqrt(np.pi)
            )
        next_step[:] *= tau
        next_step[:] += g_previous[:]
        return next_step

    return next_g


def beta_kavian_leguebe_ea_2014_1_sphere_2d(
    big_l: int,
    big_l_c: int,
    k_ep: float,
    v_rev: float,
) -> Callable[[np.ndarray], np.ndarray]:

    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.big_l_validation(big_l_c, "big_l_c")

    # Needed to obtain a spherical harmonic transform
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c
    )
    zeros = pre_vector[2, :, 0]

    # Auxiliary functions
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    eles, l_square_plus_l, el_square_plus_el_divided_by_two = (
        auxindexes.eles_combination(big_l)
    )

    def beta_function(v: np.ndarray) -> np.ndarray:
        assert len(v) == (big_l + 1) ** 2, "length of v is not (big_l + 1)**2"
        argument = v / v_rev
        temp = np.tanh(
            k_ep
            * v_rev
            * (
                np.absolute(
                    np.sum(
                        (
                            argument[:, np.newaxis, np.newaxis]
                            * spherical_harmonics
                        ),
                        axis=0,
                    )
                )
                - 1.0
            )
        )
        coefficients = pyshtools.expand.SHExpandGLQ(
            temp, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
        )
        beta = np.empty(np.shape(v))
        beta[p2_plus_p_plus_q] = coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        beta[p2_plus_p_minus_q] = coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        beta[l_square_plus_l] = coefficients[0, eles, 0]
        beta[:] *= 0.5
        beta[0] += np.sqrt(np.pi)
        return beta

    return beta_function


def next_step_z_kavian_leguebe_ea_2014_2d(
    big_l: int,
    big_l_c: int,
    tau: float,
    tau_ep: float,
    tau_res: float,
    k_ep: float,
    v_rev: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:

    beta_function = beta_kavian_leguebe_ea_2014_1_sphere_2d(big_l, big_l_c,
                                                            k_ep, v_rev)

    # Needed to obtain a spherical harmonic transform
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c
    )
    zeros = pre_vector[2, :, 0]

    # Auxiliary functions
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    eles, l_square_plus_l, el_square_plus_el_divided_by_two = (
        auxindexes.eles_combination(big_l)
    )

    def next_z(
        z_previous: np.ndarray, z_hat: np.ndarray, v_hat: np.ndarray
    ) -> np.ndarray:
        temp = np.sum(
            (beta_function(v_hat) - z_hat)[:, np.newaxis, np.newaxis]
            * spherical_harmonics,
            axis=0,
        )
        temp_2 = np.maximum(temp / tau_ep, temp / tau_res)
        coefficients = pyshtools.expand.SHExpandGLQ(
            temp_2, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
        )
        next_step = np.empty(np.shape(z_hat))
        next_step[p2_plus_p_plus_q] = coefficients[
            0, pesykus[:, 0], pesykus[:, 1]
        ]
        next_step[p2_plus_p_minus_q] = coefficients[
            1, pesykus[:, 0], pesykus[:, 1]
        ]
        next_step[l_square_plus_l] = coefficients[0, eles, 0]
        next_step[:] *= tau
        next_step[:] += z_previous[:]
        return next_step

    return next_z


def next_step_z_kavian_leguebe_ea_2014_2d_n(
    big_l: int,
    big_l_c: int,
    n: int,
    tau: float,
    tau_ep: np.ndarray,
    tau_res: np.ndarray,
    k_ep: np.ndarray,
    v_rev: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:

    # Needed to obtain a spherical harmonic transform
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c
    )
    zeros = pre_vector[2, :, 0]

    # Auxiliary functions
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    eles, l_square_plus_l, el_square_plus_el_divided_by_two = (
        auxindexes.eles_combination(big_l)
    )

    beta_functions_array = np.asarray(
        [
            beta_kavian_leguebe_ea_2014_1_sphere_2d(big_l, big_l_c, k_ep[s],
                                                    v_rev[s])
            for s in np.arange(0, n)
        ]
    )

    def next_z(
        z_previous: np.ndarray, z_hat: np.ndarray, v_hat: np.ndarray
    ) -> np.ndarray:

        next_step = np.empty(np.shape(z_hat))
        num = (big_l + 1) ** 2
        for s in np.arange(0, n):
            temp = np.sum(
                (
                    beta_functions_array[s](v_hat)
                    - z_hat[num * s : num * (s + 1)]
                )[:, np.newaxis, np.newaxis]
                * spherical_harmonics,
                axis=0,
            )
            temp_2 = np.maximum(temp / tau_ep[s], temp / tau_res[s])
            coefficients = pyshtools.expand.SHExpandGLQ(
                temp_2, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
            )
            next_step[num * s + p2_plus_p_plus_q] = coefficients[
                0, pesykus[:, 0], pesykus[:, 1]
            ]
            next_step[num * s + p2_plus_p_minus_q] = coefficients[
                1, pesykus[:, 0], pesykus[:, 1]
            ]
            next_step[num * s + l_square_plus_l] = coefficients[0, eles, 0]
        next_step[:] *= tau
        next_step[:] += z_previous[:]
        return next_step

    return next_z
