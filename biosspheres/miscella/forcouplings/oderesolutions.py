import numpy as np
import biosspheres.utils.validation.inputs as valin


def semi_implicit_fitzhughnagumo_one_sphere_next_creation(
    radius: float,
    tau: float,
    parameter_theta: float,
    parameter_b: float,
    parameter_a: float,
):

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
        next_step[0] += parameter_a * 2.0 * np.sqrt(np.pi) * radius**2
        next_step[:] *= tau
        next_step[:] += g_previous[:]
        return next_step

    return next_g


def semi_implicit_fitzhughnagumo_n_spheres_next_creation(
    big_l: int,
    n: int,
    radii: np.ndarray,
    tau: float,
    parameters_theta: np.ndarray,
    parameters_b: np.ndarray,
    parameters_a: np.ndarray,
):
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")

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
                parameters_a[counter] * 2.0 * np.sqrt(np.pi) * radii[counter]**2
            )
        next_step[:] *= tau
        next_step[:] += g_previous[:]
        return next_step

    return next_g
