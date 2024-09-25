import numpy as np
import pyshtools
import biosspheres.utils.validation.inputs as valin


def i_linear_resistive_current_function_creation_n_spheres(
    big_l: int, n: int, r_m: np.ndarray
):
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")

    num = (big_l + 1) ** 2

    def i_current(v: np.ndarray) -> np.ndarray:
        assert (
            len(v) == n * (big_l + 1) ** 2
        ), "length of v is not equal to n*(big_l+1)**2"
        i = np.empty(np.shape(v))
        for counter in np.arange(0, n):
            i[counter * num : ((counter + 1) * num)] = (
                v[counter * num : ((counter + 1) * num)] / r_m[counter]
            )
        return i

    return i_current


def i_fitzhughnagumo_one_sphere_function_creation(
    big_l: int,
    spherical_harmonics: np.ndarray,
    weights: np.ndarray,
    zeros: np.ndarray,
    pesykus: np.ndarray,
    p2_plus_p_plus_q: np.ndarray,
    p2_plus_p_minus_q: np.ndarray,
    l_square_plus_l: np.ndarray,
    eles: np.ndarray,
):
    valin.big_l_validation(big_l, "big_l")

    def i_current(v: np.ndarray, g: np.ndarray) -> np.ndarray:
        assert len(v) == (big_l + 1) ** 2, "length of v is not (big_l + 1)**2"
        assert len(g) == (big_l + 1) ** 2, "length of g is not (big_l + 1)**2"
        v_evaluated = np.sum(
            v[:, np.newaxis, np.newaxis] * spherical_harmonics, axis=0
        )
        temp = v_evaluated**3
        coefficients = pyshtools.expand.SHExpandGLQ(
            temp, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
        )
        i = np.empty(np.shape(v))
        i[p2_plus_p_plus_q] = coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        i[p2_plus_p_minus_q] = coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        i[l_square_plus_l] = coefficients[0, eles, 0]
        i[:] = i[:] / 3.0 - (v[:] + g[:])
        return i

    return i_current


def i_fitzhughnagumo_n_spheres_function_creation(
    big_l: int,
    n: int,
    spherical_harmonics: np.ndarray,
    weights: np.ndarray,
    zeros: np.ndarray,
    pesykus: np.ndarray,
    p2_plus_p_plus_q: np.ndarray,
    p2_plus_p_minus_q: np.ndarray,
    l_square_plus_l: np.ndarray,
    eles: np.ndarray,
):
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")

    i_function_for_one = i_fitzhughnagumo_one_sphere_function_creation(
        big_l,
        spherical_harmonics,
        weights,
        zeros,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
        l_square_plus_l,
        eles,
    )

    num = (big_l + 1) ** 2

    def i_current(v: np.ndarray, g: np.ndarray) -> np.ndarray:
        i = np.empty(np.shape(v))
        assert (
            len(v) == n * (big_l + 1) ** 2
        ), "length of v is not equal to n*(big_l+1)**2"
        assert (
            len(g) == n * (big_l + 1) ** 2
        ), "length of g is not equal to n*(big_l+1)**2"
        for s in np.arange(0, n):
            i[num * s : num * (s + 1)] = i_function_for_one(
                v[num * s : num * (s + 1)], g[num * s : num * (s + 1)]
            )
        return i

    return i_current
