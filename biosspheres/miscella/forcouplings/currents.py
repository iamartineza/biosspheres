from typing import Callable
import numpy as np
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes
import biosspheres.utils.validation.inputs as valin


def i_linear_resistive_current_function_creation_one_sphere(
    big_l: int, radius: float, r_m: float
) -> Callable[[np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    radius : float
        > 0. Radius of the sphere.
    r_m : float
        Must be different from zero.

    Returns
    -------
    i_current : Callable[[np.ndarray], np.ndarray]
        Its argument must be an array of length (big_l+1)**2. It returns
        an array of the same size.

    """
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.radius_validation(radius, "radius")
    valin.pi_validation(r_m, "r_m")

    num = (big_l + 1) ** 2

    def i_current(v: np.ndarray) -> np.ndarray:
        assert len(v) == num, "Length of v is not equal to (big_l+1)**2"
        i = v * (radius**2 / r_m)
        return i

    return i_current


def i_linear_resistive_current_function_creation_n_spheres(
    big_l: int, n: int, radii: np.ndarray, r_m: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    n: int
        > 0, number of spheres.
    radii : np.ndarray
        Array with the radii of the spheres.
    r_m : np.ndarray
        of floats. Length n. Each entry must be different from zero.

    Returns
    -------
    i_current : Callable[[np.ndarray], np.ndarray]
        Its argument must be an array of length n*(big_l+1)**2. It
        returns an array of the same size.
    """
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.n_validation(n, "n")
    valin.radii_validation(radii, "radii")
    valin.pii_validation(r_m, "r_m")
    valin.one_dimensional_array_length_check(r_m, "r_m", n)
    valin.one_dimensional_array_length_check(radii, "radii", n)

    num = (big_l + 1) ** 2

    def i_current(v: np.ndarray) -> np.ndarray:
        assert (
            len(v) == n * (big_l + 1) ** 2
        ), "Length of v is not equal to n*(big_l+1)**2"
        i = np.empty(np.shape(v))
        for counter in np.arange(0, n):
            i[counter * num : ((counter + 1) * num)] = v[
                counter * num : ((counter + 1) * num)
            ] * (radii[counter] ** 2 / r_m[counter])
        return i

    return i_current


def i_fitzhughnagumo_one_sphere_function_creation(
    big_l: int,
    big_l_c: int,
    radius: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    big_l_c : int
        >= 0, parameter for the integral quadratures
    radius : float
        > 0. Radius of the sphere.

    Returns
    -------
    i_current : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Each argument must be an array of length (big_l+1)**2. It returns
        an array of the same size.

    """
    # Input validation
    valin.radius_validation(radius, "radius")

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
        i[:] = (i[:] / 3.0 - (v[:] + g[:])) * radius**2
        return i

    return i_current


def i_fitzhughnagumo_n_spheres_function_creation(
    big_l: int,
    big_l_c: int,
    n: int,
    radii: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    big_l_c : int
        >= 0, parameter for the integral quadratures
    n: int
        > 0, number of spheres.
    radii : np.ndarray
        Array with the radii of the spheres.

    Returns
    -------
    i_current : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Each argument must be an array of length n*(big_l+1)**2. It
        returns an array of the same size.

    """
    # Part of input validation
    valin.n_validation(n, "n")
    valin.radii_validation(radii, "radii")

    # The rest of the input validation is here
    i_function_for_one = np.asarray(
        [
            i_fitzhughnagumo_one_sphere_function_creation(
                big_l, big_l_c, radii[s]
            )
            for s in np.arange(0, n)
        ]
    )

    num = (big_l + 1) ** 2

    def i_current(v: np.ndarray, g: np.ndarray) -> np.ndarray:
        assert (
            len(v) == n * (big_l + 1) ** 2
        ), "length of v is not equal to n*(big_l+1)**2"
        assert (
            len(g) == n * (big_l + 1) ** 2
        ), "length of g is not equal to n*(big_l+1)**2"
        i = np.empty(np.shape(v))
        for s in np.arange(0, n):
            i[num * s : num * (s + 1)] = i_function_for_one[s](
                v[num * s : num * (s + 1)], g[num * s : num * (s + 1)]
            )
        return i

    return i_current


def i_kavian_leguebe_ea_2014_1_sphere_2d(
    big_l: int,
    big_l_c: int,
    radius: float,
    s_l: float,
    s_ir: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    big_l_c : int
        >= 0, parameter for the integral quadratures
    radius : float
        > 0. Radius of the sphere.
    s_l : float
        Must be different from zero.
    s_ir : float
        Must be different from zero.

    Returns
    -------
    i_current : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Each argument must be an array of length (big_l+1)**2. It returns
        an array of the same size.

    """

    # Part of the input validation, the rest is inside the next
    # functions
    valin.radius_validation(radius, "radius")
    valin.pi_validation(s_l, "s_l")
    valin.pi_validation(s_ir, "s_ir")

    # Needed to obtain a spherical harmonic transform.
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

    def i_current(v: np.ndarray, z: np.ndarray) -> np.ndarray:
        assert len(v) == (big_l + 1) ** 2, "length of v is not (big_l + 1)**2"
        assert len(z) == (big_l + 1) ** 2, "length of z is not (big_l + 1)**2"
        v_evaluated = np.sum(
            v[:, np.newaxis, np.newaxis] * spherical_harmonics, axis=0
        )
        z_evaluated = np.sum(
            z[:, np.newaxis, np.newaxis] * spherical_harmonics, axis=0
        )
        temp = z_evaluated * v_evaluated
        coefficients = pyshtools.expand.SHExpandGLQ(
            temp, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
        )
        i = np.empty(np.shape(v))
        i[p2_plus_p_plus_q] = coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        i[p2_plus_p_minus_q] = coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        i[l_square_plus_l] = coefficients[0, eles, 0]
        i[:] *= s_ir - s_l
        i[:] += s_l * v[:]
        i[:] *= radius**2
        return i

    return i_current


def i_kavian_leguebe_ea_2014_n_sphere_2d(
    big_l: int,
    big_l_c: int,
    n: int,
    radii: np.ndarray,
    s_l: np.ndarray,
    s_ir: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    big_l_c : int
        >= 0, parameter for the integral quadratures
    n : int
        > 0, number of spheres.
    radii : np.ndarray
        Array with the radii of the spheres.
    s_l : np.ndarray
        of floats. Length n. Each entry must be different from zero.
    s_ir : np.ndarray
        of floats. Length n. Each entry must be different from zero.

    Returns
    -------
    i_current : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Each argument must be an array of length n*(big_l+1)**2. It
        returns an array of the same size.

    """

    # Part of input validation
    valin.n_validation(n, "n")
    valin.radii_validation(radii, "radii")
    valin.pii_validation(s_l, "s_l")
    valin.pii_validation(s_ir, "s_ir")
    valin.one_dimensional_array_length_check(radii, "radii", n)
    valin.one_dimensional_array_length_check(s_l, "s_l", n)
    valin.one_dimensional_array_length_check(s_ir, "s_ir", n)

    # The rest of the input validation is here
    currents_array = np.asarray(
        [
            i_kavian_leguebe_ea_2014_1_sphere_2d(big_l, big_l_c, radii[s],
                                                 s_l[s], s_ir[s])
            for s in np.arange(0, n)
        ]
    )

    def i_current(v: np.ndarray, z: np.ndarray) -> np.ndarray:
        assert (
            len(v) == n * (big_l + 1) ** 2
        ), "length of v is not equal to n*(big_l+1)**2"
        assert (
            len(z) == n * (big_l + 1) ** 2
        ), "length of z is not equal to n*(big_l+1)**2"
        i = np.empty(np.shape(v))
        num = (big_l + 1) ** 2
        for s in np.arange(0, n):
            i[num * s : num * (s + 1)] = currents_array[s](
                v[num * s : num * (s + 1)], z[num * s : num * (s + 1)]
            )
        return i

    return i_current
