"""

Routine listings
----------------
b_vector_1_sphere_mtf
b_vector_n_spheres_mtf_cte_function
b_vector_n_spheres_mtf_linear_function_z
b_vector_n_spheres_mtf_point_source
b_vector_n_spheres_mtf_plane_wave

"""

import numpy as np
import pyshtools
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes
import biosspheres.utils.validation.inputs as valin


def b_vector_1_sphere_mtf(
    r: float, pi_inv: float, b_d: np.ndarray, b_n: np.ndarray
) -> np.ndarray:
    """

    Parameters
    ----------
    r : float
        > 0, radius of the sphere
    pi_inv : float
        > 0, adimensional parameter.
    b_d : np.ndarray
        for the Dirichlet traces. Must be one dimensional.
    b_n : np.ndarray
        for the Neumann traces. Must be one dimensional

    Returns
    -------
    b : numpy array
        Length 4 * length(b_d)
    """
    valin.radius_validation(r, "r")
    valin.pii_validation(pi_inv, "pi_inv")
    valin.numpy_array_validation(b_d, "b_d")
    valin.numpy_array_validation(b_n, "b_n")
    valin.dimensions_array_validation(b_d, "b_d", 1)
    valin.dimensions_array_validation(b_n, "b_n", 1)
    valin.finite_values_in_array(b_d, "b_d")
    valin.finite_values_in_array(b_n, "b_n")

    num = len(b_d)
    b = np.empty(4 * num, dtype=b_d.dtype)
    b[0:num] = -(r**2) * b_d
    b[num : 2 * num] = -(r**2) * b_n
    b[2 * num : 3 * num] = r**2 * b_d
    b[3 * num : 4 * num] = -pi_inv * r**2 * b_n
    return b


def b_vector_n_spheres_mtf_cte_function(
    n: int, big_l: int, radii: np.ndarray, cte: float, azimuthal: bool = True
) -> np.ndarray:
    """

    Parameters
    ----------
    n : int
        >= 1, number of spheres.
    big_l : int
        >= 0, max degree.
    radii : np.ndarray
        of floats, each entry >= 0, radii of the spheres.
    cte : float
        != 0.
    azimuthal : bool
        Default True.

    Returns
    -------
    b : numpy array
        of 1 dimension.
        If azimuthal = True:
            Length 4 * n * (big_l+1)
        Else:
            Length 4 * n * (big_l+1)
    """
    if azimuthal:
        num = big_l + 1
    else:
        num = (big_l + 1) ** 2
    b = np.zeros(4 * num * n)
    index = np.arange(0, n)
    b[2 * num * n + 2 * num * index] = (
        cte * 2.0 * np.sqrt(np.pi) * radii[index] ** 2
    )
    b[2 * num * index] = -b[2 * num * n + 2 * num * index]
    return b


def b_vector_n_spheres_mtf_linear_function_z(
    big_l: int, n: int, ps, cte: float, radii: np.ndarray, x_dia: np.ndarray
) -> np.array:
    """
    Function = cte * z.

    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    big_l : int
        >= 1, max degree.
    n : int
        >= 1, number of spheres.
    ps : list of numpy arrays
        the numpy arrays are of length 3, and represent
        the position vectors of the center of the spheres.
    cte : float
         Constant parameter.
    radii : np.ndarray
        of the radii of the spheres.
    x_dia : np.ndarray
        diagonal of the matrix X.

    Returns
    -------
    b: np.ndarray
        of 1 dimension and length 4 * n * (big_l+1)**2
    """
    num = (big_l + 1) ** 2
    b = np.zeros(4 * num * n)

    mini_el = 1
    num_mini_el = (mini_el + 1) ** 2
    indexes = np.arange(0, num_mini_el)

    final_length, pre_vector, transform = (
        quadratures.real_spherical_harmonic_transform_1d(mini_el, mini_el)
    )
    for index in np.arange(0, n):
        b[2 * num * index + indexes] = np.sum(
            (radii[index] * pre_vector[2, :] + ps[index][2]) * cte * transform,
            axis=1,
        )
        b[(2 * index + 1) * num + indexes] = np.sum(
            (-pre_vector[2, :] * cte) * transform, axis=1
        )
    b[2 * num * n : 4 * num * n] = b[0 : 2 * num * n] * x_dia
    b[0 : 2 * num * n] = -b[0 : 2 * num * n]
    return b


def b_vector_n_spheres_mtf_point_source(
    n: int,
    big_l: int,
    ps: list[np.array],
    p0: np.ndarray,
    rs: np.ndarray,
    sigma_e: float,
    x_dia: np.ndarray,
    mass_n_two_j_blocks: np.ndarray,
) -> np.array:
    """

    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    n : int
        >= 1, number of spheres.
    big_l : int
        >= 0, max degree.
    ps : list of numpy arrays
        of floats, the numpy arrays are of length 3, and represent
        the position vectors of the center of the spheres.
    p0 : numpy array
        of floats, length 3. Represents the position of the point source
        in the original coordinate system.
    rs : numpy array
        of the radii of the spheres.
    sigma_e : float
        > 0, parameter.
    x_dia : np.ndarray
    mass_n_two_j_blocks : np.ndarray.

    Returns
    -------
    b : numpy array
        of 1 dimension and length 4 * n * (big_l+1)**2
    """
    num = (big_l + 1) ** 2
    b = np.zeros(4 * num * n)

    eles, eles_times_eles_plus_one, l_times_l_plus_l_divided_by_2 = (
        auxindexes.eles_combination(big_l)
    )
    eles_minus_one = eles - 1
    two_el_plus_one = 2 * eles + 1

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    for index in np.arange(0, n):
        vector = p0 - ps[index]
        p = np.linalg.norm(vector)
        phi = np.arctan2(vector[1], vector[0])
        ratio = rs[index] / p
        legendre = pyshtools.legendre.PlmON(
            big_l, vector[2] / p, csphase=-1, cnorm=0
        )
        temp_p = ratio**eles / (p * two_el_plus_one * sigma_e)
        temp_p_d = (
            -eles * ratio**eles_minus_one / (p**2 * two_el_plus_one * sigma_e)
        )
        b[2 * num * index + eles_times_eles_plus_one] = (
            temp_p * legendre[l_times_l_plus_l_divided_by_2]
        )
        b[(2 * index + 1) * num + eles_times_eles_plus_one] = (
            temp_p_d * legendre[l_times_l_plus_l_divided_by_2]
        )

        b[2 * num * index + p2_plus_p_plus_q] = (
            temp_p[pesykus[:, 0]]
            * legendre[
                l_times_l_plus_l_divided_by_2[pesykus[:, 0]] + pesykus[:, 1]
            ]
            * np.cos(phi * pesykus[:, 1])
        )
        b[2 * num * index + p2_plus_p_minus_q] = (
            temp_p[pesykus[:, 0]]
            * legendre[
                l_times_l_plus_l_divided_by_2[pesykus[:, 0]] + pesykus[:, 1]
            ]
            * np.sin(phi * pesykus[:, 1])
        )

        b[(2 * index + 1) * num + p2_plus_p_plus_q] = (
            temp_p_d[pesykus[:, 0]]
            * legendre[
                l_times_l_plus_l_divided_by_2[pesykus[:, 0]] + pesykus[:, 1]
            ]
            * np.cos(phi * pesykus[:, 1])
        )
        b[(2 * index + 1) * num + p2_plus_p_minus_q] = (
            temp_p_d[pesykus[:, 0]]
            * legendre[
                l_times_l_plus_l_divided_by_2[pesykus[:, 0]] + pesykus[:, 1]
            ]
            * np.sin(phi * pesykus[:, 1])
        )
    b[2 * num * n : 4 * num * n] = b[0 : 2 * num * n] * x_dia
    b[0 : 2 * num * n] = -b[0 : 2 * num * n] * mass_n_two_j_blocks
    return b


def b_vector_n_spheres_mtf_plane_wave(
    n: int,
    big_l: int,
    ps: list[np.array],
    p_z: float,
    k0: float,
    a: float,
    radii: np.ndarray,
    x_dia: np.ndarray,
    mass_n_two_j_blocks: np.ndarray,
) -> np.array:
    """

    Notes
    -----
    All arrays that represent vectors are in the cartesian coordinate
    system.

    Parameters
    ----------
    n : int
        >= 1, number of spheres.
    big_l : int
        >= 0, max degree.
    ps : list of numpy arrays
        of floats, the numpy arrays are of length 3, and represent
        the position vectors of the center of the spheres.
    p_z : float
    k0: float
        > 0, wave number.
    radii : numpy array
        of the radii of the spheres.
    a : float
        > 0, parameter, intensity.
    x_dia : np.ndarray
    mass_n_two_j_blocks : np.ndarray.

    Returns
    -------
    b : numpy array
        of 1 dimension and length 4 * n * (big_l+1)**2
    """
    num = (big_l + 1) ** 2
    b = np.zeros(4 * num * n, dtype=np.complex128)

    eles = np.arange(0, big_l + 1)
    eles_plus_one = eles + 1
    eles_times_eles_plus_one = eles * eles_plus_one
    for index in np.arange(0, n):
        vector_z = p_z + ps[index][2]
        b[2 * num * index + eles_times_eles_plus_one] = (
            harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
                big_l, radii[index], vector_z, k0, a, azimuthal=True
            )
        )
        b[(2 * index + 1) * num + eles_times_eles_plus_one] = (
            harmonicex.plane_wave_coefficients_neumann_expansion_0j(
                big_l, radii[index], vector_z, k0, a, azimuthal=True
            )
        )
    b[2 * num * n : 4 * num * n] = b[0 : 2 * num * n] * x_dia
    b[0 : 2 * num * n] = -b[0 : 2 * num * n] * mass_n_two_j_blocks
    return b
