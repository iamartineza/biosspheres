import numpy as np
import pyshtools
import biosspheres.miscella.auxindexes as auxindexes


def function_zero(
        big_l: int,
        azimuthal: bool = True
) -> np.ndarray:
    if azimuthal:
        num = big_l + 1
    else:
        num = (big_l + 1) ** 2
    return np.zeros(num)


def function_cte_dirichlet(
        big_l: int,
        cte: float,
        azimuthal: bool = True
) -> np.ndarray:
    if azimuthal:
        num = big_l + 1
    else:
        num = (big_l + 1) ** 2
    function_coefficients = np.zeros(num)
    function_coefficients[0] = cte * 2. * np.sqrt(np.pi)
    return function_coefficients


def point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        big_l: int,
        radius: float,
        p_0_0: float,
        sigma_e: float,
        intensity: float
) -> np.ndarray:
    """
    Returns a numpy array with the coefficients of the spherical harmonic
    expansion of the Dirichlet trace, on the surface of the sphere
    of radius 1, of the function 1 / (4 pi sigma_e ||p_0 - r||_2).
    
    Only works in one half of the z axis.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    radius : float
        > 0, sphere radius.
    p_0_0 : float
        > 0, norm of the position vector of the source.
    sigma_e : float
        > 0, parameter.
    intensity : float
        > 0, parameter.

    Returns
    -------
    values: numpy array
        Length (big_l+1).
    """
    eles = np.arange(0, big_l+1)
    l2_1_times_4 = 4 * (2 * eles + 1)
    values = intensity * (radius / p_0_0) ** eles \
        / (p_0_0 * sigma_e * np.sqrt(np.pi * l2_1_times_4))
    return values


def point_source_coefficients_dirichlet_expansion(
        big_l: int,
        radius: float,
        p_0: np.ndarray,
        sigma_e: float,
        intensity: float
) -> np.ndarray:
    """
    Returns a numpy array with the coefficients of the spherical harmonic
    expansion of the Dirichlet trace, on the surface of the sphere
    of radius 1, of the function 1 / (4 pi sigma_e ||p_0 - r||_2).
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    radius : float
        > 0, sphere radius.
    p_0 : numpy array
        of three floats, position in cartesian coordinates relative to the
        center of the sphere.
    sigma_e : float
        > 0 , parameter.
    intensity : float
        > 0 , parameter.

    Returns
    -------
    values: numpy array
        Length (big_l+1)**2.
    """
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = \
        auxindexes.pes_y_kus(big_l)
    eles = np.arange(0, big_l + 1)
    l2_1 = 2*eles + 1

    l_square_plus_l = eles * (eles + 1)
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2

    p_norm = np.linalg.norm(p_0)
    phi_p = np.arctan2(p_0[1], p_0[0])
    legendre_functions = pyshtools.legendre.PlmON(big_l, p_0[2]/p_norm,
                                                  csphase=-1, cnorm=0)
    values_l = intensity * (radius / p_norm) ** eles \
        / (p_norm * sigma_e * l2_1)
    values = np.zeros((big_l + 1) ** 2)
    values[l_square_plus_l] = \
        values_l * legendre_functions[l_times_l_plus_l_divided_by_2]
    values[p2_plus_p_plus_q] = \
        values_l[pesykus[:, 0]] \
        * legendre_functions[l_times_l_plus_l_divided_by_2[pesykus[:, 0]]
                             + pesykus[:, 1]] \
        * np.cos(pesykus[:, 1] * phi_p)
    values[p2_plus_p_minus_q] = \
        values_l[pesykus[:, 0]] \
        * legendre_functions[l_times_l_plus_l_divided_by_2[pesykus[:, 0]]
                             + pesykus[:, 1]] \
        * np.sin(pesykus[:, 1] * phi_p)
    return values


def point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        big_l: int,
        radius: float,
        p_0_0: float,
        sigma_e: float,
        intensity: float
) -> np.ndarray:
    """
    Returns a numpy array with the coefficients of the spherical harmonic
    expansion of the Neumann trace, on the surface of the sphere
    of radius 1, of the function 1 / (4 pi sigma_e ||p_0 - r||_2).
    
    Only works in one half of the z axis.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    radius : float
        > 0, sphere radius.
    p_0_0 : float
        > 0, norm of the position vector of the source.
    sigma_e : float
        > 0, parameter.
    intensity : float
        > 0, parameter.

    Returns
    -------
    values: numpy array
        Length (big_l+1).
    """
    eles = np.arange(0, big_l+1)
    l2_1_times_4 = 4 * (2 * eles + 1)
    values = \
        -intensity * eles * (radius / p_0_0) ** (eles - 1) \
        / (p_0_0 ** 2 * sigma_e * np.sqrt(np.pi * l2_1_times_4))
    return values


def point_source_coefficients_neumann_expansion_0j(
        big_l: int,
        radius: float,
        p_0: np.ndarray,
        sigma_e: float,
        intensity: float
) -> np.ndarray:
    """
    Returns a numpy array with the coefficients of the spherical harmonic
    expansion of the Neumann trace, on the surface of the sphere
    of radius 1, of the function 1 / (4 pi sigma_e ||p_0 - r||_2).

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    radius : float
        > 0, sphere radius.
    p_0 : numpy array
        of three floats, position in cartesian coordinates relative to the
        center of the sphere.
    sigma_e : float
        > 0, parameter.
    intensity : float
        > 0, parameter.

    Returns
    -------
    values: numpy array
        Length (big_l+1)**2.
    """
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    num = big_l + 1
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1

    l_square_plus_l = eles * (eles + 1)
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2

    p_norm = np.linalg.norm(p_0)
    phi_p = np.arctan2(p_0[1], p_0[0])
    legendre_functions = pyshtools.legendre.PlmON(big_l, (p_0[2] / p_norm),
                                                  csphase=-1, cnorm=0)
    values_l = -intensity * eles * (radius / p_norm) ** (eles - 1) \
        / (p_norm**2 * sigma_e * l2_1)
    values = np.zeros(num ** 2)
    values[l_square_plus_l] = \
        values_l * legendre_functions[l_times_l_plus_l_divided_by_2]
    values[p2_plus_p_plus_q] = \
        values_l[pesykus[:, 0]] \
        * legendre_functions[l_times_l_plus_l_divided_by_2[pesykus[:, 0]]
                             + pesykus[:, 1]] \
        * np.cos(pesykus[:, 1] * phi_p)
    values[p2_plus_p_minus_q] = \
        values_l[pesykus[:, 0]] \
        * legendre_functions[l_times_l_plus_l_divided_by_2[pesykus[:, 0]]
                             + pesykus[:, 1]] \
        * np.sin(pesykus[:, 1] * phi_p)
    return values
