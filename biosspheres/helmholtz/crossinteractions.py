import numpy as np
import scipy.special
from scipy import sparse
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.auxindexes as auxindexes


def v_0_sj_semi_analytic_v1d(
        big_l: int,
        k0: float,
        r_j: float,
        r_s: float,
        j_l_j: np.ndarray,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        final_length: int,
        transform: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator V_{s,j}^0 with
    Helmholtz kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array
    of one dimension.
    It is a SLOW routine, because it does not use any symmetry or
    properties of the spherical harmonics.

    Notes
    -----
    data_v[p(2p + 1) + q, l(2l + 1) + m] =
        ( V_{s,j}^0 Y_{l,m,j} ; \conjugate{Y_{p,q,s}} )_{L^2(S_s)}
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression V_{s,j}^0 Y_{l,m,j} is analytic. A quadrature scheme
    is used to compute the surface integral corresponding to the inner
    product.

    In this routine the quadrature points NEED to be ordered in an array
    of one dimension.
    It is a SLOW routine, because it does not use any symmetry or
    properties of the spherical harmonics.

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    k0 : float
        > 0
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    j_l_j : np.ndarray
    r_coord : np.ndarray
        Array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Length equals to
        final_length. Can come from the
        function from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Array of floats with the phi coordinate r of the quadrature
        points in the coordinate system s. Length equals to
        final_length. Can come from the function
        from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Array of floats with the coseno of the spherical coordinate
        theta of the quadrature points in the coordinate system s.
        Lengths equal to final_length. Can come from the function
        from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    final_length : int
        How many points for the surface integral.
        Can come from the function spherical_harmonic_transform_1d of
        the module biosspheres.quadratures.spheres.
    transform : np.ndarray
        of complex numbers with the evaluation of the spherical
        harmonics along with their weights in the quadrature points.
        Can come from the function real_spherical_harmonic_transform_1d
        of the module biosspheres.quadratures.spheres.

    Returns
    -------
    data_v : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes
        ordering.

    See Also
    --------
    v_0_sj_semi_analytic_v2d

    """
    argument = k0 * r_coord
    eles = np.arange(0, big_l + 1)
    
    legendre_functions = np.empty(
        ((big_l + 1) * (big_l + 2) // 2, final_length))
    h_l = np.empty((final_length, big_l + 1), dtype=np.complex128)
    for i in np.arange(0, final_length):
        h_l[i, :] = (scipy.special.spherical_jn(eles, argument[i])
                     + 1j * scipy.special.spherical_yn(eles, argument[i]))
        legendre_functions[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta_coord[i], csphase=-1, cnorm=1)
    
    exp_pos = np.empty((big_l, final_length), dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi_coord, out=exp_pos[m - 1, :])
    
    el_plus_1_square = (big_l + 1)**2
    data_v = np.empty(
        (el_plus_1_square, el_plus_1_square), dtype=np.complex128)
    
    eles_plus_1 = eles + 1
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    temp_l = np.empty_like(transform)
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = h_l[:, el] * transform
        temp_l_m[:] = (
                temp_l
                * legendre_functions[l_times_l_plus_l_divided_by_2[el], :])
        np.sum(temp_l_m, axis=1, out=data_v[:, l_square_plus_l[el]])
        data_v[:, l_square_plus_l[el]] = (
                j_l_j[el] * data_v[:, l_square_plus_l[el]])
        for m in np.arange(1, el + 1):
            temp_l_m[:] = (
                    temp_l
                    * legendre_functions[
                      l_times_l_plus_l_divided_by_2[el] + m, :])
            np.sum(temp_l_m * exp_pos[m - 1, :],
                   axis=1, out=data_v[:, l_square_plus_l[el] + m])
            data_v[:, l_square_plus_l[el] + m] = (
                    j_l_j[el] * data_v[:, l_square_plus_l[el] + m])
            np.sum(temp_l_m * (-1)**m / exp_pos[m - 1, :],
                   axis=1, out=data_v[:, l_square_plus_l[el] - m])
            data_v[:, l_square_plus_l[el] - m] = (
                    j_l_j[el] * data_v[:, l_square_plus_l[el] - m])
    data_v[:] = 1j * k0 * r_j**2 * r_s**2 * data_v[:]
    return data_v