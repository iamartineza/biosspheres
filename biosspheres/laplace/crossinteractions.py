import numpy as np
from scipy import sparse
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.auxindexes as auxindexes


def v_0_sj_semi_analytic_v1d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        final_length: int,
        transform: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of the
    matrix formed by the boundary integral operator V_{s,j}^0 with Laplace
    kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
    
    Notes
    -----
    data_v[p(2p + 1) + q, l(2l + 1) + m] =
        ( V_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    The expression V_{s,j}^0 Y_{l,m,j} is analytic. A quadrature scheme is used
    to compute the surface integral corresponding to the inner product.
    
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Array of floats with the spherical coordinate r of the quadrature
        points in the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Array of floats with the phi coordinate r of the quadrature points in
        the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Array of floats with the coseno of the spherical coordinate theta of
        the quadrature points in the coordinate system s. Lengths equal to
        final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    final_length : int
        How many points for the surface integral.
        Can come from the function spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.
    transform : np.ndarray
        of floats with the evaluation of the spherical harmonics along with
        their weights in the quadrature points.
        Can come from the function real_spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.

    Returns
    -------
    data_v : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.
    
    See Also
    --------
    v_0_sj_semi_analytic_v2d
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_1d
    biosspheres.quadratures.spheres.real_spherical_harmonic_transform_1d
    
    """
    ratio = r_j / r_coord

    legendre_functions = \
        np.empty(((big_l+1) * (big_l+2) // 2, final_length))
    for i in np.arange(0, final_length):
        legendre_functions[:, i] = \
            pyshtools.legendre.PlmON(big_l, cos_theta_coord[i],
                                     csphase=-1, cnorm=0)

    cos_m_phi = np.empty((big_l, final_length))
    sin_m_phi = np.empty((big_l, final_length))
    for m in np.arange(1, big_l + 1):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :])

    el_plus_1_square = (big_l+1)**2
    data_v = np.empty((el_plus_1_square, el_plus_1_square))

    eles = np.arange(0, big_l+1)
    l2_1 = 2*eles + 1
    eles_plus_1 = eles + 1

    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    temp_l = np.empty_like(transform)
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = ratio**eles_plus_1[el] * transform
        temp_l_m[:] = temp_l \
            * legendre_functions[l_times_l_plus_l_divided_by_2[el], :]
        np.sum(temp_l_m, axis=1, out=data_v[:, l_square_plus_l[el]])
        data_v[:, l_square_plus_l[el]] = (
            data_v[:, l_square_plus_l[el]] / l2_1[el])
        for m in np.arange(1, el+1):
            temp_l_m[:] = temp_l * \
                legendre_functions[l_times_l_plus_l_divided_by_2[el]+m, :]
            np.sum(temp_l_m * cos_m_phi[m-1, :],
                   axis=1, out=data_v[:, l_square_plus_l[el] + m])
            data_v[:, l_square_plus_l[el]+m] = (
                data_v[:, l_square_plus_l[el]+m] / l2_1[el])
            np.sum(temp_l_m * sin_m_phi[m-1, :],
                   axis=1, out=data_v[:, l_square_plus_l[el] - m])
            data_v[:, l_square_plus_l[el] - m] = (
                    data_v[:, l_square_plus_l[el] - m] / l2_1[el])
    data_v[:] = r_j * r_s**2 * data_v[:]
    return data_v


def v_0_sj_semi_analytic_v2d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        weights: np.ndarray,
        zeros: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: float,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of the
    matrix formed by the boundary integral operator V_{s,j}^0 with Laplace
    kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function from_sphere_s_cartesian_to_j_spherical_2d
    of the module biosspheres.quadratures.spheres.
    
    Notes
    -----
    data_v[p(2p + 1) + q, l(2l + 1) + m] =
        ( V_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    The expression V_{s,j}^0 Y_{l,m,j} is analytic. A quadrature scheme is used
    to compute the surface integral corresponding to the inner product.
    
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function from_sphere_s_cartesian_to_j_spherical_2d
    of the module biosspheres.quadratures.spheres.
    It uses functions from the package pyshtools to compute the spherical
    harmonic transforms.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the coseno of the spherical
        coordinate theta of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_theta_points : int
        how many points for the integral in theta.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_phi_points : int
        how many points for the integral in phi.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    pesykus : np.ndarray
        dtype int, shape ((big_l+1) * big_l // 2, 2).
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_plus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_minus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    
    Returns
    -------
    data_v : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.
    
    See Also
    --------
    v_0_sj_semi_analytic_v1d
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_2d
    gauss_legendre_trapezoidal_shtools_2d
    biosspheres.miscella.auxindexes.pes_y_kus
    
    """
    ratio = r_j / r_coord

    num = big_l+1

    legendre_functions = np.empty((num * (big_l+2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=0)

    cos_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    sin_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    for m in np.arange(1, num):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :, :])

    el_plus_1_square = num**2

    eles = np.arange(0, num)
    l2_1 = 2*eles + 1
    eles_plus_1 = eles + 1

    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2

    data_v = np.empty((el_plus_1_square, el_plus_1_square))
    
    coefficients = np.empty((2, big_l+1, big_l+1))
    temp_l = np.empty_like(ratio)
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = ratio ** eles_plus_1[el]
        coefficients[:] = (r_j / l2_1[el]) * pyshtools.expand.SHExpandGLQ(
            temp_l *
            legendre_functions[l_times_l_plus_l_divided_by_2[el], :, :],
            weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
        data_v[p2_plus_p_plus_q, l_square_plus_l[el]] = \
            coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        data_v[p2_plus_p_minus_q, l_square_plus_l[el]] = \
            coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        data_v[l_square_plus_l, l_square_plus_l[el]] = \
            coefficients[0, eles, 0]
        for m in np.arange(1, el+1):
            temp_l_m[:] = temp_l * \
                legendre_functions[l_times_l_plus_l_divided_by_2[el]+m, :]
            coefficients[:] = (r_j / l2_1[el]) * pyshtools.expand.SHExpandGLQ(
                temp_l_m * cos_m_phi[m-1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_v[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v[l_square_plus_l, l_square_plus_l[el] + m] = \
                coefficients[0, eles, 0]

            coefficients[:] = (r_j / l2_1[el]) * pyshtools.expand.SHExpandGLQ(
                temp_l_m * sin_m_phi[m-1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_v[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v[l_square_plus_l, l_square_plus_l[el] - m] =  \
                coefficients[0, eles, 0]
    del coefficients
    del temp_l
    del temp_l_m
    data_v[:] = r_s**2 * data_v[:]
    return data_v[:]


def v_0_js_from_v_0_sj(
        data_v_sj: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator V_{j,s}^0 with
    Laplace kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to V_{s,j}^0 (notice the
    change of the order of the indexes indicating the spheres).
    
    Notes
    -----
    data_v_js[p*(2p+1) + q, l*(2l+1) + m] =
        ( V_{j,s}^0 Y_{l,m,s} ; Y_{p,q,j} )_{L^2(S_j)}.
    Y_{l,m,s} : spherical harmonic degree l, order m, in the coordinate
        system s.
    S_j : surface of the sphere j.
    
    This computation uses the following result for this specific case:
    ( V_{j,s}^0 Y_{l,m,s} ; Y_{p,q,j} )_{L^2(S_j)}.
        = ( V_{s,j}^0 Y_{p,q,j} ; Y_{l,m,s} )_{L^2(S_s)}
    
    Parameters
    ----------
    data_v_sj: np.ndarray
        represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Laplace kernel evaluated and
        tested with spherical harmonics.

    Returns
    -------
    data_v_js: np.ndarray
        Same shape than data_v_js. See notes for the indexes ordering.
    
    See Also
    --------
    v_0_sj_semi_analytic_v1d
    v_0_sj_semi_analytic_v2d
    
    """
    data_v_js = np.transpose(data_v_sj)
    return data_v_js


def k_0_sj_semi_analytic_v1d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray, cos_theta_coord: np.ndarray,
        final_length: int,
        transform: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^0 with
    Laplace kernel evaluated and tested with spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
    
    Notes
    -----
    data_k[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    The expression K_{s,j}^0 Y_l,m,j can be obtained analytically. A quadrature
    scheme is used to compute the surface integral corresponding to the inner
    product.
    
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
            
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Array of floats with the spherical coordinate r of the quadrature
        points in the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Array of floats with the phi coordinate r of the quadrature points in
        the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Array of floats with the coseno of the spherical coordinate theta of
        the quadrature points in the coordinate system s. Lengths equal to
        final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    final_length : int
        How many points for the surface integral.
        Can come from the function spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.
    transform : np.ndarray
        of floats with the evaluation of the spherical harmonics along with
        their weights in the quadrature points.
        Can come from the function real_spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.

    Returns
    -------
    data_k : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.
    
    See Also
    --------
    k_0_sj_semi_analytic_v2d
    k_0_sj_from_v_0_sj
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_1d
    biosspheres.quadratures.spheres.real_spherical_harmonic_transform_1d
    
    """
    ratio = r_j / r_coord
    
    num = big_l + 1
    legendre_functions = \
        np.empty((num * (big_l + 2) // 2, final_length))
    for i in np.arange(0, final_length):
        legendre_functions[:, i] = \
            pyshtools.legendre.PlmON(big_l, cos_theta_coord[i],
                                     csphase=-1, cnorm=0)
    
    cos_m_phi = np.empty((big_l, final_length))
    sin_m_phi = np.empty((big_l, final_length))
    for m in np.arange(1, big_l + 1):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :])
    
    el_plus_1_square = num**2
    data_k = np.empty((el_plus_1_square, el_plus_1_square))
    
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    temp_l = np.zeros_like(transform)
    temp_l_m = np.zeros_like(temp_l)
    data_k[:, 0] = 0.
    # We omit l=0 because for that column the matrix entries are 0
    for el in eles[1:num]:
        temp_l[:] = \
            -el * ratio**eles_plus_1[el] / l2_1[el] * transform
        np.sum(
            temp_l * legendre_functions[l_times_l_plus_l_divided_by_2[el], :],
            axis=1, out=data_k[:, l_square_plus_l[el]])
        for m in np.arange(1, el + 1):
            temp_l_m[:] = (temp_l *
                           legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + m, :])
            np.sum(temp_l_m * cos_m_phi[m - 1, :], axis=1,
                   out=data_k[:, l_square_plus_l[el] + m])
            np.sum(temp_l_m * sin_m_phi[m - 1, :], axis=1,
                   out=data_k[:, l_square_plus_l[el] - m])
    data_k[:] = r_s**2 * data_k[:]
    return data_k


def k_0_sj_semi_analytic_v2d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        weights: np.ndarray,
        zeros: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: float,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of the
    matrix formed by the boundary integral operator K_{s,j}^0 with Laplace
    kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function from_sphere_s_cartesian_to_j_spherical_2d
    of the module biosspheres.quadratures.spheres.

    Notes
    -----
    data_k[p(2p + 1) + q, l(2l + 1) + m] =
        ( K_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression K_{s,j}^0 Y_l,m,j can be obtained analytically.
    A quadrature scheme is used to compute the other surface integral. In this
    routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function from_sphere_s_cartesian_to_j_spherical_2d
    of the module biosspheres.quadratures.spheres.
    It uses functions from the package pyshtools to compute the spherical
    harmonic transforms.

    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Can come from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Can come from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the coseno of the spherical
        coordinate theta of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Can come from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta variable.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta variable.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_theta_points : int
        how many points for the integral in theta.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_phi_points : int
        how many points for the integral in phi.
        Can come fron the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    pesykus : np.ndarray
        dtype int, shape ((big_l+1) * big_l // 2, 2).
        Used for the vectorization of some computations.
        Come from the function biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_plus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Come from the function biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_minus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Come from the function biosspheres.miscella.auxindexes.pes_y_kus(big_l)

    Returns
    -------
    data_k : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.

    See Also
    --------
    k_0_sj_semi_analytic_v1d
    k_0_sj_from_v_0_sj
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_2d
    gauss_legendre_trapezoidal_shtools_2d
    biosspheres.miscella.auxindexes.pes_y_kus

    """
    ratio = r_j / r_coord
    
    num = big_l + 1
    
    legendre_functions = np.empty((num * (big_l + 2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=0)
    
    cos_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    sin_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    for m in np.arange(1, num):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :, :])
    
    el_plus_1_square = num**2
    
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    data_k = np.empty((el_plus_1_square, el_plus_1_square))
    
    coefficients = np.empty((2, big_l + 1, big_l + 1))
    temp_l = np.zeros_like(ratio)
    temp_l_m = np.zeros_like(temp_l)
    data_k[:, 0] = 0.
    # We start from l=1 because for l=0 all the entries are 0.
    for el in eles[1:num]:
        temp_l[:] = -(el / l2_1[el]) * ratio**eles_plus_1[el]
        coefficients[:] = pyshtools.expand.SHExpandGLQ(
            temp_l *
            legendre_functions[l_times_l_plus_l_divided_by_2[el], :, :],
            weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
        data_k[p2_plus_p_plus_q, l_square_plus_l[el]] = \
            coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        data_k[p2_plus_p_minus_q, l_square_plus_l[el]] = \
            coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        data_k[l_square_plus_l, l_square_plus_l[el]] = \
            coefficients[0, eles, 0]
        for m in np.arange(1, el + 1):
            temp_l_m[:] = (
                temp_l *
                legendre_functions[l_times_l_plus_l_divided_by_2[el] + m, :])
            coefficients[:] = pyshtools.expand.SHExpandGLQ(
                temp_l_m * cos_m_phi[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_k[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_k[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_k[l_square_plus_l, l_square_plus_l[el] + m] = \
                coefficients[0, eles, 0]
            
            coefficients[:] = pyshtools.expand.SHExpandGLQ(
                temp_l_m * sin_m_phi[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_k[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_k[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_k[l_square_plus_l, l_square_plus_l[el] - m] = \
                coefficients[0, eles, 0]
    data_k[:] = r_s**2 * data_k[:]
    return data_k


def k_0_sj_from_v_0_sj(
        data_v: np.ndarray,
        r_j: float,
        el_diagonal: sparse.dia_array
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^0 with
    Laplace kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to the testing of
    V_{s,j}^0.
    
    Notes
    -----
    data_k[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    This computation uses the following result for this specific case:
    K_{s,j}^0 Y_{l,m} = -\frac{l}{r_j} V_{s,j}^0 Y_{l,m}.
    
    Parameters
    ----------
    data_v : np.ndarray
        that represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Laplace kernel evaluated and
        tested with spherical harmonics.
    r_j : float
        > 0, radius of the sphere j.
    el_diagonal : scipy.sparse.dia_array
        comes from biosspheres.miscella.auxindexes.diagonal_l_sparse

    Returns
    -------
    data_k : np.ndarray
        Same shape than data_v. See notes for the indexes ordering.
    
    See Also
    --------
    k_0_sj_semi_analytic_v1d
    k_0_sj_semi_analytic_v2d
    biosspheres.miscella.auxindexes.diagonal_l_sparse
    
    """
    data_k = np.matmul(data_v, (el_diagonal/-r_j).toarray())
    return data_k


def ka_0_sj_semi_analytic_recurrence_v1d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        er_times_n: np.ndarray,
        etheta_times_n: np.ndarray,
        ephi_times_n: np.ndarray,
        final_length: int,
        transform: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^{*0} with
    Laplace kernel evaluated and tested with spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
    
    Notes
    -----
    data_ka[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    The expression K_{s,j}^{*0} Y_l,m,j can be obtained analytically.
    A quadrature scheme is used to compute the other surface integral
    corresponding to the inner product.
    For computing the derivative in theta a recurrence formula for Legendre
    Functions is used.
    
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        One dimensional array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Length equals to
        final_length. Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    phi_coord : np.ndarray
        One dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Length equals to
        final_length. Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    cos_theta_coord : np.ndarray
        One dimensional array of floats with the coseno of the spherical
        coordinate theta of the quadrature points in the coordinate system s.
        Lenght equals to final_length.
        Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    er_times_n : np.ndarray
        Two dimensional array of floats with the canonical vector of the
        spherical coordinate r in the quadrature points in the coordinate
        system s. Shape equals to
        (3, final_length). Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    etheta_times_n : np.ndarray
        Two dimensional array of floats with the canonical vector of the
        spherical coordinate theta in the quadrature points in the coordinate
        system s. Shape equals to
        (3, final_length). Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    ephi_times_n : np.ndarray
        Two dimensional array of floats with the canonical vector of the
        spherical coordinate phi in the quadrature points in the coordinate
        system s. Shape equals to
        (3, final_length). Can come from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    final_length : int
        how many points for the surface integral,
        (big_l_c + 1) * (2 * big_l_c + 1).
    transform : np.ndarray
        of floats. Mapping of the real spherical harmonics times the weights.
        Shape ((big_l+1)**2, final_length)

    Returns
    -------
    data_ka : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.
    
    See Also
    --------
    ka_0_sj_semi_analytic_recurrence_v2d
    ka_0_sj_from_v_sj
    ka_0_sj_from_k_js
    
    """
    ratio = r_j / r_coord
    
    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    legendre_functions = \
        np.empty(((big_l + 1) * (big_l + 2) // 2, final_length))
    for i in np.arange(0, final_length):
        legendre_functions[:, i] = \
            pyshtools.legendre.PlmON(big_l, cos_theta_coord[i],
                                     csphase=-1, cnorm=0)
    sin_theta_coord = np.sqrt(1. - cos_theta_coord**2)
    
    cos_m_phi = np.empty((big_l, final_length))
    sin_m_phi = np.empty((big_l, final_length))
    for m in np.arange(1, big_l + 1):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :])
    
    el_plus_1_square = (big_l + 1)**2
    data_ka = np.empty((el_plus_1_square, el_plus_1_square))
    
    leg_aux = np.zeros_like(sin_theta_coord)
    index = (sin_theta_coord > 0)
    
    d_legendre_functions = np.zeros_like(ratio)
    temp_l_n = np.empty_like(transform)
    temp_l_m_n = np.empty_like(temp_l_n)
    phi_part = np.zeros_like(leg_aux)
    r_part = np.zeros_like(leg_aux)
    theta_part = np.zeros_like(leg_aux)
    for el in eles:
        if el > 0:
            d_legendre_functions[:] = (
                np.sqrt((el + 1) * el / 2.)
                * legendre_functions[l_times_l_plus_l_divided_by_2[el] + 1, :])
        temp_l_n[:] = ratio**(eles_plus_1[el] + 1) / l2_1[el] * transform
        temp_l_m_n[:] = (
            temp_l_n * (-eles_plus_1[el]
                        * legendre_functions[l_times_l_plus_l_divided_by_2[el],
                                             :]
                        * er_times_n
                        + d_legendre_functions * etheta_times_n))
        np.sum(temp_l_m_n, axis=1,
               out=data_ka[:, l_square_plus_l[el]])
        for m in np.arange(1, el + 1):
            if m < el:
                leg_aux[index] = m * \
                                 legendre_functions[
                                     l_times_l_plus_l_divided_by_2[el] + m,
                                     index] / sin_theta_coord[index]
                if m > 1:
                    d_legendre_functions[:] = \
                        (legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                            + m - 1, :]
                         * np.sqrt((el - m + 1) * (el + m))
                         - legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + m + 1, :]
                         * np.sqrt((el - m) * (el + m + 1))) * -0.5
                else:  # (m = 1) < el
                    d_legendre_functions[:] = \
                        (legendre_functions[l_times_l_plus_l_divided_by_2[el],
                         :]
                         * np.sqrt(el * (el + 1) * 2)
                         - legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + 2, :]
                         * np.sqrt((el - 1) * (el + 2))) * -0.5
                phi_part[:] = leg_aux * ephi_times_n
            else:  # el = m
                if m > 1:
                    d_legendre_functions[:] = (
                        legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                           + m - 1, :] * -np.sqrt(el / 2.))
                    leg_aux[index] = (
                        m *
                        legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                           + m, index]
                        / sin_theta_coord[index])
                    phi_part[:] = leg_aux * ephi_times_n
                else:  # l = m = 1
                    d_legendre_functions[:] = (
                        legendre_functions[l_times_l_plus_l_divided_by_2[1], :]
                        * -1.)
                    phi_part[:] = -np.sqrt(3 / np.pi) * 0.5 * ephi_times_n
            r_part[:] = (
                -eles_plus_1[el]
                * legendre_functions[l_times_l_plus_l_divided_by_2[el] + m, :]
                * er_times_n)
            theta_part[:] = d_legendre_functions * etheta_times_n
            
            temp_l_m_n[:] = temp_l_n * (
                (r_part + theta_part) * cos_m_phi[m - 1, :]
                - sin_m_phi[m - 1, :] * phi_part)
            np.sum(temp_l_m_n, axis=1, out=data_ka[:, l_square_plus_l[el] + m])
            
            temp_l_m_n[:] = temp_l_n * (
                (r_part + theta_part) * sin_m_phi[m - 1, :]
                + cos_m_phi[m - 1, :] * phi_part)
            np.sum(temp_l_m_n, axis=1, out=data_ka[:, l_square_plus_l[el] - m])
    data_ka[:] = -r_s**2 * data_ka[:]
    return data_ka


def ka_0_sj_semi_analytic_recurrence_v2d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        er_times_n: np.ndarray,
        etheta_times_n: np.ndarray,
        ephi_times_n: np.ndarray,
        weights: np.ndarray,
        zeros: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: float,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^{*0} with
    Laplace kernel evaluated and tested with spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d of the
    module biosspheres.quadratures.spheres.

    Notes
    -----
    data_ka[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression K_{s,j}^{*0} Y_l,m,j can be obtained analytically.
    A quadrature scheme is used to compute the other surface integral
    corresponding to the inner product.
    For computing the derivative in theta a recurrence formula for Legendre
    Functions is used.
    
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d of the
    module biosspheres.quadratures.spheres.
    It uses functions from the package pyshtools to compute the spherical
    harmonic transforms.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the coseno of the spherical
        coordinate theta of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Comes from biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    er_times_n : np.ndarray
        Three dimensional array of floats with the canonical vector of the
        spherical coordinate r in the quadrature points in the coordinate
        system s. Shape equals to
        (3, quantity_theta_points, quantity_phi_points).  Comes from
        biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    etheta_times_n : np.ndarray
        Three dimensional array of floats with the canonical vector of the
        spherical coordinate theta in the quadrature points in the coordinate
        system s. Shape equals to
        (3, quantity_theta_points, quantity_phi_points). Comes from
        biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    ephi_times_n : np.ndarray
        Three dimensional array of floats with the canonical vector of the
        spherical coordinate phi in the quadrature points in the coordinate
        system s. Shape equals to
        (3, quantity_theta_points, quantity_phi_points). Comes from
        biosspheres.quadratures.spheres.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_theta_points : int
        how many points for the integral in theta.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_phi_points : int
        how many points for the integral in phi.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    pesykus : np.ndarray
        dtype int, shape ((big_l+1) * big_l // 2, 2).
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_plus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_minus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)

    Returns
    -------
    data_ka : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.

    See Also
    --------
    ka_0_sj_semi_analytic_recurrence_v1d
    ka_0_sj_from_v_sj
    ka_0_sj_from_k_js
    
    """
    ratio = r_j / r_coord
    
    num = big_l + 1
    
    legendre_functions = np.empty((num * (big_l + 2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=0)
    
    cos_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    sin_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    for m in np.arange(1, num):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :, :])
    
    el_plus_1_square = num**2
    
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    data_ka = np.empty((el_plus_1_square, el_plus_1_square))
    
    sin_theta_coord = np.sqrt(1. - cos_theta_coord**2)
    leg_aux = np.zeros_like(sin_theta_coord)
    index = (sin_theta_coord > 0)
    
    int_ka = np.empty((2, big_l + 1, big_l + 1))
    d_legendre_functions = np.zeros_like(ratio)
    temp_l_n = np.zeros_like(ratio)
    temp_l_m_n = np.zeros_like(temp_l_n)
    phi_part = np.zeros_like(leg_aux)
    r_part = np.zeros_like(leg_aux)
    theta_part = np.zeros_like(leg_aux)
    
    for el in eles:
        if el > 0:
            d_legendre_functions[:] = (
                np.sqrt((el + 1) * el / 2.) *
                legendre_functions[l_times_l_plus_l_divided_by_2[el] + 1, :, :]
            )
        temp_l_n[:] = ratio**(eles_plus_1[el] + 1) / l2_1[el]
        temp_l_m_n[:] = temp_l_n * (
            -eles_plus_1[el] *
            legendre_functions[l_times_l_plus_l_divided_by_2[el], :, :]
            * er_times_n
            + d_legendre_functions * etheta_times_n)
        int_ka[:] = pyshtools.expand.SHExpandGLQ(
            temp_l_m_n, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
        data_ka[p2_plus_p_plus_q, l_square_plus_l[el]] = \
            int_ka[0, pesykus[:, 0], pesykus[:, 1]]
        data_ka[p2_plus_p_minus_q, l_square_plus_l[el]] = \
            int_ka[1, pesykus[:, 0], pesykus[:, 1]]
        data_ka[l_square_plus_l, l_square_plus_l[el]] = \
            int_ka[0, eles, 0]
        for m in np.arange(1, el + 1):
            if m < el:
                leg_aux[index] = m * \
                                 legendre_functions[
                                     l_times_l_plus_l_divided_by_2[el] + m,
                                     index] / sin_theta_coord[index]
                if m > 1:
                    d_legendre_functions[:] = \
                        (legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                            + m - 1, :, :]
                         * np.sqrt((el - m + 1) * (el + m))
                         - legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + m + 1, :, :]
                         * np.sqrt((el - m) * (el + m + 1))) * -0.5
                else:  # (m = 1) < el
                    d_legendre_functions[:] = \
                        (legendre_functions[l_times_l_plus_l_divided_by_2[el],
                         :, :]
                         * np.sqrt(el * (el + 1) * 2)
                         - legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + 2, :, :]
                         * np.sqrt((el - 1) * (el + 2))) * -0.5
                phi_part[:] = leg_aux * ephi_times_n
            else:  # el = m
                if m > 1:
                    d_legendre_functions[:] = \
                        legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                           + m - 1, :, :] * -np.sqrt(el / 2.)
                    leg_aux[index] = (
                        m *
                        legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                           + m, index] / sin_theta_coord[index]
                    )
                    phi_part[:] = leg_aux * ephi_times_n
                else:  # l = m = 1
                    d_legendre_functions[:] = (
                        legendre_functions[l_times_l_plus_l_divided_by_2[1],
                                           :, :] * -1.)
                    phi_part = -np.sqrt(3 / np.pi) * 0.5 * ephi_times_n
            r_part[:] = (
                -eles_plus_1[el] *
                legendre_functions[l_times_l_plus_l_divided_by_2[el] + m, :, :]
                * er_times_n)
            theta_part[:] = d_legendre_functions * etheta_times_n
            
            temp_l_m_n[:] = temp_l_n * (
                (r_part + theta_part) * cos_m_phi[m - 1, :, :]
                - sin_m_phi[m - 1, :, :] * phi_part)
            int_ka[:] = pyshtools.expand.SHExpandGLQ(
                temp_l_m_n,
                weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
            data_ka[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                int_ka[0, pesykus[:, 0], pesykus[:, 1]]
            data_ka[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                int_ka[1, pesykus[:, 0], pesykus[:, 1]]
            data_ka[l_square_plus_l, l_square_plus_l[el] + m] = \
                int_ka[0, eles, 0]
            
            temp_l_m_n[:] = temp_l_n * (
                (r_part + theta_part) * sin_m_phi[m - 1, :, :]
                + cos_m_phi[m - 1, :, :] * phi_part)
            int_ka[:] = pyshtools.expand.SHExpandGLQ(
                temp_l_m_n,
                weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
            data_ka[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                int_ka[0, pesykus[:, 0], pesykus[:, 1]]
            data_ka[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                int_ka[1, pesykus[:, 0], pesykus[:, 1]]
            data_ka[l_square_plus_l, l_square_plus_l[el] - m] = \
                int_ka[0, eles, 0]
    
    data_ka[:] = -r_s**2 * data_ka[:]
    return data_ka


def ka_0_sj_from_k_js(
        data_kjs: np.ndarray,
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^{*0} with
    Laplace kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to K_{j,s}^0 (notice the
    change of the order of the indexes indicating the spheres).
    
    Notes
    -----
    data_ka_sj[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    This computation uses the following result for this specific case:
    ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}
        = ( K_{j,s}^0 Y_p,q,s ; Y_l,m,j )_{L^2(S_j)}.
    
    Parameters
    ----------
    data_kjs : numpy array.
        represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Laplace kernel evaluated and
        tested with spherical harmonics.

    Returns
    -------
    data_ka_sj : numpy array.
        Same shape than data_kjs. See notes for the indexes ordering.
    
    See Also
    --------
    ka_0_sj_semi_analytic_recurrence_v1d
    ka_0_sj_semi_analytic_recurrence_v2d
    ka_0_sj_from_v_sj
    
    """
    data_ka_sj = np.transpose(data_kjs)
    return data_ka_sj


def ka_0_sj_from_v_sj(
        data_v_sj: np.ndarray,
        r_s: float,
        el_diagonal: sparse.dia_array
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^{*0} with
    Laplace kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to V_{s,j}^0.
    
    Notes
    -----
    data_ka[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.
    
    This computation uses the following result for this specific case:
    ( K_{s,j}^{*0} Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}
        = -\frac{p}{r_j} ( V_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}
    
    Parameters
    ----------
    data_v_sj : np.ndarray
        represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Laplace kernel evaluated and
        tested with spherical harmonics.
    r_s : float
        > 0, radius of the sphere s.
    el_diagonal : scipy.sparse.dia_array
        that comes from biosspheres.miscella.auxindexes.diagonal_l_sparse

    Returns
    -------
    data_ka : numpy array.
        Same shape than data_v_sj. See notes for the indexes ordering.
    
    See Also
    --------
    ka_0_sj_semi_analytic_recurrence_v1d
    ka_0_sj_semi_analytic_recurrence_v2d
    ka_0_sj_from_k_js
    
    """
    data_ka = (el_diagonal / -r_s).dot(data_v_sj)
    return data_ka


def w_0_sj_from_v_sj(
        data_v_sj: np.ndarray,
        r_j: float,
        r_s: float,
        el_diagonal: sparse.dia_array
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator W_{s,j}^0 with
    Laplace kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to V_{s,j}^0.

    Notes
    -----
    data_w[p*(2p+1) + q, l*(2l+1) + m] =
        ( W_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    This computation uses the following result for this specific case:
    ( W_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}
        = -\frac{l p}{r_j r_s} ( V_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}

    Parameters
    ----------
    data_v_sj : np.ndarray
        represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Laplace kernel evaluated and
        tested with spherical harmonics.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    el_diagonal : scipy.sparse.dia_array
        that comes from biosspheres.miscella.auxindexes.diagonal_l_sparse

    Returns
    -------
    data_w : numpy array.
        Same shape than data_v_js. See notes for the indexes ordering.

    """
    data_w = np.matmul((el_diagonal / -(r_j * r_s)).dot(data_v_sj),
                       el_diagonal.toarray())
    return data_w


def a_0_sj_and_js_v1d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        final_length: int,
        transform: np.ndarray,
        diagonal: np.ndarray
):
    """
    Returns two numpy arrays that represents a numerical approximation of two
    matrices formed by the following boundary integral operators:
    a_sj = [-K_{s,j}^0 , V_{s,j}^0 ]
           [ W_{s,j}^0 , K_{s,j}^{*0}]
    a_js = [-K_{j,s}^0 , V_{j,s}^0 ]
           [ W_{j,s}^0 , K_{j,s}^{*0}]
    with Laplace kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of one
    dimension.
    It is a SLOW routine, because it does not use any symmetry or properties
    of the spherical harmonics.
    
    Notes
    -----
    The only operator computed directly with the numerical quadrature is
    V_{s,j}^0, which follows the same steps as v_0_sj_semi_analytic_v1d.
    The others are computed with the same properties used in:
    v_0_js_from_v_0_sj
    k_0_sj_from_v_0_sj
    ka_0_sj_from_v_sj
    w_0_sj_from_v_sj
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Array of floats with the spherical coordinate r of the quadrature
        points in the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Array of floats with the phi coordinate r of the quadrature points in
        the coordinate system s. Length equals to final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Array of floats with the coseno of the spherical coordinate theta of
        the quadrature points in the coordinate system s. Lengths equal to
        final_length.
        Can come from the function from_sphere_s_cartesian_to_j_spherical_1d of
        the module biosspheres.quadratures.spheres.
    final_length : int
        How many points for the surface integral.
        Can come from the function spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.
    transform : np.ndarray
        of floats with the evaluation of the spherical harmonics along with
        their weights in the quadrature points.
        Can come from the function real_spherical_harmonic_transform_1d of the
        module biosspheres.quadratures.spheres.
    diagonal : np.ndarray
        from diagonal_l_dense

    Returns
    -------
    a_sj : numpy array
        Shape (2 * (big_l+1)**2, 2 * (big_l+1)**2).
    a_js : numpy array
        Shape (2 * (big_l+1)**2, 2 * (big_l+1)**2).
    
    See Also
    --------
    a_0_sj_and_js_v2d
    v_0_sj_semi_analytic_v1d
    v_0_js_from_v_0_sj
    k_0_sj_from_v_0_sj
    ka_0_sj_from_v_sj
    w_0_sj_from_v_sj
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_1d
    biosspheres.quadratures.spheres.real_spherical_harmonic_transform_1d
    biosspheres.miscella.auxindexes.diagonal_l_dense
    
    """
    ratio = r_j / r_coord
    
    num = big_l + 1
    
    legendre_functions = \
        np.empty((num * (big_l + 2) // 2, final_length))
    for i in np.arange(0, final_length):
        legendre_functions[:, i] = \
            pyshtools.legendre.PlmON(big_l, cos_theta_coord[i],
                                     csphase=-1, cnorm=0)
    
    cos_m_phi = np.empty((big_l, final_length))
    sin_m_phi = np.empty((big_l, final_length))
    for m in np.arange(1, num):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :])
    
    el_plus_1_square = num**2
    data_v_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_k_sj_minus = np.empty((el_plus_1_square, el_plus_1_square))
    data_ka_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_w_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_v_js = np.empty((el_plus_1_square, el_plus_1_square))
    data_k_js_minus = np.empty((el_plus_1_square, el_plus_1_square))
    data_ka_js = np.empty((el_plus_1_square, el_plus_1_square))
    data_w_js = np.empty((el_plus_1_square, el_plus_1_square))
    
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    temp_l = np.empty_like(transform)
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = ratio**eles_plus_1[el] * transform
        np.sum(temp_l
               * legendre_functions[l_times_l_plus_l_divided_by_2[el], :],
               axis=1, out=data_v_sj[:, l_square_plus_l[el]])
        data_v_sj[:, l_square_plus_l[el]] = (
            r_j * r_s**2 / l2_1[el] * data_v_sj[:, l_square_plus_l[el]])
        data_v_js[l_square_plus_l[el], :] = \
            data_v_sj[:, l_square_plus_l[el]]
        
        data_k_js_minus[l_square_plus_l[el], :] = \
            data_v_js[l_square_plus_l[el], :] * diagonal / r_s
        data_ka_sj[:, l_square_plus_l[el]] = \
            data_k_js_minus[l_square_plus_l[el], :] * -1.
        m_range = np.arange(1, el + 1)
        for m in m_range:
            temp_l_m[:] = (
                temp_l *
                legendre_functions[l_times_l_plus_l_divided_by_2[el] + m, :])
            np.sum(temp_l_m * cos_m_phi[m - 1, :], axis=1,
                   out=data_v_sj[:, l_square_plus_l[el] + m])
            data_v_sj[:, l_square_plus_l[el] + m] = (
                r_j * r_s**2 / l2_1[el]
                * data_v_sj[:, l_square_plus_l[el] + m])
            np.sum(temp_l_m * sin_m_phi[m - 1, :], axis=1,
                   out=data_v_sj[:, l_square_plus_l[el] - m])
            data_v_sj[:, l_square_plus_l[el] - m] = (
                    r_j * r_s**2 / l2_1[el]
                    * data_v_sj[:, l_square_plus_l[el] - m])
        if el > 0:
            data_k_sj_minus[:, l_square_plus_l[el]] = \
                el * data_v_sj[:, l_square_plus_l[el]] / r_j
            data_k_sj_minus[:, l_square_plus_l[el] + m_range] = \
                el * data_v_sj[:, l_square_plus_l[el] + m_range] / r_j
            data_k_sj_minus[:, l_square_plus_l[el] - m_range] = \
                el * data_v_sj[:, l_square_plus_l[el] - m_range] / r_j
            
            data_v_js[l_square_plus_l[el] + m_range, :] = \
                data_v_sj[:, l_square_plus_l[el] + m_range].T
            data_v_js[l_square_plus_l[el] - m_range, :] = \
                data_v_sj[:, l_square_plus_l[el] - m_range].T
            
            data_ka_js[l_square_plus_l[el], :] = \
                data_k_sj_minus[:, l_square_plus_l[el]] \
                * -1.
            data_ka_js[l_square_plus_l[el] + m_range, :] = \
                data_k_sj_minus[:, l_square_plus_l[el] + m_range].T \
                * -1.
            data_ka_js[l_square_plus_l[el] - m_range, :] = \
                data_k_sj_minus[:, l_square_plus_l[el] - m_range].T \
                * -1.
            
            data_k_js_minus[l_square_plus_l[el], :] = \
                data_v_js[l_square_plus_l[el], :] \
                * diagonal / r_s
            data_k_js_minus[l_square_plus_l[el] + m_range, :] = \
                data_v_js[l_square_plus_l[el] + m_range, :] \
                * diagonal / r_s
            data_k_js_minus[l_square_plus_l[el] - m_range, :] = \
                data_v_js[l_square_plus_l[el] - m_range, :] \
                * diagonal / r_s
            
            data_ka_sj[:, l_square_plus_l[el]] = \
                data_k_js_minus[l_square_plus_l[el], :] \
                * -1.
            data_ka_sj[:, l_square_plus_l[el] + m_range] = \
                data_k_js_minus[l_square_plus_l[el] + m_range, :].T \
                * -1.
            data_ka_sj[:, l_square_plus_l[el] - m_range] = \
                data_k_js_minus[l_square_plus_l[el] - m_range, :].T \
                * -1.
            
            data_w_sj[:, l_square_plus_l[el]] = \
                data_ka_sj[:, l_square_plus_l[el]] \
                * el / r_j
            data_w_sj[:, l_square_plus_l[el] + m_range] = \
                data_ka_sj[:, l_square_plus_l[el] + m_range] \
                * el / r_j
            data_w_sj[:, l_square_plus_l[el] - m_range] = \
                data_ka_sj[:, l_square_plus_l[el] - m_range] \
                * el / r_j
            
            data_w_js[l_square_plus_l[el], :] = \
                data_w_sj[:, l_square_plus_l[el]]
            data_w_js[l_square_plus_l[el] + m_range, :] = \
                data_w_sj[:, l_square_plus_l[el] + m_range].T
            data_w_js[l_square_plus_l[el] - m_range, :] = \
                data_w_sj[:, l_square_plus_l[el] - m_range].T
        
        else:
            data_k_sj_minus[:, l_square_plus_l[el]] = 0.
            data_ka_js[l_square_plus_l[el], :] = 0.
            
            data_w_sj[:, l_square_plus_l[el]] = 0.
            data_w_js[l_square_plus_l[el], :] = 0.
    
    a_js = np.concatenate((
        np.concatenate((data_k_js_minus, data_v_js), axis=1),
        np.concatenate((data_w_js, data_ka_js), axis=1)),
        axis=0)
    a_sj = np.concatenate((
        np.concatenate((data_k_sj_minus, data_v_sj), axis=1),
        np.concatenate((data_w_sj, data_ka_sj), axis=1)),
        axis=0)
    return a_sj, a_js


def a_0_sj_and_js_v2d(
        big_l: int,
        r_j: float,
        r_s: float,
        r_coord: np.ndarray,
        phi_coord: np.ndarray,
        cos_theta_coord: np.ndarray,
        weights: np.ndarray,
        zeros: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: int,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray,
        diagonal: np.ndarray
):
    """
    Returns two numpy arrays that represents a numerical approximation of two
    matrices formed by the following boundary integral operators:
    a_sj = [-K_{s,j}^0 , V_{s,j}^0 ]
           [ W_{s,j}^0 , K_{s,j}^{*0}]
    a_js = [-K_{j,s}^0 , V_{j,s}^0 ]
           [ W_{j,s}^0 , K_{j,s}^{*0}]
    with Laplace kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array of two
    dimensions, given by the function from_sphere_s_cartesian_to_j_spherical_2d
    of the module biosspheres.quadratures.spheres.
    
    Notes
    -----
    The only operator computed directly with the numerical quadrature is
    V_{s,j}^0, which follows the same steps as v_0_sj_semi_analytic_v2d.
    The others are computed with the same properties used in:
    v_0_js_from_v_0_sj
    k_0_sj_from_v_0_sj
    ka_0_sj_from_v_sj
    w_0_sj_from_v_sj
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree of spherical harmonics used.
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the coseno of the spherical
        coordinate theta of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Comes from the function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta variable.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_theta_points : int
        how many points for the integral in theta.
        Can come from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    quantity_phi_points : int
        how many points for the integral in phi.
        Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    pesykus : np.ndarray
        dtype int, shape ((big_l+1) * big_l // 2, 2).
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_plus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    p2_plus_p_minus_q : np.ndarray
        dtype int, length (big_l+1) * big_l // 2.
        Used for the vectorization of some computations.
        Comes from the function
        biosspheres.miscella.auxindexes.pes_y_kus(big_l)
    diagonal : np.ndarray
        from diagonal_l_dense

    Returns
    -------
    a_sj : numpy array
        Shape (2 * (big_l+1)**2, 2 * (big_l+1)**2).
    a_js : numpy array
        Shape (2 * (big_l+1)**2, 2 * (big_l+1)**2).
    
    See Also
    --------
    a_0_sj_and_js_v1d
    v_0_sj_semi_analytic_v2d
    v_0_js_from_v_0_sj
    k_0_sj_from_v_0_sj
    ka_0_sj_from_v_sj
    w_0_sj_from_v_sj
    biosspheres.quadratures.spheres.from_sphere_s_cartesian_to_j_spherical_2d
    gauss_legendre_trapezoidal_shtools_2d
    biosspheres.miscella.auxindexes.pes_y_kus
    biosspheres.miscella.auxindexes.diagonal_l_dense

    """
    ratio = r_j / r_coord
    
    num = big_l + 1
    
    legendre_functions = np.empty((num * (big_l + 2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=0)
    
    cos_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    sin_m_phi = np.empty((big_l, quantity_theta_points, quantity_phi_points))
    for m in np.arange(1, num):
        np.cos(m * phi_coord, out=cos_m_phi[m - 1, :, :])
        np.sin(m * phi_coord, out=sin_m_phi[m - 1, :, :])
    
    el_plus_1_square = num**2
    data_v_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_k_sj_minus = np.empty((el_plus_1_square, el_plus_1_square))
    data_ka_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_w_sj = np.empty((el_plus_1_square, el_plus_1_square))
    data_v_js = np.empty((el_plus_1_square, el_plus_1_square))
    data_k_js_minus = np.empty((el_plus_1_square, el_plus_1_square))
    data_ka_js = np.empty((el_plus_1_square, el_plus_1_square))
    data_w_js = np.empty((el_plus_1_square, el_plus_1_square))
    
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    coefficients = np.empty((2, big_l + 1, big_l + 1))
    temp_l = np.empty_like(ratio)
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = ratio**eles_plus_1[el]
        coefficients[:] = ((r_j * r_s**2 / l2_1[el])
                           * pyshtools.expand.SHExpandGLQ(
            temp_l *
            legendre_functions[l_times_l_plus_l_divided_by_2[el], :, :],
            weights, zeros, norm=4, csphase=-1, lmax_calc=big_l))
        data_v_sj[p2_plus_p_plus_q, l_square_plus_l[el]] = \
            coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        data_v_sj[p2_plus_p_minus_q, l_square_plus_l[el]] = \
            coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        data_v_sj[l_square_plus_l, l_square_plus_l[el]] = \
            coefficients[0, eles, 0]
        
        data_v_js[l_square_plus_l[el], :] = \
            data_v_sj[:, l_square_plus_l[el]]
        
        data_k_js_minus[l_square_plus_l[el], :] = \
            data_v_js[l_square_plus_l[el], :] \
            * diagonal / r_s
        
        data_ka_sj[:, l_square_plus_l[el]] = \
            data_k_js_minus[l_square_plus_l[el], :] \
            * -1.
        
        m_range = np.arange(1, el + 1)
        for m in m_range:
            temp_l_m[:] = \
                temp_l * \
                legendre_functions[l_times_l_plus_l_divided_by_2[el] + m, :]
            coefficients[:] = \
                (r_j * r_s**2 / l2_1[el]) * pyshtools.expand.SHExpandGLQ(
                    temp_l_m * cos_m_phi[m - 1, :, :], weights, zeros,
                    norm=4, csphase=-1, lmax_calc=big_l)
            data_v_sj[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v_sj[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v_sj[l_square_plus_l, l_square_plus_l[el] + m] = \
                coefficients[0, eles, 0]
            
            coefficients[:] = \
                (r_j * r_s**2 / l2_1[el]) * pyshtools.expand.SHExpandGLQ(
                    temp_l_m * sin_m_phi[m - 1, :, :], weights, zeros,
                    norm=4, csphase=-1, lmax_calc=big_l)
            data_v_sj[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v_sj[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v_sj[l_square_plus_l, l_square_plus_l[el] - m] = \
                coefficients[0, eles, 0]
        if el > 0:
            data_k_sj_minus[:, l_square_plus_l[el]] = \
                el * data_v_sj[:, l_square_plus_l[el]] / r_j
            data_k_sj_minus[:, l_square_plus_l[el] + m_range] = \
                el * data_v_sj[:, l_square_plus_l[el] + m_range] / r_j
            data_k_sj_minus[:, l_square_plus_l[el] - m_range] = \
                el * data_v_sj[:, l_square_plus_l[el] - m_range] / r_j
            
            data_v_js[l_square_plus_l[el] + m_range, :] = \
                data_v_sj[:, l_square_plus_l[el] + m_range].T
            data_v_js[l_square_plus_l[el] - m_range, :] = \
                data_v_sj[:, l_square_plus_l[el] - m_range].T
            
            data_ka_js[l_square_plus_l[el], :] = \
                data_k_sj_minus[:, l_square_plus_l[el]] \
                * -1.
            data_ka_js[l_square_plus_l[el] + m_range, :] = \
                data_k_sj_minus[:, l_square_plus_l[el] + m_range].T \
                * -1.
            data_ka_js[l_square_plus_l[el] - m_range, :] = \
                data_k_sj_minus[:, l_square_plus_l[el] - m_range].T \
                * -1.
            
            data_k_js_minus[l_square_plus_l[el], :] = \
                data_v_js[l_square_plus_l[el], :] \
                * diagonal / r_s
            data_k_js_minus[l_square_plus_l[el] + m_range, :] = \
                data_v_js[l_square_plus_l[el] + m_range, :] \
                * diagonal / r_s
            data_k_js_minus[l_square_plus_l[el] - m_range, :] = \
                data_v_js[l_square_plus_l[el] - m_range, :] \
                * diagonal / r_s
            
            data_ka_sj[:, l_square_plus_l[el]] = \
                data_k_js_minus[l_square_plus_l[el], :] \
                * -1.
            data_ka_sj[:, l_square_plus_l[el] + m_range] = \
                data_k_js_minus[l_square_plus_l[el] + m_range, :].T \
                * -1.
            data_ka_sj[:, l_square_plus_l[el] - m_range] = \
                data_k_js_minus[l_square_plus_l[el] - m_range, :].T \
                * -1.
            
            data_w_sj[:, l_square_plus_l[el]] = \
                data_ka_sj[:, l_square_plus_l[el]] \
                * el / r_j
            data_w_sj[:, l_square_plus_l[el] + m_range] = \
                data_ka_sj[:, l_square_plus_l[el] + m_range] \
                * el / r_j
            data_w_sj[:, l_square_plus_l[el] - m_range] = \
                data_ka_sj[:, l_square_plus_l[el] - m_range] \
                * el / r_j
            
            data_w_js[l_square_plus_l[el], :] = \
                data_w_sj[:, l_square_plus_l[el]]
            data_w_js[l_square_plus_l[el] + m_range, :] = \
                data_w_sj[:, l_square_plus_l[el] + m_range].T
            data_w_js[l_square_plus_l[el] - m_range, :] = \
                data_w_sj[:, l_square_plus_l[el] - m_range].T
        
        else:
            data_k_sj_minus[:, l_square_plus_l[el]] = 0.
            data_ka_js[l_square_plus_l[el], :] = 0.
            
            data_w_sj[:, l_square_plus_l[el]] = 0.
            data_w_js[l_square_plus_l[el], :] = 0.
    
    a_js = np.concatenate((
        np.concatenate((data_k_js_minus, data_v_js), axis=1),
        np.concatenate((data_w_js, data_ka_js), axis=1)),
        axis=0)
    a_sj = np.concatenate((
        np.concatenate((data_k_sj_minus, data_v_sj), axis=1),
        np.concatenate((data_w_sj, data_ka_sj), axis=1)),
        axis=0)
    return a_sj, a_js


def all_cross_interactions_n_spheres_v1d(
        n: int,
        big_l: int,
        big_l_c: int,
        radii: np.ndarray,
        center_positions
) -> np.ndarray:
    """
    Returns an array with all the cross interactions for the given n spheres.
    It is a SLOW routine.
    
    Notes
    -----
    Uses the following:
    quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    auxindexes.diagonal_l_dense(big_l)
    quadratures.from_sphere_s_cartesian_to_j_spherical_1d
    a_0_sj_and_js_v1d
    
    Parameters
    ----------
    n : int
        >= 2. Number of spheres.
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the quadrature.
    radii : np.ndarray
        Array with the radii of the spheres.
    center_positions : array_like
        List or array with the center position of the spheres

    Returns
    -------
    almost_big_a_0 : np.ndarray
        Shape ( 2 * n * (big_l + 1) ** 2, 2 * n * (big_l + 1) ** 2).
    
    See Also
    --------
    quadratures.real_spherical_harmonic_transform_1d
    auxindexes.diagonal_l_dense
    quadratures.from_sphere_s_cartesian_to_j_spherical_1d
    a_0_sj_and_js_v1d
    
    """
    big_l_plus_1_square = (big_l + 1) ** 2
    num = 2 * n * big_l_plus_1_square
    almost_big_a_0 = np.zeros((num, num))

    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    diagonal = auxindexes.diagonal_l_dense(big_l)
    
    r_coord_stf = np.empty(final_length)
    phi_coord_stf = np.empty_like(r_coord_stf)
    cos_theta_coord_stf = np.empty_like(r_coord_stf)
    
    a_sj = np.zeros((2 * big_l_plus_1_square, 2 * big_l_plus_1_square))
    a_js = np.zeros_like(a_sj)
    for s in np.arange(1, n + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, n + 1):
            j_minus_1 = j - 1
            r_coord_stf[:], phi_coord_stf[:], cos_theta_coord_stf[:] = \
                quadratures. \
                from_sphere_s_cartesian_to_j_spherical_1d(
                    radii[s_minus_1], center_positions[j_minus_1],
                    center_positions[s_minus_1], final_length, pre_vector_t)
            a_sj[:], a_js[:] = a_0_sj_and_js_v1d(
                big_l, radii[j_minus_1], radii[s_minus_1], r_coord_stf,
                phi_coord_stf, cos_theta_coord_stf, final_length, transform,
                diagonal)
            rows_sum, columns_sum = (2 * big_l_plus_1_square * j_minus_1), \
                (2 * big_l_plus_1_square * s_minus_1)
            almost_big_a_0[rows_sum:(rows_sum + 2 * big_l_plus_1_square),
                           columns_sum:(columns_sum + 2 * big_l_plus_1_square)
                           ] = a_js
            rows_sum, columns_sum = (2 * big_l_plus_1_square * s_minus_1), \
                (2 * big_l_plus_1_square * j_minus_1)
            almost_big_a_0[rows_sum:(rows_sum + 2 * big_l_plus_1_square),
                           columns_sum:(columns_sum + 2 * big_l_plus_1_square)
                           ] = a_sj
    return almost_big_a_0


def all_cross_interactions_n_spheres_v2d(
        n: int,
        big_l: int,
        big_l_c: int,
        radii: np.ndarray,
        center_positions
) -> np.ndarray:
    """
    Returns an array with all the cross interactions for the given n spheres.
    
    Notes
    -----
    Uses the following:
    auxindexes.pes_y_kus
    quadratures.quadrature_points_sphere_shtools_version_2d
    auxindexes.diagonal_l_dense
    quadratures.from_sphere_s_cartesian_to_j_spherical_2d
    a_0_sj_and_js_v2d
    
    Parameters
    ----------
    n : int
        >= 2. Number of spheres.
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the quadrature.
    radii : np.ndarray
        Array with the radii of the spheres.
    center_positions : array_like
        List or array with the center position of the spheres

    Returns
    -------
    almost_big_a_0 : np.ndarray
        Shape ( 2 * n * (big_l + 1) ** 2, 2 * n * (big_l + 1) ** 2).
    
    See Also
    --------
    Uses the following:
    auxindexes.pes_y_kus
    quadratures.quadrature_points_sphere_shtools_version_2d
    auxindexes.diagonal_l_dense
    quadratures.from_sphere_s_cartesian_to_j_spherical_2d
    a_0_sj_and_js_v2d
    
    """
    big_l_plus_1_square = (big_l + 1) ** 2
    num = 2 * n * big_l_plus_1_square
    almost_big_a_0 = np.zeros((num, num))

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    diagonal = auxindexes.diagonal_l_dense(big_l)
    
    r_coord_stf = np.empty((quantity_theta_points, quantity_phi_points))
    phi_coord_stf = np.empty_like(r_coord_stf)
    cos_theta_coord_stf = np.empty_like(r_coord_stf)
    
    a_sj = np.zeros((2 * big_l_plus_1_square, 2 * big_l_plus_1_square))
    a_js = np.zeros_like(a_sj)
    for s in np.arange(1, n + 1):
        s_minus_1 = s - 1
        for j in np.arange(s + 1, n + 1):
            j_minus_1 = j - 1
            r_coord_stf[:], phi_coord_stf[:], cos_theta_coord_stf[:] = \
                quadratures. \
                from_sphere_s_cartesian_to_j_spherical_2d(
                    radii[s_minus_1], center_positions[j_minus_1],
                    center_positions[s_minus_1],  quantity_theta_points,
                    quantity_phi_points, pre_vector_t)
            a_sj[:], a_js[:] = a_0_sj_and_js_v2d(
                big_l, radii[j_minus_1], radii[s_minus_1], r_coord_stf,
                phi_coord_stf, cos_theta_coord_stf, weights,
                pre_vector_t[2, :, 0], quantity_theta_points,
                quantity_phi_points, pesykus, p2_plus_p_plus_q,
                p2_plus_p_minus_q, diagonal)
            rows_sum, columns_sum = (2 * big_l_plus_1_square * j_minus_1), \
                (2 * big_l_plus_1_square * s_minus_1)
            almost_big_a_0[rows_sum:(rows_sum + 2 * big_l_plus_1_square),
                           columns_sum:(columns_sum + 2 * big_l_plus_1_square)
                           ] = a_js
            rows_sum, columns_sum = (2 * big_l_plus_1_square * s_minus_1), \
                (2 * big_l_plus_1_square * j_minus_1)
            almost_big_a_0[rows_sum:(rows_sum + 2 * big_l_plus_1_square),
                           columns_sum:(columns_sum + 2 * big_l_plus_1_square)
                           ] = a_sj
    return almost_big_a_0
