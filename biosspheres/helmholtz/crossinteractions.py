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
        of floats. Spherical Bessel function evaluated in k0 * r_j.
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
        Array of floats with the cosine of the spherical coordinate
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
    data_v[:] = 1j * k0 * (r_j * r_s)**2 * data_v[:]
    return data_v


def v_0_sj_semi_analytic_v2d(
        big_l: int,
        k0: float,
        r_j: float,
        r_s: float,
        j_l_j: np.ndarray,
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
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator V_{s,j}^0 with
    Helmholtz kernel evaluated and tested with real spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array
    of two dimensions, given by the function
    from_sphere_s_cartesian_to_j_spherical_2d of the module
    biosspheres.quadratures.spheres.

    Notes
    -----
    data_v[p(2p + 1) + q, l(2l + 1) + m] =
        ( V_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression V_{s,j}^0 Y_{l,m,j} is analytic. A quadrature scheme
    is used to compute the surface integral corresponding to the inner
    product.
    
    It uses functions from the package pyshtools to compute the spherical
    harmonic transforms.

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
        of floats. Spherical Bessel function evaluated in k0 * r_j.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r
        of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Comes from the function
        from_sphere_s_cartesian_to_j_spherical_2d of the module
        biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function
        from_sphere_s_cartesian_to_j_spherical_2d of the module
        biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the cosine of the spherical
        coordinate theta of the quadrature points in the coordinate
        system s. Shape equals to
        (quantity_theta_points, quantity_phi_points). Comes from the
        function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta
        variable. Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta
        variable. Comes from the function
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
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes
        ordering.

    See Also
    --------
    v_0_sj_semi_analytic_v1d
    gauss_legendre_trapezoidal_shtools_2d
    biosspheres.miscella.auxindexes.pes_y_kus

    """
    argument = k0 * r_coord
    
    num = big_l + 1
    eles = np.arange(0, num)
    
    legendre_functions = np.empty((num * (big_l + 2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    h_l = np.empty((quantity_theta_points, quantity_phi_points, big_l + 1),
                   dtype=np.complex128)
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            h_l[i, j, :] = (scipy.special.spherical_jn(eles, argument[i, j])
                            + 1j * scipy.special.spherical_yn(eles,
                                                              argument[i, j]))
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=1)
    
    exp_pos = np.empty((big_l, quantity_theta_points, quantity_phi_points),
                       dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi_coord, out=exp_pos[m - 1, :])
    
    el_plus_1_square = num**2
    
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    data_v = np.empty((el_plus_1_square, el_plus_1_square),
                      dtype=np.complex128)
    
    coefficients = np.empty((2, big_l + 1, big_l + 1), dtype=np.complex128)
    temp_l = np.empty_like(h_l[:, :, 0])
    temp_l_m = np.empty_like(temp_l)
    for el in eles:
        temp_l[:] = h_l[:, :, el]
        coefficients[:] = j_l_j[el] * pyshtools.expand.SHExpandGLQC(
            temp_l *
            legendre_functions[l_times_l_plus_l_divided_by_2[el], :, :],
            weights, zeros, norm=4, csphase=-1, lmax_calc=big_l)
        data_v[p2_plus_p_plus_q, l_square_plus_l[el]] = \
            coefficients[0, pesykus[:, 0], pesykus[:, 1]]
        data_v[p2_plus_p_minus_q, l_square_plus_l[el]] = \
            coefficients[1, pesykus[:, 0], pesykus[:, 1]]
        data_v[l_square_plus_l, l_square_plus_l[el]] = \
            coefficients[0, eles, 0]
        for m in np.arange(1, el + 1):
            temp_l_m[:] = temp_l * \
                          legendre_functions[
                          l_times_l_plus_l_divided_by_2[el] + m, :]
            coefficients[:] = j_l_j[el] * pyshtools.expand.SHExpandGLQC(
                temp_l_m * exp_pos[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_v[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v[l_square_plus_l, l_square_plus_l[el] + m] = \
                coefficients[0, eles, 0]
            
            coefficients[:] = (j_l_j[el] * (-1)**m *
                               pyshtools.expand.SHExpandGLQC(
                temp_l_m / exp_pos[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l))
            data_v[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_v[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_v[l_square_plus_l, l_square_plus_l[el] - m] = \
                coefficients[0, eles, 0]
    del coefficients
    del temp_l
    del temp_l_m
    data_v[:] = 1j * k0 * r_j**2 * r_s**2 * data_v[:]
    return data_v


def v_0_js_from_v_0_sj(
        data_v_sj: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator V_{j,s}^0 with
    Helmholtz kernel evaluated and tested with spherical harmonics.
    This routine needs the numpy array corresponding to V_{s,j}^0
    (notice the change of the order of the indexes indicating the
    spheres).

    Notes
    -----
    data_v_js[p*(2p+1) + q, l*(2l+1) + m] =
        ( V_{j,s}^0 Y_{l,m,s} ; Y_{p,q,j} )_{L^2(S_j)}.
    Y_{l,m,s} : spherical harmonic degree l, order m, in the coordinate
        system s.
    S_j : surface of the sphere j.

    This computation uses the following result for this specific case:
    ( V_{j,s}^0 Y_{l,m,s} ; Y_{p,q,j} )_{L^2(S_j)}.
        = (-1)**(m+q) ( V_{s,j}^0 Y_{p,-q,j} ; Y_{l,-m,s} )_{L^2(S_s)}

    Parameters
    ----------
    data_v_sj: np.ndarray
        represents a numerical approximation of the matrix formed by the
        boundary integral operator V_{s,j}^0 with Helmholtz kernel
        evaluated and tested with spherical harmonics.

    Returns
    -------
    data_v_js: np.ndarray
        Same shape than data_v_sj. See notes for the indexes ordering.

    See Also
    --------
    v_0_sj_semi_analytic_v1d
    v_0_sj_semi_analytic_v2d

    """
    sign_array = np.diag(
        (-np.ones(len(data_v_sj[0, :])))**(np.arange(0, len(data_v_sj[0, :]))))
    giro_array = np.eye(len(data_v_sj[0, :]))
    eles = np.arange(0, int(np.sqrt(len(data_v_sj[0, :]))))
    l_square_plus_l = eles * (eles + 1)
    for el in np.arange(1, len(eles)):
        giro_array[l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1,
        l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1] = (
            np.fliplr(giro_array[
                      l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1,
                      l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1]))
    data_v_js = giro_array @ sign_array @ data_v_sj.T @ sign_array @ giro_array
    return data_v_js


def k_0_sj_semi_analytic_v1d(
        big_l: int,
        k0: float,
        r_j: float,
        r_s: float,
        j_lp_j: np.ndarray,
        r_coord: np.ndarray,
        phi_coord: np.ndarray, cos_theta_coord: np.ndarray,
        final_length: int,
        transform: np.ndarray
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^0 with
    Laplace kernel evaluated and tested with spherical harmonics.
    In this routine the quadrature points NEED to be ordered in an array
    of one dimension.
    It is a SLOW routine, because it does not use any symmetry or
    properties of the spherical harmonics.

    Notes
    -----
    data_k[p*(2p+1) + q, l*(2l+1) + m] =
        ( K_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression K_{s,j}^0 Y_l,m,j can be obtained analytically. A
    quadrature scheme is used to compute the surface integral
    corresponding to the inner product.

    In this routine the quadrature points NEED to be ordered in an array
    of one dimension.
    It is a SLOW routine, because it does not use any symmetry or
    properties of the spherical harmonics.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    k0 : float
        > 0
    r_j : float
        > 0, radius of the sphere j.
    r_s : float
        > 0, radius of the sphere s.
    j_lp_j : np.ndarray
        of floats. Derivative of the spherical Bessel function evaluated
        in k0 * r_j.
    r_coord : np.ndarray
        Array of floats with the spherical coordinate r of the
        quadrature points in the coordinate system s. Length equals to
        final_length. Can come from the function
        from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Array of floats with the phi coordinate r of the quadrature
        points in the coordinate system s. Length equals to
        final_length. Can come from the function
        from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Array of floats with the cosine of the spherical coordinate
        theta of the quadrature points in the coordinate system s.
        Lengths equal to final_length. Can come from the function
        from_sphere_s_cartesian_to_j_spherical_1d of the module
        biosspheres.quadratures.spheres.
    final_length : int
        How many points for the surface integral.
        Can come from the function spherical_harmonic_transform_1d of
        the module biosspheres.quadratures.spheres.
    transform : np.ndarray
        of floats with the evaluation of the spherical harmonics along
        with their weights in the quadrature points.
        Can come from the function real_spherical_harmonic_transform_1d
        of the module biosspheres.quadratures.spheres.

    Returns
    -------
    data_k : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes ordering.

    See Also
    --------
    k_0_sj_semi_analytic_v2d
    k_0_sj_from_v_0_sj

    """
    argument = k0 * r_coord
    
    num = big_l + 1
    eles = np.arange(0, num)
    
    legendre_functions = \
        np.empty((num * (big_l + 2) // 2, final_length))
    h_l = np.empty((final_length, big_l + 1), dtype=np.complex128)
    for i in np.arange(0, final_length):
        h_l[i, :] = (scipy.special.spherical_jn(eles, argument[i])
                     + 1j * scipy.special.spherical_yn(eles, argument[i]))
        legendre_functions[:, i] = \
            pyshtools.legendre.PlmON(big_l, cos_theta_coord[i],
                                     csphase=-1, cnorm=1)
    
    exp_pos = np.empty((big_l, final_length), dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi_coord, out=exp_pos[m - 1, :])
    
    el_plus_1_square = num**2
    data_k = np.empty((el_plus_1_square, el_plus_1_square),
                      dtype=np.complex128)
    
    eles = np.arange(0, num)
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    temp_l = np.zeros_like(transform)
    temp_l_m = np.zeros_like(temp_l)
    for el in eles:
        temp_l[:] = h_l[:, el] * transform
        np.sum(
            temp_l * legendre_functions[l_times_l_plus_l_divided_by_2[el], :],
            axis=1, out=data_k[:, l_square_plus_l[el]])
        data_k[:, l_square_plus_l[el]] = (
                j_lp_j[el] * data_k[:, l_square_plus_l[el]])
        for m in np.arange(1, el + 1):
            temp_l_m[:] = (temp_l *
                           legendre_functions[l_times_l_plus_l_divided_by_2[el]
                                              + m, :])
            np.sum(temp_l_m * exp_pos[m - 1, :], axis=1,
                   out=data_k[:, l_square_plus_l[el] + m])
            data_k[:, l_square_plus_l[el] + m] = (
                    j_lp_j[el] * data_k[:, l_square_plus_l[el] + m])
            np.sum(temp_l_m * (-1)**m / exp_pos[m - 1, :], axis=1,
                   out=data_k[:, l_square_plus_l[el] - m])
            data_k[:, l_square_plus_l[el] - m] = (
                    j_lp_j[el] * data_k[:, l_square_plus_l[el] - m])
    data_k[:] = -1j * (k0 * r_j * r_s)**2 * data_k[:]
    return data_k


def k_0_sj_semi_analytic_v2d(
        big_l: int,
        k0: float,
        r_j: float,
        r_s: float,
        j_lp_j: np.ndarray,
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
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^0 with
    Helmholtz kernel evaluated and tested with complex spherical
    harmonics.
    In this routine the quadrature points NEED to be ordered in an array
    of two dimensions, given by the function
    from_sphere_s_cartesian_to_j_spherical_2d of the module
    biosspheres.quadratures.spheres.

    Notes
    -----
    data_k[p(2p + 1) + q, l(2l + 1) + m] =
        ( K_{s,j}^0 Y_{l,m,j} ; Y_{p,q,s} )_{L^2(S_s)}.
    Y_{l,m,j} : spherical harmonic degree l, order m, in the coordinate
        system j.
    S_s : surface of the sphere s.

    The expression K_{s,j}^0 Y_l,m,j can be obtained analytically.
    A quadrature scheme is used to compute the other surface integral.
    It uses functions from the package pyshtools to compute the
    spherical harmonic transforms.

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
    j_lp_j : np.ndarray
        of floats. Derivative of the spherical Bessel function evaluated
        in k0 * r_j.
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r
        of the quadrature points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
        Comes from the function
        from_sphere_s_cartesian_to_j_spherical_2d of the module
        biosspheres.quadratures.spheres.
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        quadrature points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
        Comes from the function
        from_sphere_s_cartesian_to_j_spherical_2d of the module
        biosspheres.quadratures.spheres.
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the cosine of the spherical
        coordinate theta of the quadrature points in the coordinate
        system s. Shape equals to
        (quantity_theta_points, quantity_phi_points). Comes from the
        function from_sphere_s_cartesian_to_j_spherical_2d of
        the module biosspheres.quadratures.spheres.
    weights : np.ndarray
        of floats. Weights for the integral quadrature in the theta
        variable. Comes from the function
        gauss_legendre_trapezoidal_shtools_2d
        from the module biosspheres.quadratures.spheres.
    zeros : np.ndarray
        of floats. Zeros of the integral quadrature in the theta
        variable. Comes from the function
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
    data_k : numpy array
        Shape ((big_l+1)**2, (big_l+1)**2). See notes for the indexes
        ordering.

    See Also
    --------
    k_0_sj_semi_analytic_v1d
    k_0_sj_from_v_0_sj
    gauss_legendre_trapezoidal_shtools_2d
    biosspheres.miscella.auxindexes.pes_y_kus

    """
    argument = k0 * r_coord
    
    num = big_l + 1
    eles = np.arange(0, num)
    
    legendre_functions = np.empty((num * (big_l + 2) // 2,
                                   quantity_theta_points, quantity_phi_points))
    h_l = np.empty((quantity_theta_points, quantity_phi_points, big_l + 1),
                   dtype=np.complex128)
    j_range = np.arange(0, quantity_phi_points)
    for i in np.arange(0, quantity_theta_points):
        for j in j_range:
            h_l[i, j, :] = (scipy.special.spherical_jn(eles, argument[i, j])
                            + 1j * scipy.special.spherical_yn(eles,
                                                              argument[i, j]))
            legendre_functions[:, i, j] = \
                pyshtools.legendre.PlmON(big_l, cos_theta_coord[i, j],
                                         csphase=-1, cnorm=1)
    
    exp_pos = np.empty((big_l, quantity_theta_points, quantity_phi_points),
                       dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi_coord, out=exp_pos[m - 1, :])
    
    el_plus_1_square = num**2
    
    eles = np.arange(0, num)
    eles_plus_1 = eles + 1
    
    l_square_plus_l = eles_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    
    data_k = np.empty((el_plus_1_square, el_plus_1_square),
                      dtype=np.complex128)
    
    coefficients = np.empty((2, big_l + 1, big_l + 1), dtype=np.complex128)
    temp_l = np.zeros_like(h_l[:, :, 0])
    temp_l_m = np.zeros_like(temp_l)
    data_k[:, 0] = 0.
    for el in eles:
        temp_l[:] = h_l[:, :, el]
        coefficients[:] = j_lp_j[el] * pyshtools.expand.SHExpandGLQC(
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
                    legendre_functions[l_times_l_plus_l_divided_by_2[el] + m,
                                       :])
            coefficients[:] = j_lp_j[el] * pyshtools.expand.SHExpandGLQC(
                temp_l_m * exp_pos[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l)
            data_k[p2_plus_p_plus_q, l_square_plus_l[el] + m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_k[p2_plus_p_minus_q, l_square_plus_l[el] + m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_k[l_square_plus_l, l_square_plus_l[el] + m] = \
                coefficients[0, eles, 0]
            
            coefficients[:] = (j_lp_j[el] * (-1)**m
                               * pyshtools.expand.SHExpandGLQC(
                temp_l_m / exp_pos[m - 1, :, :], weights, zeros,
                norm=4, csphase=-1, lmax_calc=big_l))
            data_k[p2_plus_p_plus_q, l_square_plus_l[el] - m] = \
                coefficients[0, pesykus[:, 0], pesykus[:, 1]]
            data_k[p2_plus_p_minus_q, l_square_plus_l[el] - m] = \
                coefficients[1, pesykus[:, 0], pesykus[:, 1]]
            data_k[l_square_plus_l, l_square_plus_l[el] - m] = \
                coefficients[0, eles, 0]
    data_k[:] = -1j * (k0 * r_j * r_s)**2 * data_k[:]
    return data_k


def k_0_sj_from_v_0_sj(
        data_v: np.ndarray,
        k0: float,
        r_j: float
) -> np.ndarray:
    """
    Returns a numpy array that represents a numerical approximation of
    the matrix formed by the boundary integral operator K_{s,j}^0 with
    Helmholtz kernel evaluated and tested with spherical harmonics.
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
    K_{s,j}^0 Y_{l,m} = -k0 * (j_l'(k0 r_j) / j_l (k0 r_j))
                        * V_{s,j}^0 Y_{l,m}.
    With j_l' the derivative of the spherical Bessel function, and j_l
    the spherical Bessel function.
    
    It will blow up if k0 r_j is a root of any j_l.
    
    Parameters
    ----------
    data_v : np.ndarray
        that represents a numerical approximation of the matrix formed
        by the boundary integral operator V_{s,j}^0 with Helmholtz
        kernel evaluated and tested with spherical harmonics.
    k0 : float
        > 0
    r_j : float
        > 0, radius of the sphere j.

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
    big_l = int(np.sqrt(len(data_v))) - 1
    eles = np.arange(0, big_l + 1)
    j_l_1 = scipy.special.spherical_jn(eles, r_j * k0)
    j_lp_1 = scipy.special.spherical_jn(eles, r_j * k0, derivative=True)
    jeys_array = np.diag(np.repeat(-k0 * (j_lp_1 / j_l_1), 2 * eles + 1))
    data_k = data_v@jeys_array
    return data_k
