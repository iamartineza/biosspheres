"""
This module has the implementation of functions that return
- Points and weights for doing the Gauss-Legendre quadrature with a
  composite trapezoidal quadrature rule. See
    gauss_legendre_trapezoidal_2d,
    gauss_legendre_trapezoidal_1d.
- Evaluation of real and complex spherical harmonics in the points of
  the Gauss-Legendre and composite trapezoidal quadrature. See
    gauss_legendre_trapezoidal_real_sh_mapping_2d,
    gauss_legendre_trapezoidal_complex_sh_mapping_2d.
- Arrays for implementing a slow spherical harmonic transform (real or
  complex). See
    real_spherical_harmonic_transform_1d,
    complex_spherical_harmonic_transform_1d.
- Translations from a coordinate system to another. See
    from_sphere_s_cartesian_to_j_spherical_2d,
    from_sphere_s_cartesian_to_j_spherical_1d,
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d,
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d.
"""
import numpy as np
import pyshtools


def gauss_legendre_trapezoidal_2d(
        big_l_c: int
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    This function is for obtaining the quadratures points to
    approximate numerically the integral in a surface of a sphere.
    It returns the weights and vectors for the Gauss-Legendre and
    a composite trapezoidal quadrature rule.
    
    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.
    Without considering errors produced by the approximation by finite
    numbers, the quadrature must be exact for functions consisting in
    polynomials of big_l_c degree times an exponential power to (m times
    i), with |m| <= big_l_c.

    Parameters
    ----------
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.

    Returns
    -------
    quantity_theta_points : int
        how many points for the integral in theta, (big_l_c + 1).
    quantity_phi_points : int
        how many points for the integral in phi, (2 * big_l_c + 1).
    weights : np.ndarray
        of floats, with the weights for the integral quadrature in
        theta. Length (big_l_c + 1).
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, quantity_theta_points, quantity_phi_points).
    
    See Also
    --------
    gauss_legendre_trapezoidal_1d
    
    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0, 2 * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )
    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)
    
    cos_phi = np.cos(phi)
    sen_phi = np.sin(phi)
    del phi
    
    cos_theta = zeros
    sen_theta = np.sqrt(1. - np.square(cos_theta))
    
    pre_vector = np.zeros((3, quantity_theta_points, quantity_phi_points))
    for i in np.arange(0, quantity_theta_points):
        np.multiply(sen_theta[i], cos_phi, out=pre_vector[0, i, :])
        np.multiply(sen_theta[i], sen_phi, out=pre_vector[1, i, :])
        pre_vector[2, i, :] = cos_theta[i]
    
    del sen_theta
    del cos_phi
    del sen_phi
    del cos_theta
    
    return quantity_theta_points, quantity_phi_points, weights, pre_vector


def gauss_legendre_trapezoidal_1d(
        big_l_c: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    This function is for obtaining the quadratures points to
    approximate numerically the integral in a surface of a sphere.
    It returns the weights and vectors for the Gauss-Legendre and
    composite trapezoidal quadrature rule. See the shape of the returns.
    
    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.
    Without considering errors produced by the approximation by finite
    numbers, the quadrature must be exact for functions consisting in
    polynomials of big_l_c degree times an exponential power to (m times
    i), with |m| <= big_l_c.

    Parameters
    ----------
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.

    Returns
    -------
    final_length : int
        how many points for the surface integral,
        = (big_l_c + 1) * (2 * big_l_c + 1).
    total_weights : np.ndarray
        of floats, with the weights for the integral quadrature and
        length final_length.
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, final_length).
    
    See Also
    --------
    gauss_legendre_trapezoidal_2d
    
    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0, 2 * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )

    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    del phi

    cos_theta = zeros
    sin_theta = np.sqrt(1. - np.square(cos_theta))

    # First is tile (theta integral) and then is repeat (phi integral)
    total_weights = np.tile(weights, quantity_phi_points) \
        * 2 * np.pi / quantity_phi_points

    final_length = len(total_weights)

    pre_vector = np.zeros((3, final_length))

    cos_phi = np.repeat(cos_phi, quantity_theta_points)
    sin_phi = np.repeat(sin_phi, quantity_theta_points)
    sin_theta = np.tile(sin_theta, quantity_phi_points)
    cos_theta = np.tile(cos_theta, quantity_phi_points)

    np.multiply(sin_theta, cos_phi, out=pre_vector[0, :])
    np.multiply(sin_theta, sin_phi, out=pre_vector[1, :])
    pre_vector[2, :] = cos_theta[:]

    return final_length, total_weights, pre_vector


def gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l: int,
        big_l_c: int,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is for obtaining the quadratures points to
    approximate numerically the integral in a surface of a sphere, and
    it also returns the evaluation of the real spherical harmonics in
    those points.
    It returns the weights and vectors for the Gauss-Legendre and
    composite trapezoidal quadrature rule. The real spherical
    harmonics evaluated are of degree l and order m, with l <= big_l.
    See the shape of the returns.
    
    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.
    Without considering errors produced by the approximation by finite
    numbers, the quadrature must be exact for functions consisting in
    polynomials of big_l_c degree times an exponential power to (m times
    i), with |m| <= big_l_c.
    Legendre's functions are computed used pyshtools.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.
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
    quantity_theta_points : int
        how many points for the integral in theta, (big_l_c + 1).
    quantity_phi_points : int
        how many points for the integral in phi, (2 * big_l_c + 1)
    weights : np.ndarray
        of floats, with the weights for the integral quadrature in
        theta. Length (big_l_c + 1).
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, quantity_theta_points, quantity_phi_points).
    spherical_harmonics : np.ndarray
        of floats, represent the real spherical harmonics of degree and
        order l and m evaluated in the points given by pre_vector. Shape
        ((big_l + 1)**2, quantity_theta_points, quantity_phi_points)
    
    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0, 2 * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )
    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)
    
    cos_phi = np.cos(phi)
    sen_phi = np.sin(phi)
    
    cos_m_phi = np.zeros((big_l, quantity_phi_points))
    sin_m_phi = np.zeros((big_l, quantity_phi_points))
    for m in np.arange(1, big_l + 1):
        np.cos(m * phi, out=cos_m_phi[m - 1, :])
        np.sin(m * phi, out=sin_m_phi[m - 1, :])
    del phi
    
    cos_theta = zeros
    
    legendre_functions = \
        np.zeros(((big_l + 1) * (big_l + 2) // 2, quantity_theta_points))
    i_range = np.arange(0, quantity_theta_points)
    for i in i_range:
        legendre_functions[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta[i], csphase=-1, cnorm=0)
    
    spherical_harmonics = np.zeros((
        (big_l + 1)**2, quantity_theta_points, quantity_phi_points))
    eles = np.arange(0, big_l + 1)
    el_square_plus_el = eles * (eles + 1)
    el_square_plus_el_divided_by_two = el_square_plus_el // 2
    spherical_harmonics[el_square_plus_el, :, :] = \
        legendre_functions[el_square_plus_el_divided_by_two, :, np.newaxis]
    index_temp = (pesykus[:, 0] * (pesykus[:, 0] + 1)) // 2 + pesykus[:, 1]
    index_temp_m = pesykus[:, 1] - 1
    j_range = np.arange(0, quantity_phi_points)
    for i in i_range:
        for j in j_range:
            spherical_harmonics[p2_plus_p_plus_q, i, j] = np.multiply(
                legendre_functions[index_temp, i], cos_m_phi[index_temp_m, j])
            spherical_harmonics[p2_plus_p_minus_q, i, j] = np.multiply(
                legendre_functions[index_temp, i], sin_m_phi[index_temp_m, j])
    del legendre_functions
    del cos_m_phi
    del sin_m_phi
    del j_range
    
    sen_theta = np.sqrt(1. - np.square(cos_theta))
    
    pre_vector = np.zeros((3, quantity_theta_points, quantity_phi_points))
    for i in i_range:
        np.multiply(sen_theta[i], cos_phi, out=pre_vector[0, i, :])
        np.multiply(sen_theta[i], sen_phi, out=pre_vector[1, i, :])
        pre_vector[2, i, :] = cos_theta[i]
    
    del sen_theta
    del cos_phi
    del sen_phi
    del cos_theta
    
    return quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics


def gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        big_l: int,
        big_l_c: int,
        pesykus: np.ndarray,
        p2_plus_p_plus_q: np.ndarray,
        p2_plus_p_minus_q: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is for obtaining the quadratures points to
    approximate numerically the integral in a surface of a sphere, and
    it also returns the evaluation of the complex spherical harmonics in
    those points.
    It returns the weights and vectors for the Gauss-Legendre and
    composite trapezoidal quadrature rule. The complex spherical
    harmonics evaluated are of degree l and order m, with l <= big_l.
    See the shape of the returns.

    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.
    Without considering errors produced by the approximation by finite
    numbers, the quadrature must be exact for functions consisting in
    polynomials of big_l_c degree times an exponential power to (m times
    i), with |m| <= big_l_c.
    Legendre's functions are computed used pyshtools.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0. It's the parameter used to compute the points of the
        quadrature.
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
    quantity_theta_points : int
        how many points for the integral in theta, (big_l_c + 1).
    quantity_phi_points : int
        how many points for the integral in phi, (2 * big_l_c + 1)
    weights : np.ndarray
        of floats, with the weights for the integral quadrature in
        theta. Length (big_l_c + 1).
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, quantity_theta_points, quantity_phi_points).
    spherical_harmonics : np.ndarray
        of complex numbers, represent the real spherical harmonics of
        degree and order l and m evaluated in the points given by
        pre_vector. Shape
        ((big_l + 1)**2, quantity_theta_points, quantity_phi_points)

    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0, 2 * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )
    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)
    
    cos_phi = np.cos(phi)
    sen_phi = np.sin(phi)
    
    exp_pos = np.zeros((big_l, quantity_phi_points), dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi, out=exp_pos[m - 1, :])
    del phi
    exp_neg = (-1.)**np.arange(1, big_l + 1) / exp_pos
    
    cos_theta = zeros
    
    legendre_functions = \
        np.zeros(((big_l + 1) * (big_l + 2) // 2, quantity_theta_points))
    i_range = np.arange(0, quantity_theta_points)
    for i in i_range:
        legendre_functions[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta[i], csphase=-1, cnorm=1)
    
    spherical_harmonics = np.zeros(
        ((big_l + 1)**2, quantity_theta_points, quantity_phi_points),
        dtype=np.complex128
    )
    eles = np.arange(0, big_l + 1)
    el_square_plus_el = eles * (eles + 1)
    el_square_plus_el_divided_by_two = el_square_plus_el // 2
    spherical_harmonics[el_square_plus_el, :, :] = \
        legendre_functions[el_square_plus_el_divided_by_two, :, np.newaxis]
    index_temp = (pesykus[:, 0] * (pesykus[:, 0] + 1)) // 2 + pesykus[:, 1]
    index_temp_m = pesykus[:, 1] - 1
    j_range = np.arange(0, quantity_phi_points)
    for i in i_range:
        for j in j_range:
            spherical_harmonics[p2_plus_p_plus_q, i, j] = np.multiply(
                legendre_functions[index_temp, i], exp_pos[index_temp_m, j])
            spherical_harmonics[p2_plus_p_minus_q, i, j] = np.multiply(
                legendre_functions[index_temp, i], exp_neg[index_temp_m, j])
    del legendre_functions
    del j_range
    
    sen_theta = np.sqrt(1. - np.square(cos_theta))
    
    pre_vector = np.zeros((3, quantity_theta_points, quantity_phi_points))
    for i in i_range:
        np.multiply(sen_theta[i], cos_phi, out=pre_vector[0, i, :])
        np.multiply(sen_theta[i], sen_phi, out=pre_vector[1, i, :])
        pre_vector[2, i, :] = cos_theta[i]
    
    del i_range
    del sen_theta
    del cos_phi
    del sen_phi
    del cos_theta
    
    return quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics


def real_spherical_harmonic_transform_1d(
        big_l: int,
        big_l_c: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    It returns the vectors for the Gauss-Legendre and trapezoidal
    quadrature rule for computing a numerical integral in the surface
    of a sphere. It also returns a vector that can be used to calculate
    the real spherical harmonic transform on the surface of a sphere of
    radius equal to one.
    
    The use of the results of this routine is for SLOW routines, because
    the vector used for the spherical harmonic transform does not have
    any performance improvements.
    
    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.
    
    Parameters
    ----------
    big_l : int
        >= 0.
    big_l_c: int
        >= 0. It's the parameter used to compute the points of the
        quadrature.

    Returns
    -------
    final_length : int
        how many points for the surface integral,
        = (big_l_c + 1) * (2 * big_l_c + 1).
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, final_length).
    transform : np.ndarray
        of floats. Vector that can be used to calculate the real
        spherical transform. Shape ((big_l+1)**2, final_length)
    
    See Also
    --------
    gauss_legendre_trapezoidal_1d
    
    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0., 2. * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )
    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)

    cos_m_phi = np.zeros((big_l, quantity_phi_points))
    sin_m_phi = np.zeros((big_l, quantity_phi_points))
    for m in np.arange(1, big_l + 1):
        np.cos(m * phi, out=cos_m_phi[m-1, :])
        np.sin(m * phi, out=sin_m_phi[m-1, :])
    del phi

    legendre_functions = \
        np.zeros(((big_l + 1) * (big_l + 2) // 2, quantity_theta_points))
    cos_theta = zeros
    for i in np.arange(0, quantity_theta_points):
        legendre_functions[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta[i], csphase=-1, cnorm=0)

    sin_theta = np.sqrt(1. - np.square(cos_theta))

    # Help:
    # first is tile (theta integral) and then is repeat (phi integral)
    total_weights = weights * 2.*np.pi / quantity_phi_points

    final_length = quantity_theta_points * quantity_phi_points

    transform = np.zeros(((big_l+1)**2, final_length))
    for el in np.arange(0, big_l+1):
        el_square_plus_el = (el + 1) * el
        el_square_plus_el_divided_by_two = el_square_plus_el // 2
        transform[el_square_plus_el, :] = np.tile(
            legendre_functions[el_square_plus_el_divided_by_two, :]
            * total_weights, quantity_phi_points)
        for m in np.arange(1, el+1):
            temp = np.tile(
                legendre_functions[el_square_plus_el_divided_by_two + m, :]
                * total_weights, quantity_phi_points)
            transform[el_square_plus_el+m, :] = temp \
                * np.repeat(cos_m_phi[m-1, :], quantity_theta_points)
            transform[el_square_plus_el - m, :] = temp \
                * np.repeat(sin_m_phi[m - 1, :], quantity_theta_points)

    pre_vector = np.zeros((3, final_length))

    cos_phi = np.repeat(cos_m_phi[0, :], quantity_theta_points)
    sin_phi = np.repeat(sin_m_phi[0, :], quantity_theta_points)
    sin_theta = np.tile(sin_theta, quantity_phi_points)
    cos_theta = np.tile(cos_theta, quantity_phi_points)

    np.multiply(sin_theta, cos_phi, out=pre_vector[0, :])
    np.multiply(sin_theta, sin_phi, out=pre_vector[1, :])
    pre_vector[2, :] = cos_theta[:]

    return final_length, pre_vector, transform


def complex_spherical_harmonic_transform_1d(
        big_l: int,
        big_l_c: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    It returns the vectors for the Gauss-Legendre and trapezoidal
    quadrature rule for computing a numerical integral in the surface
    of a sphere. It also returns a vector that can be used to calculate
    the complex spherical harmonic transform on the surface of a sphere of
    radius equal to one.
    
    The use of the results of this routine is for SLOW routines, because
    the vector used for the spherical harmonic transform does not have
    any performance improvements.

    Notes
    -----
    Gauss-legendre quadrature in theta. This one uses the package
    pyshtools.
    Composite trapezoidal rule in phi.
    Integral on theta are (big_l_c + 1) quadrature points.
    Integral on phi are (2 * big_l_c + 1) quadrature points.

    Parameters
    ----------
    big_l : int
        >= 0.
    big_l_c: int
        >= 0. It's the parameter used to compute the points of the
        quadrature.

    Returns
    -------
    final_length : int
        how many points for the surface integral,
        = (big_l_c + 1) * (2 * big_l_c + 1).
    pre_vector : np.ndarray
        of floats. Represents the vectors of the quadrature points.
        Shape (3, final_length).
    transform : np.ndarray
        of complex numbers. Vector that can be used to calculate the
        complex spherical transform. Shape ((big_l+1)**2, final_length)

    See Also
    --------
    gauss_legendre_trapezoidal_1d

    """
    zeros, weights = pyshtools.expand.SHGLQ(big_l_c)
    phi = np.linspace(0., 2. * np.pi,
                      num=(2 * big_l_c + 1),
                      endpoint=False
                      )
    quantity_theta_points = len(zeros)
    quantity_phi_points = len(phi)
    
    cos_phi = np.repeat(np.cos(phi), quantity_theta_points)
    sin_phi = np.repeat(np.sin(phi), quantity_theta_points)
    
    exp_pos = np.zeros((big_l, quantity_phi_points), dtype=np.complex128)
    for m in np.arange(1, big_l + 1):
        np.exp(1j * m * phi, out=exp_pos[m - 1, :])
    del phi
    exp_neg = (-1.)**np.arange(1, big_l + 1)[:, np.newaxis] / exp_pos
    
    legendre_functions = \
        np.zeros(((big_l + 1) * (big_l + 2) // 2, quantity_theta_points))
    cos_theta = zeros
    for i in np.arange(0, quantity_theta_points):
        legendre_functions[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta[i], csphase=-1, cnorm=1)
    
    sin_theta = np.sqrt(1. - np.square(cos_theta))
    
    # Help:
    # first is tile (theta integral) and then is repeat (phi integral)
    total_weights = weights * 2. * np.pi / quantity_phi_points
    
    final_length = quantity_theta_points * quantity_phi_points
    
    transform = np.zeros(((big_l + 1)**2, final_length),
                         dtype=np.complex128)
    for el in np.arange(0, big_l + 1):
        el_square_plus_el = (el + 1) * el
        el_square_plus_el_divided_by_two = el_square_plus_el // 2
        transform[el_square_plus_el, :] = np.tile(
            legendre_functions[el_square_plus_el_divided_by_two, :]
            * total_weights, quantity_phi_points)
        for m in np.arange(1, el + 1):
            temp = np.tile(
                legendre_functions[el_square_plus_el_divided_by_two + m, :]
                * total_weights, quantity_phi_points)
            transform[el_square_plus_el + m, :] = temp * np.repeat(
                np.conjugate(exp_pos[m - 1, :]), quantity_theta_points)
            transform[el_square_plus_el - m, :] = temp * np.repeat(
                np.conjugate(exp_neg[m - 1, :]), quantity_theta_points)
    
    pre_vector = np.zeros((3, final_length))
    
    sin_theta = np.tile(sin_theta, quantity_phi_points)
    cos_theta = np.tile(cos_theta, quantity_phi_points)
    
    np.multiply(sin_theta, cos_phi, out=pre_vector[0, :])
    np.multiply(sin_theta, sin_phi, out=pre_vector[1, :])
    pre_vector[2, :] = cos_theta[:]
    
    return final_length, pre_vector, transform


def from_sphere_s_cartesian_to_j_spherical_2d(
        r_s: float,
        p_j: np.ndarray,
        p_s: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: int,
        pre_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given points in the cartesian coordinate system "s", this algorithm
    writes them in the spherical coordinate system "j". The "j" has its
    center in a different point than the center of the coordinate system
    "s".
    
    Notes
    -----
    Input:
        x_s = r_s sin theta cos varphi ,
        y_s = r_s sin theta sin varphi ,
        z_s = r_s cos theta  ,
    To the cartesian coordinate system j:
        x_j = r_s sin theta cos varphi + (p_s - p_j)_x ,
        y_j = r_s sin theta sin varphi + (p_s - p_j)_y ,
        z_j = r_s cos theta  + (p_s - p_j)_z ,
    To the spherical coordinate system j:
        r_j = sqrt{x^2_j + y^2_j + z^2_j} ,
        cos theta = frac{r_j}{z_j} ,
        varphi = arctan2 ( frac{y_j}{x_j} )

    Parameters
    ----------
    r_s : float
        > 0, radius of the sphere s.
    p_j : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere j.
    p_s : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere s.
    quantity_theta_points : int
        how many points in theta.
    quantity_phi_points : int
        how many points in phi.
    pre_vector : np.ndarray
        of floats. Represents the vectors. Shape
        (3, quantity_theta_points, quantity_phi_points).

    Returns
    -------
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r
        of the points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the cosine of the spherical
        coordinate theta of the points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
    
    See Also
    --------
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d
    from_sphere_s_cartesian_to_j_spherical_1d
    
    """
    # temp is an array of
    # 3 x quantity_theta_points x quantity_phi_points
    # with its components in the Cartesian coordinate system of
    # system s.
    # The first row are the x coordinates,
    # the second the y,
    # the third the z.
    temp = np.multiply(r_s, pre_vector)

    # I want to write temp in the Cartesian coordinate system of
    # system j.
    # For that, I add d_js to temp.
    d_js = p_s - p_j
    erres = np.zeros((3, quantity_theta_points, quantity_phi_points))
    erres[0, :, :] = temp[0, :, :] + d_js[0]
    erres[1, :, :] = temp[1, :, :] + d_js[1]
    erres[2, :, :] = temp[2, :, :] + d_js[2]

    del d_js

    # Now I want to have the spherical coordinates of erres.
    r_coord = np.sqrt(np.sum(np.square(erres), axis=0))
    phi_coord = np.arctan2(erres[1, :, :], erres[0, :, :])
    cos_theta_coord = erres[2, :, :] / r_coord

    return r_coord, phi_coord, cos_theta_coord


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        r_s: float,
        p_j: np.ndarray,
        p_s: np.ndarray,
        quantity_theta_points: int,
        quantity_phi_points: int,
        pre_vector: np.ndarray
) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Given points in the cartesian coordinate system "s", this algorithm
    writes them in the spherical coordinate system "j", and also gives
    the dot product between the unitary vectors of the spherical system
    obtained times the normal of the unitary sphere.
    The "j" has its center in a different point than the center of the
    coordinate system "s".
    
    Notes
    -----
    Input:
        x_s = r_s sin theta cos varphi ,
        y_s = r_s sin theta sin varphi ,
        z_s = r_s cos theta  ,
    To the cartesian coordinate system j:
        x_j = r_s sin theta cos varphi + (p_s - p_j)_x ,
        y_j = r_s sin theta sin varphi + (p_s - p_j)_y ,
        z_j = r_s cos theta  + (p_s - p_j)_z ,
    To the spherical coordinate system j:
        r_j = sqrt{x^2_j + y^2_j + z^2_j} ,
        cos theta = frac{r_j}{z_j} ,
        varphi = arctan2 ( frac{y_j}{x_j} )
    Unitary vectors:
        widehat{mathbf{e}}_r = sin theta cos varphi widehat{mathbf{e}}_x
            + sin theta sin varphi widehat{mathbf{e}}_y
            + cos theta widehat{mathbf{e}}_z
        widehat{mathbf{e}}_theta = cos theta cos varphi
            widehat{mathbf{e}}_x
            + cos theta sin varphi widehat{mathbf{e}}_y
            - sin theta widehat{mathbf{e}}_z
        widehat{mathbf{e}}_{varphi} = -sin varphi widehat{mathbf{e}}_x
            + cos varphi widehat{mathbf{e}}_y

    Parameters
    ----------
    r_s : float
        > 0, radius of the sphere s.
    p_j : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere j.
    p_s : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere s.
    quantity_theta_points : int
        how many points in theta.
    quantity_phi_points : int
        how many points in phi.
    pre_vector : np.ndarray
        of floats. Represents the vectors. Shape
        (3, quantity_theta_points, quantity_phi_points).

    Returns
    -------
    r_coord : np.ndarray
        Two dimensional array of floats with the spherical coordinate r
        of the points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
    phi_coord : np.ndarray
        Two dimensional array of floats with the phi coordinate r of the
        points in the coordinate system s. Shape equals to
        (quantity_theta_points, quantity_phi_points).
    cos_theta_coord : np.ndarray
        Two dimensional array of floats with the cosine of the spherical
        coordinate theta of the points in the coordinate system s.
        Shape equals to (quantity_theta_points, quantity_phi_points).
    er_times_n : np.ndarray
        Two dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate r in the points
        in the coordinate system s and the normal of a sphere. Shape
        equals to (quantity_theta_points, quantity_phi_points).
    etheta_times_n : np.ndarray
        Two dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate theta in the points
        in the coordinate system s and the normal of a sphere. Shape
        equals to (quantity_theta_points, quantity_phi_points).
    ephi_times_n : np.ndarray
        Two dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate phi in the points
        in the coordinate system s and the normal of a sphere. Shape
        equals to (quantity_theta_points, quantity_phi_points).
    
    See Also
    --------
    from_sphere_s_cartesian_to_j_spherical_2d
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    
    """
    # temp is an array of
    # 3 x quantity_theta_points x quantity_phi_points
    # with its components in the Cartesian coordinate system of
    # system s.
    # The first row are the x coordinates,
    # the second the y,
    # the third the z.
    temp = np.multiply(r_s, pre_vector)

    # I want to write temp in the Cartesian coordinate system of
    # system j.
    # For that, I add d_js to temp.
    d_js = p_s - p_j
    erres = np.zeros((3, quantity_theta_points, quantity_phi_points))
    erres[0, :, :] = temp[0, :, :] + d_js[0]
    erres[1, :, :] = temp[1, :, :] + d_js[1]
    erres[2, :, :] = temp[2, :, :] + d_js[2]

    del d_js

    # Now I want to have the spherical coordinates of erres.
    r_coord = np.sqrt(np.sum(np.square(erres), axis=0))
    phi_coord = np.arctan2(erres[1, :, :], erres[0, :, :])
    cos_theta_coord = erres[2, :, :] / r_coord

    er_times_n = np.sum(np.multiply(erres / r_coord, pre_vector), axis=0)

    sin_theta_coord = np.sqrt(1 - cos_theta_coord ** 2)

    cos_phi_coord = np.cos(phi_coord)
    sin_phi_coord = np.sin(phi_coord)

    etheta_coord = np.zeros((3, quantity_theta_points, quantity_phi_points))
    np.multiply(cos_theta_coord, cos_phi_coord, out=etheta_coord[0, :, :])
    np.multiply(cos_theta_coord, sin_phi_coord, out=etheta_coord[1, :, :])
    np.multiply(-1, sin_theta_coord, out=etheta_coord[2, :, :])

    etheta_times_n = np.sum(np.multiply(etheta_coord, pre_vector), axis=0)
    del etheta_coord

    ephi_coord = np.zeros((3, quantity_theta_points, quantity_phi_points))
    np.multiply(-1, sin_phi_coord, out=ephi_coord[0, :, :])
    ephi_coord[1, :, :] = cos_phi_coord[:]

    del sin_phi_coord
    del cos_phi_coord

    ephi_times_n = np.sum(np.multiply(ephi_coord, pre_vector), axis=0)

    return r_coord, phi_coord, cos_theta_coord, \
        er_times_n, etheta_times_n, ephi_times_n


def from_sphere_s_cartesian_to_j_spherical_1d(
        r_s: float,
        p_j: np.ndarray,
        p_s: np.ndarray,
        final_length: int,
        pre_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given points in the cartesian coordinate system "s", this algorithm
    writes them in the spherical coordinate system "j". The "j" has its
    center in a different point than the center of the coordinate system
    "s".
    
    This one is for a slow routine.
    
    Notes
    -----
    Input:
        x_s = r_s sin theta cos varphi ,
        y_s = r_s sin theta sin varphi ,
        z_s = r_s cos theta  ,
    To the cartesian coordinate system j:
        x_j = r_s sin theta cos varphi + (p_s - p_j)_x ,
        y_j = r_s sin theta sin varphi + (p_s - p_j)_y ,
        z_j = r_s cos theta  + (p_s - p_j)_z ,
    To the spherical coordinate system j:
        r_j = sqrt{x^2_j + y^2_j + z^2_j} ,
        cos theta = frac{r_j}{z_j} ,
        varphi = arctan2 ( frac{y_j}{x_j} )

    Parameters
    ----------
    r_s : float
        > 0, radius of the sphere s.
    p_j : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere j.
    p_s : np.ndarray
        of floats of dimension 1, length 3, represents the position
        vector of the center of the sphere s.
    final_length : int
        how many points.
    pre_vector : np.ndarray
        of floats. Represents the vectors. Shape (3, final_length).

    Returns
    -------
    r_coord : np.ndarray
        One dimensional array of floats with the spherical coordinate r
        of the points in the coordinate system s. Length equals to
        final_length.
    phi_coord : np.ndarray
        One dimensional array of floats with the phi coordinate r of the
        points in the coordinate system s. Length equals to
        final_length.
    cos_theta_coord : np.ndarray
        One dimensional array of floats with the cosine of the spherical
        coordinate theta of the points in the coordinate system s.
        Length equals to final_length.

    See Also
    --------
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d
    from_sphere_s_cartesian_to_j_spherical_2d
    
    """
    # temp is an array of
    # 3 x quantity_theta_points x quantity_phi_points
    # with its components in the Cartesian coordinate system of
    # system s.
    # The first row are the x coordinates,
    # the second the y,
    # the third the z.
    temp = np.multiply(r_s, pre_vector)

    # I want to write temp in the Cartesian coordinate system of
    # system j.
    # For that, I add d_js to temp.
    d_js = p_s - p_j
    erres = np.zeros((3, final_length))
    erres[0, :] = temp[0, :] + d_js[0]
    erres[1, :] = temp[1, :] + d_js[1]
    erres[2, :] = temp[2, :] + d_js[2]

    del d_js

    # Now I want to have the spherical coordinates of erres.
    r_coord = np.linalg.norm(erres, axis=0)
    phi_coord = np.arctan2(erres[1, :], erres[0, :])
    cos_theta_coord = erres[2, :] / r_coord
    return r_coord, phi_coord, cos_theta_coord


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        r_s: float,
        p_j: np.ndarray,
        p_s: np.ndarray,
        final_length: int,
        pre_vector: np.ndarray
) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Given points in the cartesian coordinate system "s", this algorithm
    writes them in the spherical coordinate system "j", and also gives
    the dot product between the unitary vectors of the spherical system
    obtained times the normal of the unitary sphere.
    The "j" has its center in a different point than the center of the
    coordinate system "s".
    
    Notes
    -----
    Input:
        x_s = r_s sin theta cos varphi ,
        y_s = r_s sin theta sin varphi ,
        z_s = r_s cos theta  ,
    To the cartesian coordinate system j:
        x_j = r_s sin theta cos varphi + (p_s - p_j)_x ,
        y_j = r_s sin theta sin varphi + (p_s - p_j)_y ,
        z_j = r_s cos theta  + (p_s - p_j)_z ,
    To the spherical coordinate system j:
        r_j = sqrt{x^2_j + y^2_j + z^2_j} ,
        cos theta = frac{r_j}{z_j} ,
        varphi = arctan2 ( frac{y_j}{x_j} )
    Unitary vectors:
        widehat{mathbf{e}}_r = sin theta cos varphi widehat{mathbf{e}}_x
            + sin theta sin varphi widehat{mathbf{e}}_y
            + cos theta widehat{mathbf{e}}_z
        widehat{mathbf{e}}_theta = cos theta cos varphi
            widehat{mathbf{e}}_x
            + cos theta sin varphi widehat{mathbf{e}}_y
            - sin theta widehat{mathbf{e}}_z
        widehat{mathbf{e}}_{varphi} = -sin varphi widehat{mathbf{e}}_x
            + cos varphi widehat{mathbf{e}}_y

    Parameters
    ----------
    r_s : float
        > 0, radius of the sphere s.
    p_j : np.ndarray
        of floats. Length 3, represents the
        position vector of the center of the sphere j.
    p_s : np.ndarray
        of floats. Length 3, represents the
        position vector of the center of the sphere s.
    final_length : int
        how many points.
    pre_vector : np.ndarray
        of floats. Represents the vectors. Shape (3, final_length).

    Returns
    -------
    r_coord : np.ndarray
        One dimensional array of floats with the spherical coordinate r
        of the points in the coordinate system s. Length equals to
        final_length.
    phi_coord : np.ndarray
        One dimensional array of floats with the phi coordinate r of the
        points in the coordinate system s. Length equals to
        final_length.
    cos_theta_coord : np.ndarray
        One dimensional array of floats with the cosine of the spherical
        coordinate theta of the points in the coordinate system s.
        Length equals to final_length.
    er_times_n : np.ndarray
        One dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate r in the points
        in the coordinate system s and the normal of a sphere.
        Length final_length.
    etheta_times_n : np.ndarray
        One dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate theta in the points
        in the coordinate system s and the normal of a sphere.
        Length final_length.
    ephi_times_n : np.ndarray
        One dimensional array of floats with the dot product of the
        canonical vector of the spherical coordinate phi in the points
        in the coordinate system s and the normal of a sphere.
        Length final_length.
        
    """
    # temp is an array of
    # 3 x quantity_theta_points x quantity_phi_points
    # with its components in the Cartesian coordinate system of
    # system s.
    # The first row are the x coordinates,
    # the second the y,
    # the third the z.
    temp = np.multiply(r_s, pre_vector)

    # I want to write temp in the Cartesian coordinate system of
    # system j.
    # For that, I add d_js to temp.
    d_js = p_s - p_j
    erres = np.zeros((3, final_length))
    erres[0, :] = temp[0, :] + d_js[0]
    erres[1, :] = temp[1, :] + d_js[1]
    erres[2, :] = temp[2, :] + d_js[2]

    del d_js

    # Now I want to have the spherical coordinates of erres.
    r_coord = np.linalg.norm(erres, axis=0)
    phi_coord = np.arctan2(erres[1, :], erres[0, :])
    cos_theta_coord = erres[2, :] / r_coord

    er_times_n = np.sum(np.multiply(erres / r_coord, pre_vector), axis=0)

    sin_theta_coord = np.sqrt(1 - cos_theta_coord ** 2)

    cos_phi_coord = np.cos(phi_coord)
    sin_phi_coord = np.sin(phi_coord)

    etheta_coord = np.zeros((3, final_length))
    np.multiply(cos_theta_coord, cos_phi_coord, out=etheta_coord[0, :])
    np.multiply(cos_theta_coord, sin_phi_coord, out=etheta_coord[1, :])
    np.multiply(-1, sin_theta_coord, out=etheta_coord[2, :])

    etheta_times_n = np.sum(np.multiply(etheta_coord, pre_vector), axis=0)
    del etheta_coord

    ephi_coord = np.zeros((3, final_length))
    np.multiply(-1, sin_phi_coord, out=ephi_coord[0, :])
    ephi_coord[1, :] = cos_phi_coord[:]

    del sin_phi_coord
    del cos_phi_coord

    ephi_times_n = np.sum(np.multiply(ephi_coord, pre_vector), axis=0)

    return r_coord, phi_coord, cos_theta_coord, \
        er_times_n, etheta_times_n, ephi_times_n
