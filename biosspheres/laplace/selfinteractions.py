from typing import Callable
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg


def v_jj_azimuthal_symmetry(
        big_l: int,
        r: float
) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator V_{j,j} with
    Laplace kernel evaluated and tested with spherical harmonics of order 0.
    
    Notes
    -----
    v[l] = < V_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = r**3 / (2 l + 1)
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.

    Returns
    -------
    v : np.ndarray
        of floats, Length (big_l+1). See the section notes for the
        ordering of the array.

    """
    l2_1 = 2 * np.arange(0, big_l + 1) + 1
    v = r**3 / l2_1
    return v


def k_0_jj_azimuthal_symmetry(
        big_l: int,
        r: float
) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator K_{j,j}^0, with
    normals from out to in of the sphere, with Laplace kernel evaluated and
    tested with spherical harmonics of order 0.
    
    Notes
    -----
    k_0_jj[l] = < K_{j,j}^0 Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = r**2 / (2*(2l + 1))
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    
    Notice that in this specific case
    < K_{j,j}^0 Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    = < K_{j,j}^{*0} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    where K_{j,j}^{*0} has normals from out to in of the sphere, and with
    Laplace kernel.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.

    Returns
    -------
    k_0_jj : np.ndarray
        of floats, length (big_l+1). See the section notes for the
        ordering of the array.
    
    """
    denominator = 2 * (2 * np.arange(0, big_l + 1) + 1)
    k_0_jj = r**2 / denominator
    return k_0_jj


def k_1_jj_azimuthal_symmetry(
        big_l: int,
        r: float,
) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator K_{j,j}, with
    normals from in to out of the sphere, with Laplace kernel evaluated and
    tested with spherical harmonics of order 0.
    
    Notes
    -----
    k_jj[l] = < K_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = -r**2 / (2*(2l + 1))
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.

    Notice that in this specific case
    < K_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    = < K_{j,j}^{*} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    where K_{j,j}^{*} has normals from in to out of the sphere, and with
    Laplace kernel.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.

    Returns
    -------
    k_jj : np.ndarray
        of floats, length (big_l+1). See the section notes for the
        ordering of the array.

    """
    k_jj = - k_0_jj_azimuthal_symmetry(big_l, r)
    return k_jj


def w_jj_azimuthal_symmetry(
        big_l: int,
        r: float
) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator W_{j,j} with
    Laplace kernel evaluated and tested with spherical harmonics of order 0.
    
    Notes
    -----
    w[l] = < W_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = r * l(l+1) / (2l + 1)
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    
    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.

    Returns
    -------
    w : np.ndarray
        of floats, length (big_l+1). See the section notes for the
        ordering of the array.
    
    """
    eles = np.arange(0, big_l + 1)
    w = r * (eles * (eles + 1) / (2 * eles + 1))
    return w


def bio_jj(
        big_l: int,
        r: float,
        bio_azimuthal: Callable[[int, float], np.ndarray]
) -> np.ndarray:
    """
    Returns a numpy array with the corresponding boundary integral operator
    from the function bio_azimuthal with Laplace kernel evaluated and tested
    with spherical harmonics of all orders.
    
    Notes
    -----
    D[l*(2l+1) + m] = < D_{j,j} Y_l,m ; Y_l,m >_L^2(surface sphere radius r).
    which in this case is equal to
    = D[l*(2l+1)]
    for each l such that 0 <= l <= big_l, and with
    D: boundary integral operator corresponding to the function bio_azimuthal
    Y_l,m: spherical harmonic degree l, order m.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    bio_azimuthal : Callable[[int, float], np.ndarray]
        python function for computing the operator for m = 0. Must be one of
        the functions written before this one.

    Returns
    -------
    op: np.ndarray
        of floats, length (big_l+1)**2. See the section notes for the
        ordering of the array.

    """
    op = bio_azimuthal(big_l, r)
    l2_1 = 2 * np.arange(0, big_l + 1) + 1
    op = np.repeat(op, l2_1)
    return op


def a_0j_matrix(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array that represents the following matrix boundary
    integral operator
    A_{j,j}^0 = [ -K_{j,j}^0 , V_{j,j}^0  ]
                [  W_{j,j}^0 , K*_{j,j}^0 ]
    with Helmholtz kernel evaluated and tested with spherical harmonics of
    order 0 if azimuthal = True, or all orders if azimuthal = False.
    
    Each block is a diagonal matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    matrix_0j : np.ndarray
        of floats.
        If azimuthal = True
            Shape (2*(big_l+1), 2*(big_l+1))
        Else
            Shape (2*(big_l+1)**2, 2*(big_l+1)**2)
    
    See Also
    --------
    a_0j_linear_operator
    v_jj_azimuthal_symmetry
    k_0_jj_azimuthal_symmetry
    w_jj_azimuthal_symmetry
    bio_jj

    """
    num = big_l + 1
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    k_0 = r**2 / (2 * l2_1)
    
    if azimuthal:
        matrix_0j = np.zeros((2 * num, 2 * num))
        matrix_0j[eles, eles] = -k_0
        matrix_0j[num + eles, num + eles] = k_0
        matrix_0j[eles, num + eles] = r**3 / l2_1
        matrix_0j[num + eles, eles] = (eles * (eles + 1)) * r / l2_1
    else:
        num = num**2
        rango = np.arange(0, num)
        
        matrix_0j = np.zeros((2 * num, 2 * num))
        matrix_0j[rango, rango] = np.repeat(-k_0, l2_1)
        matrix_0j[num + rango, num + rango] = np.repeat(k_0, l2_1)
        matrix_0j[rango, num + rango] = np.repeat(r**3 / l2_1, l2_1)
        matrix_0j[num + rango, rango] = np.repeat(
            (eles * (eles + 1)) * r / l2_1, l2_1)
    
    return matrix_0j


def a_j_matrix(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array that represents the following matrix boundary
    integral operator
    A_{j,j}^0 = [ -K_{j,j} , V_{j,j}  ]
                [  W_{j,j} , K*_{j,j} ]
    with Helmholtz kernel evaluated and tested with spherical harmonics of
    order 0 if azimuthal = True, or all orders if azimuthal = False.
    
    Each block is a diagonal matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    matrix_j : np.ndarray
        of floats.
        If azimuthal = True
            Shape (2*(big_l+1), 2*(big_l+1))
        Else
            Shape (2*(big_l+1)**2, 2*(big_l+1)**2)
    
    See Also
    --------
    a_j_linear_operator
    v_jj_azimuthal_symmetry
    k_1_jj_azimuthal_symmetry
    w_jj_azimuthal_symmetry
    bio_jj

    """
    num = big_l + 1
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    k_0 = r**2 / (2 * l2_1)
    
    if azimuthal:
        matrix_j = np.zeros((2 * num, 2 * num))
        matrix_j[eles, eles] = k_0
        matrix_j[num + eles, num + eles] = -k_0
        matrix_j[eles, num + eles] = r**3 / l2_1
        matrix_j[num + eles, eles] = (eles * (eles + 1)) * r / l2_1
    else:
        num = num**2
        rango = np.arange(0, num)
        
        matrix_j = np.zeros((2 * num, 2 * num))
        matrix_j[rango, rango] = np.repeat(k_0, l2_1)
        matrix_j[num + rango, num + rango] = np.repeat(-k_0, l2_1)
        matrix_j[rango, num + rango] = np.repeat(r**3 / l2_1, l2_1)
        matrix_j[num + rango, rango] = np.repeat(
            (eles * (eles + 1)) * r / l2_1, l2_1)
    
    return matrix_j


def a_0j_linear_operator(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> scipy.sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the given by a_0j_matrix.
    
    Parameters
    ----------
    azimuthal
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    linear_operator : sparse.linalg.LinearOperator
        a scipy linear operator.

    See Also
    --------
    a_0j_matrix
    
    """
    num = big_l + 1
    eles = np.arange(0, num)
    l2_1 = 2 * eles + 1
    eles_1_eles = (eles + 1) * eles
    del eles
    
    if not azimuthal:
        eles_1_eles = np.repeat(eles_1_eles, l2_1)
        l2_1 = np.repeat(l2_1, l2_1)
        num = num**2
    
    def operator_0j_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = r**2 * (-0.5 * v[0:num]
                           + r * v[num:2 * num]) / l2_1
        x[num:2 * num] = r * (eles_1_eles * v[0:num]
                              + 0.5 * r * v[num:2 * num]) / l2_1
        return x
    
    def operator_0j_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = r * (-0.5 * r * v[0:num]
                        + eles_1_eles
                        * (r * v[num:2 * num])) / l2_1
        x[num:2 * num] = r**2 * (r * v[0:num]
                                 + 0.5 * v[num:2 * num]) / l2_1
        return x
    
    linear_operator = scipy.sparse.linalg.LinearOperator(
        (2 * num, 2 * num),
        matvec=operator_0j_times_vector,
        matmat=operator_0j_times_vector,
        rmatvec=operator_0j_transpose_times_vector,
        rmatmat=operator_0j_transpose_times_vector)
    
    return linear_operator


def a_j_linear_operator(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> scipy.sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the given by a_j_matrix.
    
    Parameters
    ----------
    azimuthal
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    linear_operator : sparse.linalg.LinearOperator
        a scipy linear operator.
    
    See Also
    --------
    a_j_matrix
    
    """
    num = big_l + 1
    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1
    eles_1_eles = (eles + 1) * eles
    del eles
    
    if not azimuthal:
        eles_1_eles = np.repeat(eles_1_eles, l2_1)
        l2_1 = np.repeat(l2_1, l2_1)
        num = num**2
    
    def operator_j_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = r**2 * (0.5 * v[0:num] + r * v[num:2 * num]) / l2_1
        x[num:2 * num] = r * (eles_1_eles * v[0:num]
                              - 0.5 * r * v[num:2 * num]) / l2_1
        return x
    
    def operator_j_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = r * (0.5 * r * v[0:num]
                        + eles_1_eles
                        * v[num:2 * num]) / l2_1
        x[num:2 * num] = r**2 * (r * v[0:num]
                                 - 0.5 * v[num:2 * num]) / l2_1
        return x
    
    linear_operator = \
        scipy.sparse.linalg.LinearOperator(
            (2 * num, 2 * num),
            matvec=operator_j_times_vector,
            matmat=operator_j_times_vector,
            rmatvec=operator_j_transpose_times_vector,
            rmatmat=operator_j_transpose_times_vector)
    
    return linear_operator


def a_0_a_n_sparse_matrices(
        n: int,
        big_l: int,
        radii: np.ndarray,
        azimuthal: bool = False
):
    if not azimuthal:
        num = (big_l + 1)**2
    else:
        num = big_l + 1
    
    big_a_0 = np.empty((4 * n * num))
    big_a_n = np.empty((4 * n * num))
    rows_big_a_sparse = np.empty((4 * n * num), dtype=int)
    columns_big_a_sparse = np.empty((4 * n * num), dtype=int)
    
    rango = np.arange(0, num)
    
    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1
    eles_1_eles = (eles + 1) * eles
    
    number = 0
    if not azimuthal:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2
            
            big_a_0[number:(number + num)] = np.repeat(
                radii[s_minus_1]**3 / l2_1, l2_1)
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * s_minus_1_times_2)
            columns_big_a_sparse[number:(number + num)] = (
                    num * (1 + s_minus_1_times_2) + rango)
            number += num
            
            big_a_n[number:(number + num)] = np.repeat(
                radii[s_minus_1]**2 / (2 * l2_1), l2_1)
            big_a_0[number:(number + num)] = -big_a_n[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * s_minus_1_times_2)
            columns_big_a_sparse[number:(number + num)] = (
                    num * s_minus_1_times_2 + rango)
            number += num
            
            big_a_0[number:(number + num)] = big_a_n[number:(number + num)]
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * (s_minus_1_times_2 + 1))
            columns_big_a_sparse[number:(number + num)] = (
                    num * (s_minus_1_times_2 + 1) + rango)
            number += num
            
            big_a_0[number:(number + num)] = np.repeat(
                radii[s_minus_1] * eles_1_eles / l2_1, l2_1)
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * (s_minus_1_times_2 + 1))
            columns_big_a_sparse[number:(number + num)] = (
                    num * s_minus_1_times_2 + rango)
            number += num
    else:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2
            
            big_a_0[number:(number + num)] = radii[s_minus_1]**3 / l2_1
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * s_minus_1_times_2)
            columns_big_a_sparse[number:(number + num)] = (
                    num * (1 + s_minus_1_times_2) + rango)
            number += num
            
            big_a_n[number:(number + num)] = radii[s_minus_1]**2 / (2 * l2_1)
            big_a_0[number:(number + num)] = -big_a_n[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * s_minus_1_times_2)
            columns_big_a_sparse[number:(number + num)] = (
                    num * s_minus_1_times_2 + rango)
            number += num
            
            big_a_0[number:(number + num)] = big_a_n[number:(number + num)]
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * (s_minus_1_times_2 + 1))
            columns_big_a_sparse[number:(number + num)] = (
                    num * (s_minus_1_times_2 + 1) + rango)
            number += num
            
            big_a_0[number:(number + num)] = (
                    radii[s_minus_1] * eles_1_eles / l2_1)
            big_a_n[number:(number + num)] = big_a_0[number:(number + num)]
            rows_big_a_sparse[number:(number + num)] = (
                    rango + num * (s_minus_1_times_2 + 1))
            columns_big_a_sparse[number:(number + num)] = (
                    num * s_minus_1_times_2 + rango)
            number += num
    sparse_big_a_0 = sparse.bsr_array(
        (big_a_0, (rows_big_a_sparse, columns_big_a_sparse)))
    sparse_big_a_n = sparse.bsr_array(
        (big_a_n, (rows_big_a_sparse, columns_big_a_sparse)))
    return sparse_big_a_0, sparse_big_a_n
