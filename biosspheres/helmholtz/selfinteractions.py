from typing import Callable
import numpy as np
from scipy import special
from scipy import sparse


def v_jj_azimuthal_symmetry(big_l: int, r: float, k: float) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator V_{j,j}
    with Helmholtz kernel evaluated and tested with spherical harmonics
    of order 0.

    Notes
    -----
    v[l] = < V_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = i r**4 k j_l(kr) h_l(kr)
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    j_l: spherical Bessel function.
    h_l: spherical Hanckel function of first kind.

    Notice that the expression could also be computed as:
    0.5 i r pi J_{l+0.5}(k r) H_{l+0.5}(k r)
    with J_{nu} the Bessel function and
    H_{nu} the Hanckel function of first kind.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.

    Returns
    -------
    v : np.ndarray
        with complex values. Length (big_l+1). See the section notes for
        the ordering of the array.

    """
    r_k = r * k
    eles = np.arange(0, big_l + 1)
    j_l = special.spherical_jn(eles, r_k)
    y_l = special.spherical_yn(eles, r_k)
    v = (r**3 * r_k * j_l) * (1j * (j_l + 1j * y_l))
    return v


def k_0_jj_azimuthal_symmetry(big_l: int, r: float, k: float) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator K_{j,j}^0,
    with normals from out to in of the sphere, with Helmholtz kernel
    evaluated and tested with spherical harmonics of order 0.

    Notes
    -----
    k_0_jj[l] = <K_{j,j}^0 Y_l,0 ; Y_l,0>_L^2(surface sphere radius r).
    = -1j * k**2 * r**4 * 0.5
      * (j_l(r k) * h_l_d(r k) + j_l_d(r k) * h_l(r k))
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    j_l: spherical Bessel function, and j_l_d its derivative.
    h_l: spherical Hanckel function of first kind, and j_l_d its
         derivative.

    Notice that in this specific case
    < K_{j,j}^0 Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    = < K_{j,j}^{*0} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    where K_{j,j}^{*0} has normals from out to in of the sphere, and
    with Helmholtz kernel.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.

    Returns
    -------
    k_0 : np.ndarray
        with complex values. Length (big_l+1). See the section notes for
        the ordering of the array.

    """
    r_k = r * k
    eles = np.arange(0, big_l + 1)
    j_l = special.spherical_jn(eles, r_k)
    j_l_d = special.spherical_jn(eles, r_k, derivative=True)
    h_l = j_l + 1j * special.spherical_yn(eles, r_k)
    h_l_d = j_l_d + 1j * special.spherical_yn(eles, r_k, derivative=True)
    k_0 = -(r_k**2 * r**2 * 0.5) * (1j * (j_l * h_l_d + j_l_d * h_l))
    return k_0


def k_1_jj_azimuthal_symmetry(big_l: int, r: float, k: float) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator K_{j,j},
    with normals from in to out of the sphere, with Helmholtz kernel
    evaluated and tested with spherical harmonics of order 0.

    Notes
    -----
    k_jj[l] = < K_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = 1j * k**2 * r**4 * 0.5
      * (j_l(r k) * h_l_d(r k) + j_l_d(r k) * h_l(r k))
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    j_l: spherical Bessel function, and j_l_d its derivative.
    h_l: spherical Hanckel function of first kind, and j_l_d its
         derivative.

    Notice that in this specific case
    < K_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    = < K_{j,j}^{*} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r)
    where K_{j,j}^{*} has normals from in to out of the sphere, and with
    Helmholtz kernel.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.

    Returns
    -------
    k_jj : np.ndarray
        with complex values, length (big_l+1). See the section notes for
        the ordering of the array.

    """
    k_jj = -k_0_jj_azimuthal_symmetry(big_l, r, k)
    return k_jj


def w_jj_azimuthal_symmetry(big_l: int, r: float, k: float) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator W_{j,j}
    with Helmholtz kernel evaluated and tested with spherical harmonics
    of order 0.

    Notes
    -----
    w[l] = < W_{j,j} Y_l,0 ; Y_l,0 >_L^2(surface sphere radius r).
    = -1j k**3 * r**4 * j_l_d(k r) h_l_d(k r)
    for each l such that 0 <= l <= big_l, and with
    Y_l,0: spherical harmonic degree l, order 0.
    j_l_d: derivative of the spherical Bessel function.
    h_l_d: derivative of the spherical Hanckel function of first kind.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.

    Returns
    -------
    w: np.ndarray
        with complex values, length (big_l+1). See the section notes for
        the ordering of the array.

    """
    r_k = r * k
    eles = np.arange(0, big_l + 1)
    j_l_d = special.spherical_jn(eles, r_k, derivative=True)
    h_l_d = j_l_d + 1j * special.spherical_yn(eles, r_k, derivative=True)
    w = (-(r_k**3) * r) * (j_l_d * (1j * h_l_d))
    return w


def bio_jj(
    big_l: int,
    r: float,
    k: float,
    bio_azimuthal: Callable[[int, float, float], np.ndarray],
) -> np.ndarray:
    """
    Returns a numpy array with the corresponding boundary integral
    operator from the function bio_azimuthal with Helmholtz kernel
    evaluated and tested with spherical harmonics of all orders.

    Notes
    -----
    D[l*(2l+1) + m] = <D_{j,j}Y_l,m;Y_l,m>_L^2(surface sphere radius r).
    which in this case is equal to
    = D[l*(2l+1)]
    for each l such that 0 <= l <= big_l, and with
    D: boundary integral operator corresponding to the function
       bio_azimuthal
    Y_l,m: spherical harmonic degree l, order m.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.
    bio_azimuthal : Callable[[int, float, float]
        python function for computing the operator for m = 0. Must be
        one of the functions written before this one.

    Returns
    -------
    np.ndarray
        with complex values, length (big_l+1)**2. See the section notes
        for the ordering of the array.

    """
    op = bio_azimuthal(big_l, r, k)
    l2_1 = 2 * np.arange(0, big_l + 1) + 1
    return np.repeat(op, l2_1)


def a_0j_matrix(
    big_l: int, r: float, k: float, azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array that represents the following matrix boundary
    integral operator
    A_{j,j}^0 = [ -K_{j,j}^0 , V_{j,j}^0  ]
                [  W_{j,j}^0 , K*_{j,j}^0 ]
    with Helmholtz kernel evaluated and tested with spherical harmonics
    of order 0 if azimuthal = True, or all orders if azimuthal = False.

    Each block is a diagonal matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.
    azimuthal : bool
        Default True.

    Returns
    -------
    matrix_0j : np.ndarray
        with complex values.
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

    r_k = r * k
    j_l = special.spherical_jn(eles, r_k)
    h_l = j_l + 1j * special.spherical_yn(eles, r_k)
    j_l_d = special.spherical_jn(eles, r_k, derivative=True)
    h_l_d = j_l_d + 1j * special.spherical_yn(eles, r_k, derivative=True)
    k_0 = (r_k**2 * r**2 * 0.5) * (-1j * (j_l * h_l_d + j_l_d * h_l))

    if azimuthal:
        matrix_0j = np.zeros((2 * num, 2 * num), dtype=np.complex128)
        matrix_0j[eles, eles] = -k_0
        matrix_0j[num + eles, num + eles] = k_0
        matrix_0j[eles, num + eles] = r**3 * r_k * j_l * (1j * h_l)
        matrix_0j[num + eles, eles] = -(r_k**3 * r) * (j_l_d * (1j * h_l_d))
    else:
        num = num**2
        rango = np.arange(0, num)
        matrix_0j = np.zeros((2 * num, 2 * num), dtype=np.complex128)
        matrix_0j[rango, rango] = np.repeat(-k_0, l2_1)
        matrix_0j[num + rango, num + rango] = np.repeat(k_0, l2_1)
        matrix_0j[rango, num + rango] = np.repeat(
            r**3 * r_k * j_l * (1j * h_l), l2_1
        )
        matrix_0j[num + rango, rango] = np.repeat(
            (-(r_k**3) * r) * (j_l_d * (1j * h_l_d)), l2_1
        )
    return matrix_0j


def a_j_matrix(
    big_l: int, r: float, k: float, azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array that represents the following matrix boundary
    integral operator
    A_{j,j}^0 = [ -K_{j,j} , V_{j,j}  ]
                [  W_{j,j} , K*_{j,j} ]
    with Helmholtz kernel evaluated and tested with spherical harmonics
    of order 0 if azimuthal = True, or all orders if azimuthal = False.

    Each block is a diagonal matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.
    azimuthal : bool
        Default True.

    Returns
    -------
    matrix_j : np.ndarray
        with complex values.
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

    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1

    r_k = r * k
    j_l = special.spherical_jn(eles, r_k)
    h_l = j_l + 1j * special.spherical_yn(eles, r_k)
    j_l_d = special.spherical_jn(eles, r_k, derivative=True)
    h_l_d = j_l_d + 1j * special.spherical_yn(eles, r_k, derivative=True)
    k_0 = (r_k**2 * r**2 * 0.5) * (-1j * (j_l * h_l_d + j_l_d * h_l))

    if azimuthal:
        matrix_j = np.zeros((2 * num, 2 * num), dtype=np.complex128)
        matrix_j[eles, eles] = k_0
        matrix_j[num + eles, num + eles] = -k_0
        matrix_j[eles, num + eles] = r**3 * r_k * j_l * (1j * h_l)
        matrix_j[num + eles, eles] = -(r_k**3 * r) * (j_l_d * (1j * h_l_d))
    else:
        num = num**2
        rango = np.arange(0, num)
        matrix_j = np.zeros((2 * num, 2 * num), dtype=np.complex128)
        matrix_j[rango, rango] = np.repeat(k_0, l2_1)
        matrix_j[num + rango, num + rango] = np.repeat(-k_0, l2_1)
        matrix_j[rango, num + rango] = np.repeat(
            r**3 * r_k * j_l * (1j * h_l), l2_1
        )
        matrix_j[num + rango, rango] = np.repeat(
            (-(r_k**3) * r) * (j_l_d * (1j * h_l_d)), l2_1
        )
    return matrix_j


def a_0j_linear_operator(
    big_l: int, r: float, k: float, azimuthal: bool = True
) -> sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the given by
    a_0j_matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.
    azimuthal : bool
        Default True

    Returns
    -------
    linear_operator : sparse.linalg.LinearOperator
        a scipy linear operator.

    See Also
    --------
    a_0j_matrix

    """
    if azimuthal:
        num = big_l + 1
        v_0 = v_jj_azimuthal_symmetry(big_l, r, k)
        k_0 = k_0_jj_azimuthal_symmetry(big_l, r, k)
        w_0 = w_jj_azimuthal_symmetry(big_l, r, k)
    else:
        num = (big_l + 1) ** 2

        v_0 = bio_jj(big_l, r, k, v_jj_azimuthal_symmetry)
        k_0 = bio_jj(big_l, r, k, k_0_jj_azimuthal_symmetry)
        w_0 = bio_jj(big_l, r, k, w_jj_azimuthal_symmetry)

    def operator_0j_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = -k_0 * v[0:num] + v_0 * v[num : 2 * num]
        x[num : 2 * num] = w_0 * v[0:num] + k_0 * v[num : 2 * num]
        return x

    def operator_0j_conjugate_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = (
            -np.conjugate(k_0) * v[0:num] + np.conjugate(w_0) * v[num : 2 * num]
        )
        x[num : 2 * num] = (
            np.conjugate(v_0) * v[0:num] + np.conjugate(k_0) * v[num : 2 * num]
        )
        return x

    linear_operator = sparse.linalg.LinearOperator(
        (2 * num, 2 * num),
        matvec=operator_0j_times_vector,
        matmat=operator_0j_times_vector,
        rmatvec=operator_0j_conjugate_transpose_times_vector,
        rmatmat=operator_0j_conjugate_transpose_times_vector,
    )

    return linear_operator


def a_j_linear_operator(
    big_l: int, r: float, k: float, azimuthal: bool = True
) -> sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the given by
    a_j_matrix.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k : float
        wave number.
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
    if azimuthal:
        num = big_l + 1

        v_1 = v_jj_azimuthal_symmetry(big_l, r, k)
        k_1 = k_1_jj_azimuthal_symmetry(big_l, r, k)
        w_1 = w_jj_azimuthal_symmetry(big_l, r, k)
    else:
        num = (big_l + 1) ** 2

        v_1 = bio_jj(big_l, r, k, v_jj_azimuthal_symmetry)
        k_1 = bio_jj(big_l, r, k, k_1_jj_azimuthal_symmetry)
        w_1 = bio_jj(big_l, r, k, w_jj_azimuthal_symmetry)

    def operator_j_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = -k_1 * v[0:num] + v_1 * v[num : 2 * num]
        x[num : 2 * num] = w_1 * v[0:num] + k_1 * v[num : 2 * num]
        return x

    def operator_j_conjugate_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = (
            -np.conjugate(k_1) * v[0:num] + np.conjugate(w_1) * v[num : 2 * num]
        )
        x[num : 2 * num] = (
            np.conjugate(v_1) * v[0:num] + np.conjugate(k_1) * v[num : 2 * num]
        )
        return x

    linear_operator = sparse.linalg.LinearOperator(
        (2 * num, 2 * num),
        matvec=operator_j_times_vector,
        matmat=operator_j_times_vector,
        rmatvec=operator_j_conjugate_transpose_times_vector,
        rmatmat=operator_j_conjugate_transpose_times_vector,
    )

    return linear_operator


def a_0_a_n_sparse_matrices(
    n: int,
    big_l: int,
    radii: np.ndarray,
    kii: np.ndarray,
    azimuthal: bool = False,
) -> tuple[sparse.bsr_array, sparse.bsr_array]:
    if not azimuthal:
        num = (big_l + 1) ** 2
    else:
        num = big_l + 1

    big_a_0 = np.empty((4 * n * num), dtype=np.complex128)
    big_a_n = np.empty((4 * n * num), dtype=np.complex128)
    rows_big_a_sparse = np.empty((4 * n * num), dtype=int)
    columns_big_a_sparse = np.empty((4 * n * num), dtype=int)

    rango = np.arange(0, num)

    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1

    number = 0
    if not azimuthal:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2

            big_a_0[number : (number + num)] = np.repeat(
                v_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]), l2_1
            )
            big_a_n[number : (number + num)] = np.repeat(
                v_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s]), l2_1
            )
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (1 + s_minus_1_times_2) + rango
            )
            number += num

            big_a_n[number : (number + num)] = np.repeat(
                -k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s]),
                l2_1,
            )
            big_a_0[number : (number + num)] = np.repeat(
                -k_0_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]),
                l2_1,
            )
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num

            big_a_0[number : (number + num)] = np.repeat(
                k_0_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]), l2_1
            )
            big_a_n[number : (number + num)] = np.repeat(
                k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s]), l2_1
            )
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (s_minus_1_times_2 + 1) + rango
            )
            number += num

            big_a_0[number : (number + num)] = np.repeat(
                w_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]), l2_1
            )
            big_a_n[number : (number + num)] = np.repeat(
                w_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s]), l2_1
            )
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num
            pass
    else:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2

            big_a_0[number : (number + num)] = v_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            )
            big_a_n[number : (number + num)] = v_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (1 + s_minus_1_times_2) + rango
            )
            number += num

            big_a_n[number : (number + num)] = -k_1_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            big_a_0[number : (number + num)] = -k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            )
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num

            big_a_0[number : (number + num)] = k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            )
            big_a_n[number : (number + num)] = k_1_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (s_minus_1_times_2 + 1) + rango
            )
            number += num

            big_a_0[number : (number + num)] = w_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            )
            big_a_n[number : (number + num)] = w_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num
            pass
    sparse_big_a_0 = sparse.bsr_array(
        (big_a_0, (rows_big_a_sparse, columns_big_a_sparse))
    )
    sparse_big_a_n = sparse.bsr_array(
        (big_a_n, (rows_big_a_sparse, columns_big_a_sparse))
    )
    return sparse_big_a_0, sparse_big_a_n


def reduced_a_sparse_matrix(
    n: int,
    big_l: int,
    radii: np.ndarray,
    kii: np.ndarray,
    pii: np.ndarray,
    azimuthal: bool = False,
) -> sparse.bsr_array:
    if not azimuthal:
        num = (big_l + 1) ** 2
    else:
        num = big_l + 1

    reduced_big_a = np.empty((4 * n * num), dtype=np.complex128)
    rows_big_a_sparse = np.empty((4 * n * num), dtype=int)
    columns_big_a_sparse = np.empty((4 * n * num), dtype=int)

    rango = np.arange(0, num)

    eles = np.arange(0, big_l + 1)
    l2_1 = 2 * eles + 1

    number = 0
    if not azimuthal:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2

            v = v_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]) - pii[
                s_minus_1
            ] ** (-1) * v_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            reduced_big_a[number : (number + num)] = np.repeat(v, l2_1)
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (1 + s_minus_1_times_2) + rango
            )
            number += num

            k = -k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) + k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            reduced_big_a[number : (number + num)] = np.repeat(k, l2_1)
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num

            ka = k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) - k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            reduced_big_a[number : (number + num)] = np.repeat(ka, l2_1)
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (s_minus_1_times_2 + 1) + rango
            )
            number += num

            w = w_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[0]) - pii[
                s_minus_1
            ] * w_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            reduced_big_a[number : (number + num)] = np.repeat(w, l2_1)
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num
            pass
    else:
        for s in np.arange(1, n + 1):
            s_minus_1 = s - 1
            s_minus_1_times_2 = s_minus_1 * 2

            reduced_big_a[number : (number + num)] = v_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) - pii[s_minus_1] ** (-1) * v_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (1 + s_minus_1_times_2) + rango
            )
            number += num

            reduced_big_a[number : (number + num)] = -k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) + k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            rows_big_a_sparse[number : (number + num)] = (
                rango + num * s_minus_1_times_2
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num

            reduced_big_a[number : (number + num)] = k_0_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) - k_1_jj_azimuthal_symmetry(big_l, radii[s_minus_1], kii[s])
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * (s_minus_1_times_2 + 1) + rango
            )
            number += num

            reduced_big_a[number : (number + num)] = w_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[0]
            ) - pii[s_minus_1] * w_jj_azimuthal_symmetry(
                big_l, radii[s_minus_1], kii[s]
            )
            rows_big_a_sparse[number : (number + num)] = rango + num * (
                s_minus_1_times_2 + 1
            )
            columns_big_a_sparse[number : (number + num)] = (
                num * s_minus_1_times_2 + rango
            )
            number += num
            pass
    sparse_reduced_big_a = sparse.bsr_array(
        (reduced_big_a, (rows_big_a_sparse, columns_big_a_sparse))
    )
    return sparse_reduced_big_a
