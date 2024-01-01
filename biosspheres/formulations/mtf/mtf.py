from typing import Callable
import numpy as np
from scipy import sparse
import biosspheres.helmholtz.selfinteractions as helmholtz


def x_j_diagonal(
        big_l: int,
        r: float,
        pi: np.ndarray,
        azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array with the diagonal of the matrix:
    X_j = [I ,       0      ]
          [0 , -(pi)^{-1} I ]
    with I[l*(2l+1) + m] = < I Y_l,m ; Y_l,m >_L^2(surface sphere radius r).
              = r**2
    for each l such that 0 <= l <= big_l, and with
    Y_l,m: spherical harmonic degree l, order m.

    Parameters
    ----------
    big_l : int
        >= 0
    r : float
        > 0, radius
    pi : float
        > 0, adimensional parameter.
    azimuthal : bool
        default = True

    Returns
    -------
    x_j : np.ndarray
        of floats. If azimuthal = False, its is length 2*(big_l+1), else
        it is 2*(big_l+1)**2.
    
    """
    num = big_l + 1
    if not azimuthal:
        num = num**2
    eles = np.arange(0, num)
    x_j = r**2 * np.ones((2 * num))
    x_j[num + eles] = x_j[num + eles] / -pi
    return x_j


def x_j_diagonal_inv(
        big_l: int,
        r: float,
        pi: np.ndarray,
        azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array with the diagonal of the matrix:
    X_j = [I ,       0      ]
          [0 , -(pi)^{-1} I ]
    with I[l*(2l+1) + m] = < I Y_l,m ; Y_l,m >_L^2(surface sphere radius r).
              = r**2
    for each l such that 0 <= l <= big_l, and with
    Y_l,m: spherical harmonic degree l, order m.

    Parameters
    ----------
    big_l : int
        >= 0
    r : float
        > 0, radius
    pi : float
        > 0, adimensional parameter.
    azimuthal : bool
        default = True

    Returns
    -------
    x_j : np.ndarray
        of floats. If azimuthal = False, its is length 2*(big_l+1), else
        it is 2*(big_l+1)**2.

    """
    num = big_l + 1
    if not azimuthal:
        num = num**2
    eles = np.arange(0, num)
    x_j = r**2 * np.ones((2 * num))
    x_j[num + eles] = x_j[num + eles] * -pi
    return x_j


def x_diagonal_with_its_inv(
        n: int,
        big_l: int,
        radii: float,
        pii: np.ndarray,
        azimuthal: bool = False
):
    """
    Returns a numpy array with the diagonal of the matrix:
    X

    Parameters
    ----------
    n : int
        >=1 number of spheres
    big_l : int
        >= 0
    radii : np.ndarray
        of float > 0, array with the radius
    pii : np.ndarray
        of float > 0, adimensional parameter.
    azimuthal : bool
        default = False

    Returns
    -------
    x_j : np.ndarray
        of floats. If azimuthal = False, its is length 2*(big_l+1), else
        it is 2*(big_l+1)**2.

    """
    num = big_l + 1
    if not azimuthal:
        num = num**2
    
    x_dia = np.empty((2 * n * num))
    x_inv = np.empty_like(x_dia)
    
    for s in np.arange(1, n + 1):
        s_minus_1 = s - 1
        s_minus_1_times_2 = s_minus_1 * 2
        
        x_dia[(s_minus_1_times_2 * num):((s_minus_1_times_2 + 1) * num)] = \
            radii[s_minus_1]**2
        x_inv[(s_minus_1_times_2 * num):((s_minus_1_times_2 + 1) * num)] = \
            x_dia[(s_minus_1_times_2 * num):((s_minus_1_times_2 + 1) * num)]
        
        x_dia[((s_minus_1_times_2 + 1) * num):(s * 2 * num)] = (
                -radii[s_minus_1]**2 / pii[s_minus_1])
        x_inv[((s_minus_1_times_2 + 1) * num):(s * 2 * num)] = (
                -radii[s_minus_1]**2 * pii[s_minus_1])
    return x_dia, x_inv


def mtf_1_matrix(
        r: float,
        pi: float,
        a_0j: np.ndarray,
        a_j: np.ndarray
) -> np.ndarray:
    """
    Returns the matrix
    M = [2 A_{1,1}^0 , -X^{-1}  ]
        [     -X     , 2A_{1,1} ]
      = [ -2K_0 ,    2V_0    ,   -I  ,     0    ]
        [  2W_0 ,    2K*_0   ,    0  ,   pi I   ]
        [  -I   ,      0     , -2K_1 ,   2V_1   ]
        [   0   ,  pi^(-1) I ,  2W_1 ,   2K*_1  ]
    All blocks are diagonal matrices.

    Parameters
    ----------
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.
    a_0j: np.ndarray
        A_{1,1}^0
    a_j: np.ndarray
        A_{1,1}

    Returns
    -------
    mtf_matrix : np.ndarray
        Same type than a_0j. Shape 2 * Shape(a_0j)

    """
    num = len(a_0j[0, :]) // 2
    eles = np.arange(0, num)
    
    mtf_matrix = np.zeros((4 * num, 4 * num), dtype=type(a_0j[0, 0]))
    mtf_matrix[eles, eles] = 2. * a_0j[eles, eles]
    mtf_matrix[num + eles, num + eles] = 2. * a_0j[num + eles, num + eles]
    mtf_matrix[eles, num + eles] = 2. * a_0j[eles, num + eles]
    mtf_matrix[num + eles, eles] = 2. * a_0j[num + eles, eles]
    
    mtf_matrix[2 * num + eles, 2 * num + eles] = 2. * a_j[eles, eles]
    mtf_matrix[3 * num + eles, 3 * num + eles] = (2. *
                                                  a_j[num + eles, num + eles])
    mtf_matrix[2 * num + eles, 3 * num + eles] = 2. * a_j[eles, num + eles]
    mtf_matrix[3 * num + eles, 2 * num + eles] = 2. * a_j[num + eles, eles]
    
    mtf_matrix[eles, 2 * num + eles] = -r**2.
    mtf_matrix[2 * num + eles, eles] = -r**2.
    
    mtf_matrix[num + eles, 3 * num + eles] = pi * r**2
    mtf_matrix[3 * num + eles, num + eles] = r**2 / pi
    
    return mtf_matrix


def mtf_1_linear_operator(
        a_0j: sparse.linalg.LinearOperator,
        a_j: sparse.linalg.LinearOperator,
        x_j: np.ndarray,
        x_j_inv: np.ndarray
) -> sparse.linalg.LinearOperator:
    """
        Returns a scipy linear operator, it is equivalent to the matrix
        M = [2 A_{1,1}^0 , -X^{-1}  ]
            [     -X     , 2A_{1,1} ]
          = [ -2K_0 ,    2V_0    ,   -I  ,     0    ]
            [  2W_0 ,    2K*_0   ,    0  ,   pi I   ]
            [  -I   ,      0     , -2K_1 ,   2V_1   ]
            [   0   ,  pi^(-1) I ,  2W_1 ,   2K*_1  ]
        All blocks are diagonal matrices.

        Parameters
        ----------
        a_0j : sparse.linalg.LinearOperator,
        a_j : sparse.linalg.LinearOperator,
        x_j : np.ndarray
        x_j_inv : np.ndarray

        Returns
        -------
        linear_operator : sparse.linalg.LinearOperator
            a scipy linear operator.

        See Also
        --------
        mtf_1_matrix

        """
    
    num = len(x_j) // 2
    
    def block_matrix_times_vector(v) -> np.ndarray:
        x = np.empty_like(v)
        x[0:2 * num] = 2. * a_0j.matvec(v[0:2*num]) - x_j_inv * v[2*num:4*num]
        x[2 * num:4 * num] = (-x_j * v[0:2*num]
                              + 2. * a_j.matvec(v[2*num:4*num]))
        return x
    
    def block_matrix_transpose_times_vector(v) -> np.ndarray:
        x = np.empty_like(v)
        x[0:2 * num] = (2. * a_0j.rmatvec(v[0:2 * num])
                        - x_j * v[2 * num:4 * num])
        x[2 * num:4 * num] = (-x_j_inv * v[0:2 * num]
                              + 2. * a_j.rmatvec(v[2*num:4*num]))
        return x
    
    linear_operator = sparse.linalg.LinearOperator(
        (4 * num, 4 * num),
        matvec=block_matrix_times_vector,
        matmat=block_matrix_times_vector,
        rmatvec=block_matrix_transpose_times_vector,
        rmatmat=block_matrix_transpose_times_vector,
        dtype=a_0j.dtype)
    
    return linear_operator


def mtf_n_matrix(
        big_a_0_cross: np.ndarray,
        sparse_big_a_0_self: sparse.bsr_array,
        sparse_big_a_n: sparse.bsr_array,
        x_dia: np.ndarray,
        x_dia_inv: np.ndarray
) -> np.ndarray:
    num = len(big_a_0_cross[0, :]) // 2
    mtf_matrix = np.zeros((4 * num, 4 * num), dtype=big_a_0_cross.dtype)
    mtf_matrix[0:2*num, 0:2*num] = 2. * big_a_0_cross + (
            2. * sparse_big_a_0_self).toarray()
    mtf_matrix[2 * num:4 * num, 2 * num:4 * num] = (
            2. * sparse_big_a_n).toarray()
    rango = np.arange(0, 2 * num)
    mtf_matrix[rango, rango + 2 * num] = -x_dia_inv[rango]
    mtf_matrix[rango + 2 * num, rango] = -x_dia[rango]

    return mtf_matrix


def mtf_n_linear_operator_v1(
        big_a_0_cross: np.ndarray,
        sparse_big_a_0_self: sparse.bsr_array,
        sparse_big_a_n: sparse.bsr_array,
        x_dia: np.ndarray,
        x_dia_inv: np.ndarray
) -> sparse.linalg.LinearOperator:
    num = len(x_dia) // 2
    
    def block_matrix_times_vector(v) -> np.ndarray:
        x = np.empty_like(v)
        x[0:2 * num] = (2. * (big_a_0_cross.matmul(v[0:2*num])
                              + sparse_big_a_0_self.dot(v[0:2*num]))
                        - x_dia_inv * v[2*num:4*num])
        x[2 * num:4 * num] = (-x_dia * v[0:2*num]
                              + 2. * sparse_big_a_n.dot(v[2*num:4*num]))
        return x
    
    linear_operator = sparse.linalg.LinearOperator(
        (4 * num, 4 * num),
        matvec=block_matrix_times_vector,
        dtype=big_a_0_cross.dtype)
    
    return linear_operator


def mtf_1_reduced_matrix_laplace(
        pi: float,
        a_0: np.ndarray,
) -> np.ndarray:
    """
    Returns the matrix M_red

    Notes
    -----
    In this case it holds:
    M_red = 2 [2 K_0        , (1 + pi^{-1}) V_0]
              [(1 + pi) W_0 ,      2 K_0^{*}   ]


    Parameters
    ----------
    pi : float
        > 0, adimensional parameter.
    a_0 : np.ndarray
        A_{1,1}^0

    Returns
    -------
    matrix : np.ndarray
        of floats. Same shape than a_0.

    """
    num = len(a_0[0, :]) // 2
    eles = np.arange(0, num)
    
    matrix = np.zeros((2 * num, 2 * num))
    matrix[eles, eles] = 4. * a_0[eles, eles]
    matrix[num + eles, num + eles] = 4 * a_0[num + eles, num + eles]
    matrix[eles, num + eles] = 2. * (1. + pi**(-1)) * a_0[eles, num + eles]
    matrix[num + eles, eles] = 2. * (1. + pi) * a_0[num + eles, eles]
    
    return matrix


def mtf_1_reduced_linear_operator(
        big_l: int,
        r: float,
        pi: float,
        azimuthal: bool = True
) -> sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the matrix
    M_red

    Notes
    -----
    In this case it holds:
    M_red = 2 [2 K_0        , (1 + pi^{-1}) V_0]
              [(1 + pi) W_0 ,      2 K_0^{*}   ]

    Parameters
    ----------
    azimuthal
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    pi : float
        > 0, adimensional parameter.

    Returns
    -------
    linear_operator : sparse.linalg.LinearOperator
        a scipy linear operator.

    See Also
    --------
    mtf_1_reduced_matrix
    a_0j_matrix
    a_j_matrix
    biosspheres.miscella.frommtf.x_j_diagonal

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
    
    pi_inv = pi**(-1)
    
    def matrix_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = 2. * r**2 * (
                -v[0:num] + r * (1. + pi_inv) * v[num:2 * num]) / l2_1
        x[num:2 * num] = 2. * r**2 * (
                (1. + pi) / r * (eles_1_eles * v[0:num])
                + v[num:2 * num]) / l2_1
        return x
    
    def matrix_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v))
        x[0:num] = 2. * r**2 * (
                -v[0:num] + (1. + pi) / r
                * (eles_1_eles * v[num:2 * num])) / l2_1
        x[num:2 * num] = 2. * r**2 * (
                r * (1. + pi_inv) * v[0:num] + v[num:2 * num]
        ) / l2_1
        return x
    
    linear_operator = \
        sparse.linalg.LinearOperator(
            (2 * num, 2 * num),
            matvec=matrix_times_vector,
            matmat=matrix_times_vector,
            rmatvec=matrix_transpose_times_vector,
            rmatmat=matrix_transpose_times_vector)
    return linear_operator


def mtf_1_reduced_matrix_helmholtz(
        big_l: int,
        r: float,
        k0: float,
        k1: float,
        pi: float,
        azimuthal: bool
) -> np.ndarray:
    """
    Returns the matrix
    M_red = 2 (A_{1,1}^0 - X^{-1} A_{1,1} X)

    Notes
    -----
    In this case it holds:
    M_red = 2 [ -(K_0 - K_1) , V_0 + pi^{-1} V_1]
              [ W_0 + pi W_1 ,     K_0 - K_1    ]


    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r: float
        > 0, radius.
    k0: float
    k1: float
    pi: float
        > 0, adimensional parameter.
    azimuthal: bool

    Returns
    -------
    matrix : np.ndarray
        with complex values. Shape (2*(big_l+1)**2, 2*(big_l+1)**2)

    """
    if azimuthal:
        num = big_l + 1
        v_0 = helmholtz.v_jj_azimuthal_symmetry(big_l, r, k0)
        k_0 = helmholtz.k_0_jj_azimuthal_symmetry(big_l, r, k0)
        w_0 = helmholtz.w_jj_azimuthal_symmetry(big_l, r, k0)
        v_1 = helmholtz.v_jj_azimuthal_symmetry(big_l, r, k1)
        k_1 = helmholtz.k_1_jj_azimuthal_symmetry(big_l, r, k1)
        w_1 = helmholtz.w_jj_azimuthal_symmetry(big_l, r, k1)
    else:
        num = (big_l + 1)**2
        v_0 = helmholtz.bio_jj(big_l, r, k0, helmholtz.v_jj_azimuthal_symmetry)
        k_0 = helmholtz.bio_jj(big_l, r, k0,
                               helmholtz.k_0_jj_azimuthal_symmetry)
        w_0 = helmholtz.bio_jj(big_l, r, k0, helmholtz.w_jj_azimuthal_symmetry)
        v_1 = helmholtz.bio_jj(big_l, r, k1, helmholtz.v_jj_azimuthal_symmetry)
        k_1 = helmholtz.bio_jj(big_l, r, k1,
                               helmholtz.k_1_jj_azimuthal_symmetry)
        w_1 = helmholtz.bio_jj(big_l, r, k1, helmholtz.w_jj_azimuthal_symmetry)

    eles_rep = np.arange(0, num)
    
    matrix = np.zeros((2 * num, 2 * num), dtype=np.complex128)
    matrix[eles_rep, eles_rep] = -2. * (k_0 - k_1)
    matrix[num + eles_rep, num + eles_rep] = -matrix[eles_rep, eles_rep]
    matrix[eles_rep, num + eles_rep] = 2. * (v_0 + pi**(-1) * v_1)
    matrix[num + eles_rep, eles_rep] = 2. * (w_0 + pi * w_1)
    
    return matrix


def mtf_1_reduced_linear_operator_helmholtz(
        big_l: int,
        r: float,
        k0: float,
        k1: float,
        pi: float,
        azimuthal: bool
) -> sparse.linalg.LinearOperator:
    """
    Returns a scipy linear operator equivalent to the matrix
    M_red = 2 (A_{1,1}^0 - X^{-1} A_{1,1} X)
    
    Notes
    -----
    In this case it holds:
    M_red = 2 [ -(K_0 - K_1) , V_0 + pi^{-1} V_1]
              [ W_0 + pi W_1 ,     K_0 - K_1    ]

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.
    k0 : float
    k1 : float
    pi : float
        > 0, adimensional parameter.
    azimuthal : bool

    Returns
    -------
    linear_operator : sparse.linalg.LinearOperator
        a scipy linear operator.

    See Also
    --------
    mtf_1_reduced_matrix

    """
    if azimuthal:
        num = big_l + 1
        v_0 = helmholtz.v_jj_azimuthal_symmetry(big_l, r, k0)
        k_0 = helmholtz.k_0_jj_azimuthal_symmetry(big_l, r, k0)
        w_0 = helmholtz.w_jj_azimuthal_symmetry(big_l, r, k0)
        v_1 = helmholtz.v_jj_azimuthal_symmetry(big_l, r, k1)
        k_1 = helmholtz.k_1_jj_azimuthal_symmetry(big_l, r, k1)
        w_1 = helmholtz.w_jj_azimuthal_symmetry(big_l, r, k1)
    else:
        num = (big_l + 1)**2
        v_0 = helmholtz.bio_jj(big_l, r, k0, helmholtz.v_jj_azimuthal_symmetry)
        k_0 = helmholtz.bio_jj(big_l, r, k0,
                               helmholtz.k_0_jj_azimuthal_symmetry)
        w_0 = helmholtz.bio_jj(big_l, r, k0, helmholtz.w_jj_azimuthal_symmetry)
        v_1 = helmholtz.bio_jj(big_l, r, k1, helmholtz.v_jj_azimuthal_symmetry)
        k_1 = helmholtz.bio_jj(big_l, r, k1,
                               helmholtz.k_1_jj_azimuthal_symmetry)
        w_1 = helmholtz.bio_jj(big_l, r, k1, helmholtz.w_jj_azimuthal_symmetry)
    
    k_0__k_1 = k_0 - k_1
    del k_0, k_1
    v_0__v_1 = v_0 + pi**(-1) * v_1
    del v_0, v_1
    w_0__w_1 = w_0 + pi * w_1
    del w_0, w_1
    
    def matrix_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = 2. * (-k_0__k_1 * v[0:num] + v_0__v_1 * v[num:2 * num])
        x[num:2 * num] = 2. * (w_0__w_1 * v[0:num] + k_0__k_1 * v[num:2 * num])
        return x
    
    def matrix_transpose_times_vector(v) -> np.ndarray:
        x = np.empty(np.shape(v), dtype=np.complex128)
        x[0:num] = 2. * (-np.conjugate(k_0__k_1) * v[0:num]
                         + np.conjugate(w_0__w_1) * v[num:2 * num])
        x[num:2 * num] = 2. * (np.conjugate(v_0__v_1) * v[0:num]
                               + np.conjugate(k_0__k_1) * v[num:2 * num])
        return x
    
    linear_operator = \
        sparse.linalg.LinearOperator(
            (2 * num, 2 * num),
            matvec=matrix_times_vector,
            matmat=matrix_times_vector,
            rmatvec=matrix_transpose_times_vector,
            rmatmat=matrix_transpose_times_vector)
    return linear_operator
