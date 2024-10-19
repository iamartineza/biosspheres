from functools import lru_cache
import numpy as np
import scipy.sparse
import biosspheres.utils.validation.inputs as valin


@lru_cache(maxsize=1)
def pes_y_kus(big_l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three helping arrays.

    Parameters
    ----------
    big_l : integer
        >= 0, max degree.

    Returns
    -------
    pesykus : numpy array
        dtype int, shape ((big_l+1) * big_l // 2, 2).
    p2_plus_p_plus_q : numpy array
        dtype int, length (big_l+1) * big_l // 2.
    p2_plus_p_minus_q : numpy array
        dtype int, length (big_l+1) * big_l // 2.

    """
    # Input validation
    valin.big_l_validation(big_l, "big_l")

    pesykus = np.zeros(((big_l + 1) * big_l // 2, 2), dtype=int)
    contador = 0
    for el in np.arange(1, big_l + 1):
        for m in np.arange(1, el + 1):
            pesykus[contador, 0] = el
            pesykus[contador, 1] = m
            contador = contador + 1
            pass
        pass
    # el * (el+1), without el = 0.
    p2_plus_p = pesykus[:, 0] * (pesykus[:, 0] + 1)
    p2_plus_p_plus_q = p2_plus_p + pesykus[:, 1]
    p2_plus_p_minus_q = p2_plus_p - pesykus[:, 1]

    return pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q


@lru_cache(maxsize=1)
def eles_combination(big_l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Input validation
    valin.big_l_validation(big_l, "big_l")

    eles = np.arange(0, big_l + 1)
    el_square_plus_el = eles * (eles + 1)
    el_square_plus_el_divided_by_two = el_square_plus_el // 2

    return eles, el_square_plus_el, el_square_plus_el_divided_by_two


def rows_columns_big_a_sparse_1_sphere(
    big_l: int, azimuthal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two helping arrays

    Parameters
    ----------
    big_l : integer
        >= 0, max degree.
    azimuthal : bool
        Default False.

    Returns
    -------
    rows_big_a_sparse : numpy array
        dtype = int,
        If azimuthal = True:
            length (big_l+1)
        Else:
            length (big_l+1) ** 2.
    columns_big_a_sparse : numpy array
        If azimuthal = True:
            length (big_l+1)
        Else:
            length (big_l+1) ** 2.

    """

    # Input validation
    valin.big_l_validation(big_l, "big_l")

    if azimuthal:
        num = big_l + 1
    else:
        num = (big_l + 1) ** 2
    num_big_a = 4 * num

    rows_big_a_sparse = np.empty(num_big_a, dtype=int)
    columns_big_a_sparse = np.empty(num_big_a, dtype=int)

    rango = np.arange(0, num)

    number = 0

    s = 1
    s_minus_1_times_2 = (s - 1) * 2

    # V
    rows_big_a_sparse[number : (number + num)] = rango + num * s_minus_1_times_2
    columns_big_a_sparse[number : (number + num)] = (
        num * (1 + s_minus_1_times_2) + rango
    )
    number = number + num

    # K
    rows_big_a_sparse[number : (number + num)] = rango + num * s_minus_1_times_2
    columns_big_a_sparse[number : (number + num)] = (
        num * s_minus_1_times_2 + rango
    )
    number = number + num

    # Kast
    rows_big_a_sparse[number : (number + num)] = rango + num * (
        s_minus_1_times_2 + 1
    )
    columns_big_a_sparse[number : (number + num)] = (
        num * (s_minus_1_times_2 + 1) + rango
    )
    number = number + num

    # W
    rows_big_a_sparse[number : (number + num)] = rango + num * (
        s_minus_1_times_2 + 1
    )
    columns_big_a_sparse[number : (number + num)] = (
        num * s_minus_1_times_2 + rango
    )

    return rows_big_a_sparse, columns_big_a_sparse


def diagonal_l_sparse(big_l: int) -> scipy.sparse.dia_array:

    # Input validation
    valin.big_l_validation(big_l, "big_l")

    eles = np.arange(0, big_l + 1)
    eles_times_two_plus_one = eles * 2 + 1
    diagonal = np.repeat(eles, eles_times_two_plus_one)
    num = (big_l + 1) ** 2
    el_diagonal = scipy.sparse.dia_array(
        (diagonal, np.array([0])), shape=(num, num)
    )

    return el_diagonal


def diagonal_l_dense(big_l: int) -> np.ndarray:

    # Input validation
    valin.big_l_validation(big_l, "big_l")

    eles = np.arange(0, big_l + 1)
    eles_times_two_plus_one = eles * 2 + 1
    diagonal = np.repeat(eles, eles_times_two_plus_one)

    return diagonal


def giro_sign(big_l: int) -> np.ndarray:

    # Input validation
    valin.big_l_validation(big_l, "big_l")

    num = (big_l + 1) ** 2
    sign = np.diag((-np.ones(num)) ** (np.arange(0, num)))
    giro = np.eye(num)
    eles = np.arange(0, big_l + 1)
    l_square_plus_l = eles * (eles + 1)
    for el in np.arange(1, len(eles)):
        giro[
            l_square_plus_l[el] - el : l_square_plus_l[el] + el + 1,
            l_square_plus_l[el] - el : l_square_plus_l[el] + el + 1,
        ] = np.fliplr(
            giro[
                l_square_plus_l[el] - el : l_square_plus_l[el] + el + 1,
                l_square_plus_l[el] - el : l_square_plus_l[el] + el + 1,
            ]
        )
        pass
    return giro @ sign
