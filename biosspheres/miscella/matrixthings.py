import numpy as np
import scipy.sparse as sparse


def sparsify_csr_array(
        matrix: np.ndarray
) -> sparse.csc_array:
    matrix_csr = sparse.csc_array(matrix)
    matrix_csr.eliminate_zeros()
    return matrix_csr


def diagonal_matrix_from_array(
        matrix: np.ndarray
) -> sparse.dia_array:
    matrix = sparse.dia_array(
        (matrix[np.newaxis, :], 0),
        shape=(len(matrix), len(matrix)))
    return matrix
