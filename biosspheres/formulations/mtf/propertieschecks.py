import numpy as np
import scipy.sparse as sparse


def calderon_and_jump_checks_n_spheres(
        big_a_0_cross: np.ndarray,
        sparse_big_a_0_self: sparse.bsr_array,
        sparse_big_a_n: sparse.bsr_array,
        x_dia: np.ndarray,
        mass_n_two: np.ndarray,
        b: np.ndarray,
        solution2: np.ndarray
):
    # Discrete Calderon errors:
    aux_index = len(big_a_0_cross[0])
    exterior_calderon = np.linalg.norm(2. * (
            np.matmul(big_a_0_cross, solution2[0:aux_index])
            + sparse_big_a_0_self.dot(solution2[0:aux_index]))
                                       - mass_n_two * solution2[0:aux_index])
    interior_calderon = np.linalg.norm(2. * sparse_big_a_n.dot(
        solution2[aux_index:2*aux_index]
    ) - mass_n_two * solution2[aux_index:2*aux_index])
    
    # Jump error
    jump_error = np.linalg.norm(
        -solution2[0:aux_index] * x_dia
        + mass_n_two * solution2[aux_index:2 * aux_index]
        - b[aux_index:2*aux_index])
    return exterior_calderon, interior_calderon, jump_error
