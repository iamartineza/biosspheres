import numpy as np
import biosspheres.formulations.mtf.righthands as righthands


def phi_part_of_b_separable_in_space_time(
    space_b_classic_mtf: np.ndarray, time_function, c_m
):

    def b_phi_part(time: float) -> np.ndarray:
        return space_b_classic_mtf * time_function(time)

    return b_phi_part


def phi_part_of_b_cte_space_and_time(
    big_l: int, n: int, radii: np.ndarray, cte: float
):
    b = righthands.b_vector_n_spheres_mtf_cte_function(
        n, big_l, radii, cte, azimuthal=False
    )

    def b_phi_part(time: float) -> np.ndarray:
        return b

    return b_phi_part
