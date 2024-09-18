import numpy as np
import biosspheres.miscella.spherearrangements as pos
import biosspheres.formulations.mtf.solvertemplates as solver


def mtf_example_notebook() -> None:
    """
    Summary of the code (without the plots) of the jupyter notebook
    mtf_example.ipynb
    """
    n = 8
    big_l = 15
    big_l_c = 55

    r = 0.875
    radii = np.ones(n) * r

    d = 1.15
    center_positions = pos.cube_vertex_positions(int(n ** (1 / 3)), r, d)

    sigma_e = 1.75
    sigma_i = 0.75
    sigmas = np.ones(n + 1) * sigma_i
    sigmas[0] = sigma_e

    p0 = np.ones(3) * -5.0
    p0

    tolerance = 10 ** (-10)

    traces_dir = solver.mtf_laplace_n_spheres_point_source_direct_solver(
        n, big_l, big_l_c, radii, center_positions, sigmas, p0
    )
    traces_in = solver.mtf_laplace_n_spheres_point_source_indirect_solver(
        n, big_l, big_l_c, radii, center_positions, sigmas, p0, tolerance
    )
    print("Difference between the solutions")
    print(np.linalg.norm(traces_dir - traces_in))
    pass


if __name__ == "__main__":
    mtf_example_notebook()
