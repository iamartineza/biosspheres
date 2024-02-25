import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.selfinteractions as self
import biosspheres.helmholtz.crossinteractions as cross
import biosspheres.miscella.auxindexes as auxindexes
import biosspheres.miscella.extensions as extensions
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.quadratures.sphere as quadratures


def phantom_1_plane_wave_azimuthal() -> None:
    print("\nPhantom spheres experiments for the MTF,")
    print("- One sphere, plane wave, azimuthal symmetry.")
    max_l = 20
    r = 1.3
    p_z = 5.0
    k0 = 2.0

    pi = 1.0
    a = 1.0
    num = max_l + 1

    # --- Build of phi_e.
    b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        max_l, r, p_z, k0, a, azimuthal=True
    )
    b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
        max_l, r, p_z, k0, a, azimuthal=True
    )
    b = righthands.b_vector_1_sphere_mtf(r, 1.0 / pi, b_d, b_n)

    a_0 = self.a_0j_matrix(max_l, r, k0, azimuthal=True)
    a_1 = self.a_j_matrix(max_l, r, k0, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution2 = np.linalg.solve(matrix, b)

    print("--- Checking of errors.")
    dirichlet_ex = solution2[0:num]
    neumann_ex = solution2[num : 2 * num]
    dirichlet_in = solution2[2 * num : 3 * num]
    neumann_in = solution2[3 * num : 4 * num]
    exterior_u = solution2[0 : 2 * num]
    print("---- Norm of the exterior trace (should be near zero).")
    print(np.linalg.norm(exterior_u))
    print("---- Norm of the difference between $\\gamma^{01} \\phi_e^L$")
    print("and the interior trace (absolute error):")
    print(
        np.linalg.norm(
            np.concatenate((b_d, -b_n)) - solution2[2 * num : 4 * num]
        )
    )
    print("---- Discrete Calderon errors:")
    print(
        np.linalg.norm(
            2 * np.matmul(a_0, solution2[0 : 2 * num])
            - r**2 * solution2[0 : 2 * num]
        )
    )
    print(
        np.linalg.norm(
            2 * np.matmul(a_1, solution2[2 * num : 4 * num])
            - r**2 * solution2[2 * num : 4 * num]
        )
    )
    print("---- Jump errors.")
    print("----- Dirichlet trace:")
    jump_dirichlet = np.linalg.norm(dirichlet_ex - dirichlet_in + b_d)
    print(jump_dirichlet)
    print("----- Neumann trace:")
    jump_neumann = np.linalg.norm((neumann_ex + b_n) + neumann_in)
    print(jump_neumann)
    print("----- Total jump error:")
    print(np.sqrt(jump_dirichlet**2 + jump_neumann**2))

    print("--- For plotting the convergence when the degree is increasing.")
    solutions = np.zeros((4 * num, max_l), dtype=np.complex128)
    errores = np.zeros((4 * num, max_l), dtype=np.complex128)
    for el in range(0, max_l):
        now_num = el + 1
        b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
            el, r, p_z, k0, a, azimuthal=True
        )
        b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
            el, r, p_z, k0, a, azimuthal=True
        )
        b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
        a_0 = self.a_0j_matrix(el, r, k0, azimuthal=True)
        a_1 = self.a_j_matrix(el, r, k0, azimuthal=True)
        matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
        solution = np.linalg.solve(matrix, b)
        solutions[0:now_num, el] = solution[0:now_num]
        solutions[num : num + now_num, el] = solution[now_num : 2 * now_num]
        solutions[2 * num : 2 * num + now_num, el] = solution[
            2 * now_num : 3 * now_num
        ]
        solutions[3 * num : 3 * num + now_num, el] = solution[
            3 * now_num : 4 * now_num
        ]
        errores[:, el] = solutions[:, el] - solution2
    y1 = np.linalg.norm(errores[0:num], axis=0)
    y2 = np.linalg.norm(errores[num : 2 * num], axis=0)
    y3 = np.linalg.norm(errores[2 * num : 3 * num], axis=0)
    y4 = np.linalg.norm(errores[3 * num : 4 * num], axis=0)
    plt.figure()
    plt.semilogy(
        y1,
        marker="p",
        linestyle="dashed",
        label="$||\\gamma_D^{01}u_0 - u_{D,01}^{L}||_{L^2(\\Gamma_1)}$",
    )
    plt.semilogy(
        y2,
        marker="*",
        linestyle="dashed",
        label="$||\\gamma_N^{01}u_0 - u_{N,01}^{L}||_{L^2(\\Gamma_1)}$",
    )
    plt.semilogy(
        y3,
        marker="x",
        linestyle="dashed",
        label="$||\\gamma_D^{1}u_1 - u_{D,1}^{L}||_{L^2(\\Gamma_1)}$",
    )
    plt.semilogy(
        y4,
        marker=".",
        linestyle="dashed",
        label="$||\\gamma_N^{1}u_1 - u_{N,1}^{L}||_{L^2(\\Gamma_1)}$",
    )
    plt.xlabel("$L$")
    plt.ylabel("Error")
    plt.legend()
    print("--- Pictures of the phantom sphere.")

    big_l_c = 100
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(max_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        max_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )
    eles = np.arange(0, max_l + 1)
    l_square_plus_l = (eles + 1) * eles

    vector = pre_vector * r

    surface_field_d0 = np.real(
        np.sum(
            solution2[0:num, np.newaxis, np.newaxis]
            * spherical_harmonics[l_square_plus_l, :, :],
            axis=0,
        )
    )
    surface_field_d0_max = np.max(np.abs(surface_field_d0))
    surface_field_d0_min = -surface_field_d0_max
    surface_field_d0 = (surface_field_d0 - surface_field_d0_min) / (
        surface_field_d0_max - surface_field_d0_min
    )
    surface_field_n0 = np.real(
        np.sum(
            solution2[num : 2 * num, np.newaxis, np.newaxis]
            * spherical_harmonics[l_square_plus_l, :, :],
            axis=0,
        )
    )
    surface_field_n0_max = np.max(np.abs(surface_field_n0))
    surface_field_n0_min = -surface_field_n0_max
    surface_field_n0 = (surface_field_n0 - surface_field_n0_min) / (
        surface_field_n0_max - surface_field_n0_min
    )
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.RdBu(np.real(surface_field_d0)),
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(
                vmin=surface_field_d0_min, vmax=surface_field_d0_max
            ),
            cmap=cm.RdBu,
        ),
        ax=ax_1,
        label="[V]",
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.RdBu(np.real(surface_field_n0)),
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(
                vmin=surface_field_d0_min, vmax=surface_field_d0_max
            ),
            cmap=cm.RdBu,
        ),
        ax=ax,
        label="",
    )
    plt.show()
    pass


def mix_phantom_total_3_different_1_plane_wave():
    n = 3
    big_l = 20
    big_l_c = 2 * big_l + 5
    radii = np.asarray([1.15, 1.2, 1.3])
    center_positions = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([5.0, 0.0, 0.0]),
        np.asarray([-6.0, 0.0, 0.0]),
    ]
    sigma_e = 1.0
    sigma_i = 1.25

    k0 = 2.0
    k1 = 2.5
    kii = np.ones(n + 1) * k0
    kii[1] = k1

    p_z = 5.0
    a = 1.0

    print("- " + str(n) + " spheres.")
    print("-- Space convergence of the traces of u for given phi_e.")
    sigmas = np.ones(n + 1) * sigma_e
    sigmas[1] = sigma_i
    pii = sigmas[1 : len(sigmas)] / sigma_e

    j_l = np.empty((n, big_l + 1))
    j_lp = np.empty((n, big_l + 1))

    for i in np.arange(0, n):
        j_l[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0]
        )
        j_lp[i, :] = special.spherical_jn(
            np.arange(0, big_l + 1), radii[i] * kii[0], derivative=True
        )
        pass

    big_a_0_cross = cross.all_cross_interactions_n_spheres_from_v_2d(
        n,
        big_l,
        big_l_c,
        kii[0],
        radii,
        center_positions,
        j_l,
        j_lp,
    )
    sparse_big_a_0_self, sparse_big_a_n = self.a_0_a_n_sparse_matrices(
        n, big_l, radii, kii, azimuthal=False
    )

    x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
        n, big_l, radii, pii, azimuthal=False
    )
    matrix = mtf.mtf_n_matrix(
        big_a_0_cross, sparse_big_a_0_self, sparse_big_a_n, x_dia, x_dia_inv
    )

    mass_n_two = mass.n_two_j_blocks(big_l, radii, azimuthal=False)

    b = righthands.b_vector_n_spheres_mtf_plane_wave(
        n, big_l, center_positions, p_z, kii[0], a, radii, x_dia, mass_n_two
    )
    solution2 = np.linalg.solve(matrix, b)

    print("--- Discrete Calderon errors:")
    print(
        np.linalg.norm(
            2.0
            * (
                np.matmul(
                    big_a_0_cross, solution2[0 : 2 * n * (big_l + 1) ** 2]
                )
                + sparse_big_a_0_self.dot(
                    solution2[0 : 2 * n * (big_l + 1) ** 2]
                )
            )
            - mass_n_two * solution2[0 : 2 * n * (big_l + 1) ** 2]
        )
    )
    print(
        np.linalg.norm(
            2.0
            * sparse_big_a_n.dot(
                solution2[2 * n * (big_l + 1) ** 2 : 4 * n * (big_l + 1) ** 2]
            )
            - mass_n_two
            * solution2[2 * n * (big_l + 1) ** 2 : 4 * n * (big_l + 1) ** 2]
        )
    )

    print("--- Jump error:")
    jump_error = np.linalg.norm(
        -solution2[0 : 2 * n * (big_l + 1) ** 2] * x_dia
        + mass_n_two
        * solution2[2 * n * (big_l + 1) ** 2 : 4 * n * (big_l + 1) ** 2]
        - b[2 * n * (big_l + 1) ** 2 : 4 * n * (big_l + 1) ** 2]
    )
    print(jump_error)

    plt.figure()
    plt.plot(np.abs(solution2), marker="x")
    plt.xlabel("index")
    plt.title("Coefficients of solution")

    plt.figure()
    plt.plot(np.abs(b), marker="x")
    plt.xlabel("index")
    plt.title("Coefficients of b")

    r = radii[0]
    max_l = big_l

    b_d = harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
        max_l, r, center_positions[0][2] + p_z, kii[0], a, azimuthal=True
    )
    b_n = harmonicex.plane_wave_coefficients_neumann_expansion_0j(
        max_l, r, center_positions[0][2] + p_z, kii[0], a, azimuthal=True
    )
    b_max = righthands.b_vector_1_sphere_mtf(r, 1.0 / pii[0], b_d, b_n)

    a_0 = self.a_0j_matrix(max_l, r, kii[0], azimuthal=True)
    a_1 = self.a_j_matrix(max_l, r, kii[1], azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pii[0], a_0, a_1)
    auxiliar = np.linalg.solve(matrix, b_max)
    result_one_sphere = np.zeros(4 * (big_l + 1) ** 2, dtype=np.complex128)
    result_one_sphere[0 : (big_l + 1) ** 2] = (
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, auxiliar[0 : (big_l + 1)]
        )
    )
    result_one_sphere[(big_l + 1) ** 2 : 2 * (big_l + 1) ** 2] = (
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, auxiliar[(big_l + 1) : 2 * (big_l + 1)]
        )
    )
    result_one_sphere[2 * (big_l + 1) ** 2 : 3 * (big_l + 1) ** 2] = (
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, auxiliar[2 * (big_l + 1) : 3 * (big_l + 1)]
        )
    )
    result_one_sphere[3 * (big_l + 1) ** 2 : 4 * (big_l + 1) ** 2] = (
        extensions.azimuthal_trace_to_general_with_zeros(
            big_l, auxiliar[3 * (big_l + 1) : 4 * (big_l + 1)]
        )
    )

    analytic_error = np.linalg.norm(
        result_one_sphere
        - np.concatenate(
            (
                solution2[0 : 2 * (big_l + 1) ** 2],
                solution2[
                    2 * n * (big_l + 1) ** 2 : 2 * n * (big_l + 1) ** 2
                    + 2 * (big_l + 1) ** 2
                ],
            )
        )
    ) / np.linalg.norm(result_one_sphere)
    print("--- Analytic error of the first sphere:")
    print(analytic_error)
    print("-- Coefficients of analytic solution 1.")
    plt.figure()
    plt.plot(np.abs(result_one_sphere), marker="x")
    plt.xlabel("index")
    plt.plot(
        np.abs(
            np.concatenate(
                (
                    solution2[0 : 2 * (big_l + 1) ** 2],
                    solution2[
                        2 * n * (big_l + 1) ** 2 : 2 * n * (big_l + 1) ** 2
                        + 2 * (big_l + 1) ** 2
                    ],
                )
            )
        ),
        marker="x",
    )
    plt.show()
    pass


if __name__ == "__main__":
    mix_phantom_total_3_different_1_plane_wave()
    phantom_1_plane_wave_azimuthal()
