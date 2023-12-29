import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.laplace.selfinteractions as laplace
import biosspheres.laplace.drawing as draw
import biosspheres.miscella.auxindexes as auxindexes
import biosspheres.miscella.extensions as extensions
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.mathfunctions as mathfunctions
import biosspheres.quadratures.sphere as quadratures


def phantom_1_point_source_azimuthal(
        max_l: int = 50,
        r: float = 1.3,
        distance: float = 20.,
        intensity: float = 1.,
        resolution: int = 10,
        horizontal: float = 10.,
        vertical: float = 10.,
) -> None:
    print('\nPhantom spheres experiments for the MTF,')
    print('- One sphere, point source, azimuthal symmetry.')
    pi = 1.
    sigma_e = 1.
    num = max_l + 1
    
    # --- Build of phi_e.
    b_d = (harmonicex.
    point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
    point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
    
    a_0 = laplace.a_0j_matrix(max_l, r, azimuthal=True)
    a_1 = laplace.a_j_matrix(max_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution2 = np.linalg.solve(matrix, b)
    
    print('--- Checking of errors.')
    dirichlet_ex = solution2[0:num]
    neumann_ex = solution2[num:2 * num]
    dirichlet_in = solution2[2 * num:3 * num]
    neumann_in = solution2[3 * num:4 * num]
    exterior_u = solution2[0:2 * num]
    print('---- Norm of the exterior trace (should be near zero).')
    print(np.linalg.norm(exterior_u))
    print('---- Norm of the difference between $\\gamma^{01} \\phi_e^L$')
    print('and the interior trace (absolute error):')
    print(np.linalg.norm(
        np.concatenate((b_d, -b_n)) - solution2[2 * num:4 * num]))
    print('---- Discrete Calderon errors:')
    print(np.linalg.norm(2 * np.matmul(a_0, solution2[0:2 * num])
                         - r**2 * solution2[0:2 * num]))
    print(
        np.linalg.norm(2 * np.matmul(a_1, solution2[2 * num:4 * num])
                       - r**2 * solution2[2 * num:4 * num]))
    print('---- Jump errors.')
    print('----- Dirichlet trace:')
    jump_dirichlet = np.linalg.norm(dirichlet_ex - dirichlet_in + b_d)
    print(jump_dirichlet)
    print('----- Neumann trace:')
    jump_neumann = \
        np.linalg.norm((neumann_ex + b_n) + neumann_in)
    print(jump_neumann)
    print('----- Total jump error:')
    print(np.sqrt(jump_dirichlet**2 + jump_neumann**2))
    
    print('--- For plotting the convergence when the degree is increasing.')
    solutions = np.zeros((4 * num, max_l))
    errores = np.zeros((4 * num, max_l))
    for el in range(0, max_l):
        now_num = el + 1
        b_d = (harmonicex.
        point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity))
        b_n = harmonicex. \
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity)
        b = righthands.b_vector_1_sphere_mtf(r, pi, b_d, b_n)
        a_0 = laplace.a_0j_matrix(el, r, azimuthal=True)
        a_1 = laplace.a_j_matrix(el, r, azimuthal=True)
        matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
        solution = np.linalg.solve(matrix, b)
        solutions[0:now_num, el] = solution[0:now_num]
        solutions[num:num + now_num, el] = solution[now_num:2 * now_num]
        solutions[2 * num:2 * num + now_num, el] = solution[
                                                   2 * now_num:3 * now_num]
        solutions[3 * num:3 * num + now_num, el] = solution[
                                                   3 * now_num:4 * now_num]
        errores[:, el] = solutions[:, el] - solution2
    y1 = np.linalg.norm(errores[0:num], axis=0)
    y2 = np.linalg.norm(errores[num:2 * num], axis=0)
    y3 = np.linalg.norm(errores[2 * num:3 * num], axis=0)
    y4 = np.linalg.norm(errores[3 * num:4 * num], axis=0)
    plt.figure()
    plt.semilogy(
        y1, marker='p', linestyle='dashed',
        label='$||\\gamma_D^{01}u_0 - u_{D,01}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y2, marker='*', linestyle='dashed',
        label='$||\\gamma_N^{01}u_0 - u_{N,01}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y3, marker='x', linestyle='dashed',
        label='$||\\gamma_D^{1}u_1 - u_{D,1}^{L}||_{L^2(\\Gamma_1)}$')
    plt.semilogy(
        y4, marker='.', linestyle='dashed',
        label='$||\\gamma_N^{1}u_1 - u_{N,1}^{L}||_{L^2(\\Gamma_1)}$')
    plt.xlabel('$L$')
    plt.ylabel('Error')
    plt.legend()
    print('--- Pictures of the phantom sphere.')
    
    big_l_c = 100
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(max_l)
    quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics = \
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            max_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    eles = np.arange(0, max_l + 1)
    l_square_plus_l = (eles + 1) * eles
    
    vector = pre_vector * r
    
    surface_field_d0 = np.sum(
        solution2[0:num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                   l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_d0_max = np.max(surface_field_d0)
    surface_field_d0_min = np.min(surface_field_d0)
    surface_field_d0 = (surface_field_d0 - surface_field_d0_min) \
                       / (surface_field_d0_max - surface_field_d0_min)
    surface_field_n0 = np.sum(
        solution2[num:2 * num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                         l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_n0_max = np.max(surface_field_n0)
    surface_field_n0_min = np.min(surface_field_n0)
    surface_field_n0 = (surface_field_n0 - surface_field_n0_min) \
                       / (surface_field_n0_max - surface_field_n0_min)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_d0))
    ax_1.set_xlabel('$x \\ [\\mu m]$')
    ax_1.set_ylabel('$y \\ [\\mu m]$')
    ax_1.set_zlabel('$z \\ [\\mu m]$')
    ax_1.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax_1,
        label='[V]'
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_n0))
    ax.set_xlabel('$x \\ [\\mu m]$')
    ax.set_ylabel('$y \\ [\\mu m]$')
    ax.set_zlabel('$z \\ [\\mu m]$')
    ax.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax,
        label='[V $/ \\mu m$ ]'
    )
    
    def point_source(x: np.ndarray) -> float:
        return 0.
    
    center = np.asarray([0., 0., 0.])
    inter_horizontal = resolution
    inter_vertical = resolution
    
    p = np.array([0., 0., distance])
    
    r = 1.
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    temp = np.zeros_like(solution2[2 * (max_l + 1):4 * (max_l + 1)])
    temp[:] = solution2[2 * (max_l + 1):4 * (max_l + 1)]
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = 0.
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.colorbar(label='[V]')
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    
    def point_source(x: np.ndarray) -> float:
        return mathfunctions.point_source(x, p, sigma_e)
    
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = temp[:]
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    solution2[2 * (max_l + 1):4 * (max_l + 1)] = 0.
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]])
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution2, r, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               norm=colors.CenteredNorm(),
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]])
    plt.colorbar(label='[V]')
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    
    pass


def non_phantom_1_point_source_z_alignment_distance_convergence(
        max_l: int = 50,
        r: float = 1.3,
        sigma_e: float = 5.,
        sigma_i: float = 0.455,
        distance: float = 20.,
        intensity: float = 1.,
        resolution: int = 10,
        horizontal: float = 10.,
        vertical: float = 10.,
) -> None:
    print(
        '1 sphere, non phantom, z-alignment, point source external function.')
    print('Fixed distance, convergence in degree.')
    pi = sigma_i / sigma_e
    num = max_l + 1
    
    b_d = (harmonicex.
    point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b_n = (harmonicex.
    point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
        max_l, r, distance, sigma_e, intensity))
    b_max = righthands.b_vector_1_sphere_mtf(r, 1. / pi, b_d, b_n)
    
    a_0 = laplace.a_0j_matrix(max_l, r, azimuthal=True)
    a_1 = laplace.a_j_matrix(max_l, r, azimuthal=True)
    matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
    solution2 = np.linalg.solve(matrix, b_max)
    
    print('--- Checking of errors.')
    dirichlet_ex = solution2[0:num]
    neumann_ex = solution2[num:2 * num]
    dirichlet_in = solution2[2 * num:3 * num]
    neumann_in = solution2[3 * num:4 * num]
    print('---- Discrete Calderon errors:')
    print(np.linalg.norm(2 * np.matmul(a_0, solution2[0:2 * num])
                         - r**2 * solution2[0:2 * num]))
    print(
        np.linalg.norm(2 * np.matmul(a_1, solution2[2 * num:4 * num])
                       - r**2 * solution2[2 * num:4 * num]))
    print('---- Jump errors.')
    print('----- Dirichlet trace:')
    jump_dirichlet = np.linalg.norm(dirichlet_ex - dirichlet_in + b_d)
    print(jump_dirichlet)
    print('----- Neumann trace:')
    jump_neumann = \
        np.linalg.norm(sigma_e * (neumann_ex + b_n) + sigma_i * neumann_in)
    print(jump_neumann)
    print('----- Total jump error:')
    print(np.sqrt(jump_dirichlet**2 + jump_neumann**2))
    
    solutions = np.zeros((4 * num, max_l))
    errores = np.zeros((4 * num, max_l))
    for el in np.arange(0, max_l):
        now_num = el + 1
        b_d = (harmonicex.
        point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity))
        b_n = harmonicex. \
            point_source_coefficients_neumann_expansion_0j_azimuthal_symmetry(
            el, r, distance, sigma_e, intensity)
        b = righthands.b_vector_1_sphere_mtf(r, 1. / pi, b_d, b_n)
        a_0 = laplace.a_0j_matrix(el, r, azimuthal=True)
        a_1 = laplace.a_j_matrix(el, r, azimuthal=True)
        matrix = mtf.mtf_1_matrix(r, pi, a_0, a_1)
        solution = np.linalg.solve(matrix, b)
        solutions[0:now_num, el] = solution[0:now_num]
        solutions[num:num + now_num, el] = solution[now_num:2 * now_num]
        solutions[2 * num:2 * num + now_num, el] = solution[
                                                   2 * now_num:3 * now_num]
        solutions[3 * num:3 * num + now_num, el] = solution[
                                                   3 * now_num:4 * now_num]
        errores[:, el] = solutions[:, el] - solution2
    y1 = np.linalg.norm(errores[0:num], axis=0) / np.linalg.norm(dirichlet_ex)
    y2 = np.linalg.norm(errores[num:2 * num], axis=0) / np.linalg.norm(
        neumann_ex)
    y3 = np.linalg.norm(errores[2 * num:3 * num], axis=0) / np.linalg.norm(
        dirichlet_in)
    y4 = np.linalg.norm(errores[3 * num:4 * num], axis=0) / np.linalg.norm(
        neumann_in)
    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.semilogy(
        y1, marker='p', linestyle='dashed',
        label='$RE2\\left(\\gamma_D^{01}u_0, u_{D,01}^{L}\\right)_1$')
    plt.semilogy(
        y2, marker='*', linestyle='dashed',
        label='$RE2\\left(\\gamma_N^{01}u_0, u_{N,01}^{L}\\right)_1$')
    plt.semilogy(
        y3, marker='x', linestyle='dashed',
        label='$RE2\\left(\\gamma_D^{1}u_0, u_{D,1}^{L}\\right)_1$')
    plt.semilogy(
        y4, marker='.', linestyle='dashed',
        label='$RE2\\left(\\gamma_N^{1}u_0, u_{N,1}^{L}\\right)_1$')
    plt.xlabel('$L$')
    plt.ylabel('Error')
    plt.legend(edgecolor='white')
    plt.rcParams.update({'font.size': 20})
    center = np.asarray([0., 0., 0.])
    inter_horizontal = resolution
    inter_vertical = resolution
    center_positions = [center]
    radius = np.asarray([r])
    
    num_big = num**2
    aux_drawing = np.zeros(4 * num_big)
    aux_drawing[
    0:num_big] = extensions.azimuthal_trace_to_general_with_zeros(
        max_l, solution2[0:num])
    aux_drawing[
    num_big:2 * num_big] = extensions.azimuthal_trace_to_general_with_zeros(
        max_l, solution2[num:2 * num])
    
    big_l_c = 100
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(
        max_l)
    quantity_theta_points, quantity_phi_points, \
        weights, pre_vector, spherical_harmonics = \
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            max_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    eles = np.arange(0, max_l + 1)
    l_square_plus_l = (eles + 1) * eles
    
    vector = pre_vector * radius
    
    surface_field_d0 = np.sum(
        solution2[0:num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                   l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_d0_max = np.max(surface_field_d0)
    surface_field_d0_min = np.min(surface_field_d0)
    surface_field_d0 = (surface_field_d0 - surface_field_d0_min) \
                       / (surface_field_d0_max - surface_field_d0_min)
    surface_field_n0 = np.sum(
        solution2[num:2 * num, np.newaxis, np.newaxis] * spherical_harmonics[
                                                         l_square_plus_l, :, :]
        , axis=0
    )
    surface_field_n0_max = np.max(surface_field_n0)
    surface_field_n0_min = np.min(surface_field_n0)
    surface_field_n0 = (surface_field_n0 - surface_field_n0_min) \
                       / (surface_field_n0_max - surface_field_n0_min)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_d0))
    ax_1.set_xlabel('$x \\ [\\mu m]$')
    ax_1.set_ylabel('$y \\ [\\mu m]$')
    ax_1.set_zlabel('$z \\ [\\mu m]$')
    ax_1.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax_1,
        label='[V]'
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.viridis(surface_field_n0))
    ax.set_xlabel('$x \\ [\\mu m]$')
    ax.set_ylabel('$y \\ [\\mu m]$')
    ax.set_zlabel('$z \\ [\\mu m]$')
    ax.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(vmin=surface_field_d0_min,
                              vmax=surface_field_d0_max),
        cmap=cm.viridis),
        ax=ax,
        label='[V $/ \\mu m$ ]'
    )
    
    def point_source(x: np.ndarray) -> float:
        return 0.
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm(halfrange=0.00045)
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar()
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',  # vmin=0., vmax=0.0032,
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',  # vmin=0., vmax=0.0032,
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    aux_drawing[2 * num_big:3 * num_big] = \
        extensions.azimuthal_trace_to_general_with_zeros(
            max_l, solution2[2 * num:3 * num] + b_max[0:num])
    aux_drawing[3 * num_big:4 * num_big] = \
        extensions.azimuthal_trace_to_general_with_zeros(
            max_l, solution2[3 * num:4 * num] - b_max[num:2 * num])
    
    cut = 1
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar()
    
    cut = 2
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    
    cut = 3
    x1, y1, data = draw.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        aux_drawing, radius, center_positions, max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]')
    pass


def mix_phantom_total_3_different_1_point_source(
        resolution: int
):
    big_l = 50
    big_l_c = 58
    n = 3
    radii = np.asarray([10., 8., 9.])
    center_positions = [np.asarray([0., 0., 0.]),
                        np.asarray([25., 0., 0.]),
                        np.asarray([-24., 0., 0.])]
    sigma_e = 5.
    sigma_i = 0.455
    p0 = np.asarray([0., 0., 20.])
    
    print('- ' + str(n) + ' spheres.')
    print('-- Space convergence of the traces of u for given phi_e.')
    sigmas = np.ones(n + 1) * sigma_e
    sigmas[1] = sigma_i
    
    big_a_0_times_2, sparse_big_a_1_n_times_2, x_dia = \
        bioslaplace.mtf_numpy_scipy_pieces_lazy_version_n_medios_2d(
            sigmas, radii, n, big_l, big_l_c, center_positions)
    matrix = np.concatenate(
        (np.concatenate((big_a_0_times_2,
                         scipy.sparse.dia_matrix(
                             ((1. / x_dia), np.array([0])),
                             shape=np.shape(big_a_0_times_2)).toarray()),
                        axis=1),
         np.concatenate((
             scipy.sparse.dia_matrix(
                 (x_dia, np.array([0])),
                 shape=np.shape(big_a_0_times_2)).toarray(),
             sparse_big_a_1_n_times_2.toarray()), axis=1)), axis=0)
    b = bioslaplace.b_vector_n_spheres_mtf_point_source(
        big_l, n, center_positions, p0, radii, sigmas[0], x_dia)
    solution2 = np.linalg.solve(matrix, b)
    print('--- Discrete Calderon errors:')
    print(np.linalg.norm(
        big_a_0_times_2.dot(solution2[0:2 * n * (big_l + 1)**2])
        - solution2[0:2 * n * (big_l + 1)**2]))
    print(np.linalg.norm(
        sparse_big_a_1_n_times_2.dot(
            solution2[2 * n * (big_l + 1)**2:4 * n * (big_l + 1)**2])
        - solution2[2 * n * (big_l + 1)**2:4 * n * (big_l + 1)**2]))
    
    print('--- Jump error:')
    jump_error = np.linalg.norm(
        solution2[0:2 * n * (big_l + 1)**2] * x_dia
        + solution2[2 * n * (big_l + 1)**2:4 * n * (big_l + 1)**2]
        - b[0:2 * n * (big_l + 1)**2] * x_dia)
    print(jump_error)
    
    print('-- Coefficients of solution.')
    plt.figure()
    plt.plot(solution2, marker='x')
    plt.xlabel('index')
    
    print('-- Coefficients of b.')
    plt.figure()
    plt.plot(b, marker='x')
    plt.xlabel('index')
    
    intensity = 1.
    sigma_e_i, sigma_i_e = \
        functionsofphysicalparameters.coefficients_pre_computations_2_mediums(
            sigmas[0], sigmas[1])
    max_l = big_l
    num = max_l + 1
    b_d = bioslaplace.point_source_coefficients_dirichlet_expansion(
        max_l, radii[0], p0 - center_positions[0], sigmas[0], intensity)
    b_n = bvectorsandtracefunctions. \
        point_source_coefficients_neumann_expansion_0j(
        max_l, radii[0], p0 - center_positions[0], sigmas[0], intensity)
    b = bvectorsandtracefunctions.b_vector_1_sphere_mtf(max_l, sigma_e_i, b_d,
                                                        b_n)
    matrix = bioslaplace.laplace_mtf_1_sphere_matrix(
        max_l, radii[0], sigmas[0], sigmas[1])
    result_one_sphere = np.linalg.solve(matrix, b)
    
    analytic_error = \
        np.linalg.norm(
            result_one_sphere - np.concatenate(
                (solution2[0:2 * (big_l + 1)**2],
                 solution2[2 * n * (big_l + 1)**2:
                           2 * n * (big_l + 1)**2 + 2 * (
                                   big_l + 1)**2]))) / np.linalg.norm(
            result_one_sphere)
    print('--- Analytic error of the first sphere:')
    print(analytic_error)
    print('-- Coefficients of analytic solution 1.')
    plt.figure()
    plt.plot(result_one_sphere, marker='x')
    plt.xlabel('index')
    plt.plot(np.concatenate(
        (solution2[0:2 * (big_l + 1)**2],
         solution2[2 * n * (big_l + 1)**2:
                   2 * n * (big_l + 1)**2 + 2 * (big_l + 1)**2])),
        marker='x')
    plt.figure()
    b1 = bioslaplace.b_vector_n_spheres_mtf_point_source(
        big_l, n, center_positions, p0, radii, sigmas[0], x_dia)
    plt.plot(b, marker='x')
    plt.plot(np.concatenate(
        (b1[0:2 * (big_l + 1)**2],
         b1[2 * n * (big_l + 1)**2:
            2 * n * (big_l + 1)**2 + 2 * (big_l + 1)**2])),
        marker='x')
    center = np.asarray([0., 0., 0.])
    horizontal = 100.
    vertical = 30.
    inter_horizontal = resolution
    inter_vertical = resolution // 2
    
    radius = radii
    max_l = big_l
    
    def point_source(x: np.ndarray) -> float:
        return 0.
    
    solution_cut = np.zeros(np.shape(solution2))
    solution_cut[0:2 * num * n] = solution2[0:2 * num * n]
    cut = 1
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        solution_cut,
        radius, center_positions,
        max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',  # vmax=0.004,
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    
    cut = 2
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(cut, center,
                                                         horizontal,
                                                         vertical,
                                                         inter_horizontal,
                                                         inter_vertical,
                                                         solution_cut,
                                                         radius,
                                                         center_positions,
                                                         max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    
    cut = 3
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(cut, center,
                                                         horizontal,
                                                         vertical,
                                                         inter_horizontal,
                                                         inter_vertical,
                                                         solution_cut,
                                                         radius,
                                                         center_positions,
                                                         max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    
    solution_cut[2 * num * n:3 * num * n] = solution_cut[
                                            2 * num * n:3 * num * n] + b[
                                                                       0:num * n]
    solution_cut[3 * num * n:4 * num * n] = \
        solution_cut[3 * num * n:4 * num * n] - b[num * n:2 * num * n]
    
    cut = 1
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(cut, center,
                                                         horizontal,
                                                         vertical,
                                                         inter_horizontal,
                                                         inter_vertical,
                                                         solution_cut,
                                                         radius,
                                                         center_positions,
                                                         max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[1], horizontal / 2 + center[1],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$y \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    
    cut = 2
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(cut, center,
                                                         horizontal,
                                                         vertical,
                                                         inter_horizontal,
                                                         inter_vertical,
                                                         solution_cut,
                                                         radius,
                                                         center_positions,
                                                         max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[2], vertical / 2 + center[2]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$z \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    
    cut = 3
    x1, y1, data = bioslaplace.draw_cut_laplace_n_sphere(cut, center,
                                                         horizontal,
                                                         vertical,
                                                         inter_horizontal,
                                                         inter_vertical,
                                                         solution_cut,
                                                         radius,
                                                         center_positions,
                                                         max_l, point_source)
    plt.figure()
    plt.imshow(data, origin='lower',
               extent=[-horizontal / 2 + center[0], horizontal / 2 + center[0],
                       -vertical / 2 + center[1], vertical / 2 + center[1]],
               norm=colors.CenteredNorm()
               )
    plt.xlabel('$x \\ [\\mu m]$')
    plt.ylabel('$y \\ [\\mu m]$')
    plt.colorbar(label='[V]', orientation='horizontal')
    return


if __name__ == '__main__':
    phantom_1_point_source_azimuthal(resolution=100)
    plt.show()
    non_phantom_1_point_source_z_alignment_distance_convergence(resolution=100)
    plt.show()