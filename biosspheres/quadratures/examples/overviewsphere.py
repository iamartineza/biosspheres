import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import biosspheres.quadratures.sphere as quadratures
from biosspheres.miscella.auxindexes import pes_y_kus


def overview_gauss_legendre_trapezoidal_2d() -> None:
    big_l_c = 15
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c))
    
    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.scatter3D(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :])
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_1d() -> None:
    big_l_c = 15
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c))
    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.scatter3D(
        vector[0, :],
        vector[1, :],
        vector[2, :])
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_real_sh_mapping_2d() -> None:
    big_l = 1
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    big_l_c = 50
    (quantity_theta_points, quantity_phi_points, weights, pre_vector,
     spherical_harmonics) = (
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q))
    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.scatter3D(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :])
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    plt.show()
    
    import numpy as np
    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax_1 = fig.add_subplot(141, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[0, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[0, :, :] / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 0$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(142, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[1, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[1, :, :] / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 1$, $m = -1$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(143, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[2, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[2, :, :] / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 1$, $m = 0$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(144, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[3, :, :]))
    ax_1.view_init(30, 135, 0)
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[3, :, :] / surface_max))
    ax_1.set_title('$l = 1$, $m = 1$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d() -> None:
    big_l = 1
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    big_l_c = 50
    (quantity_theta_points, quantity_phi_points, weights, pre_vector,
     spherical_harmonics) = (
        quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
            big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q))
    vector = pre_vector
    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax_1 = fig.add_subplot(141, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[0, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[0, :, :]) / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 0$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(142, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[1, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[1, :, :]) / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 1$, $m = -1$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(143, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[2, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[2, :, :]) / surface_max))
    ax_1.view_init(30, 135, 0)
    ax_1.set_title('$l = 1$, $m = 0$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    
    ax_1 = fig.add_subplot(144, projection='3d')
    surface_max = np.max(np.abs(spherical_harmonics[3, :, :]))
    ax_1.view_init(30, 135, 0)
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1, cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[3, :, :]) / surface_max))
    ax_1.set_title('$l = 1$, $m = 1$')
    ax_1.set_xlabel('$x$')
    ax_1.set_ylabel('$y$')
    ax_1.set_zlabel('$z$')
    ax_1.set_aspect('equal')
    fig.colorbar(
        cm.ScalarMappable(norm=colors.CenteredNorm(halfrange=surface_max),
                          cmap=cm.coolwarm), ax=ax_1, shrink=0.6,
        orientation='vertical', location='left')
    plt.show()
    pass


def overview_real_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c))
    
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (quantity_theta_points, quantity_phi_points, weights, pre_vector,
     spherical_harmonics) = (
        quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
            big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q))
    
    integral = np.sum(
        spherical_harmonics[0, :, :].flatten('F') * transform[0, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten('F') * transform[1, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten('F') * transform[2, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten('F') * transform[3, :])
    print(integral)
    pass


def overview_complex_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c))
    
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (quantity_theta_points, quantity_phi_points, weights, pre_vector,
     spherical_harmonics) = (
        quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
            big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q))
    
    integral = np.sum(
        spherical_harmonics[0, :, :].flatten('F') * transform[0, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten('F') * transform[1, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten('F') * transform[2, :])
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten('F') * transform[3, :])
    print(integral)
    pass


if __name__ == '__main__':
    overview_gauss_legendre_trapezoidal_2d()
    overview_gauss_legendre_trapezoidal_1d()
    overview_gauss_legendre_trapezoidal_real_sh_mapping_2d()
    overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d()
    overview_real_spherical_harmonic_transform_1d()
    overview_complex_spherical_harmonic_transform_1d()