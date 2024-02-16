import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import biosspheres.quadratures.sphere as quadratures
from biosspheres.miscella.auxindexes import pes_y_kus


def overview_gauss_legendre_trapezoidal_2d() -> None:
    big_l_c = 15
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )

    print("big_l_c : ", big_l_c)
    print(
        "Quantity of points for the quadrature in the theta variable: ",
        quantity_theta_points,
    )
    print(
        "Quantity of points for the quadrature in the phi variable:   ",
        quantity_phi_points,
    )

    print("Gauss-Legendre Weights:\n", weights)

    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.scatter3D(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_1d() -> None:
    big_l_c = 15
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
    )
    print("big_l_c : ", big_l_c)
    print("Quantity of points for the quadrature: ", final_length)
    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.scatter3D(
        vector[0, :],
        vector[1, :],
        vector[2, :],
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_real_sh_mapping_2d() -> None:
    big_l = 1
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    big_l_c = 50
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )
    vector = pre_vector
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.scatter3D(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax_1 = fig.add_subplot(141, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[0, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[0, :, :] / surface_max),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 0$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(142, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[1, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[1, :, :] / surface_max),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 1$, $m = -1$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(143, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[2, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[2, :, :] / surface_max),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 1$, $m = 0$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(144, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[3, :, :]))
    ax_1.view_init(30, 135, 0)
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(spherical_harmonics[3, :, :] / surface_max),
    )
    ax_1.set_title("$l = 1$, $m = 1$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )
    plt.show()
    pass


def overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d() -> None:
    big_l = 1
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    big_l_c = 50
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )
    vector = pre_vector
    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax_1 = fig.add_subplot(141, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[0, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[0, :, :]) / surface_max
        ),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 0$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(142, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[1, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[1, :, :]) / surface_max
        ),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 1$, $m = -1$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(143, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[2, :, :]))
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[2, :, :]) / surface_max
        ),
    )
    ax_1.view_init(30, 135, 0)
    ax_1.set_title("$l = 1$, $m = 0$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )

    ax_1 = fig.add_subplot(144, projection="3d")
    surface_max = np.max(np.abs(spherical_harmonics[3, :, :]))
    ax_1.view_init(30, 135, 0)
    ax_1.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(
            np.imag(spherical_harmonics[3, :, :]) / surface_max
        ),
    )
    ax_1.set_title("$l = 1$, $m = 1$")
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.CenteredNorm(halfrange=surface_max), cmap=cm.coolwarm
        ),
        ax=ax_1,
        shrink=0.6,
        orientation="vertical",
        location="left",
    )
    plt.show()
    pass


def overview_real_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    )

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )

    integral = np.sum(
        spherical_harmonics[0, :, :].flatten("F") * transform[0, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten("F") * transform[1, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten("F") * transform[2, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten("F") * transform[3, :]
    )
    print(integral)
    pass


def overview_complex_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )

    integral = np.sum(
        spherical_harmonics[0, :, :].flatten("F") * transform[0, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten("F") * transform[1, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten("F") * transform[2, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten("F") * transform[3, :]
    )
    print(integral)
    pass


def overview_from_sphere_s_cartesian_to_j_spherical_2d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1

    big_l_c = 10
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord, phi_coord, cos_theta_coord = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector,
        )
    )
    fig = plt.figure(figsize=plt.figaspect(1.0), dpi=68.0, layout="constrained")
    ax_1 = fig.add_subplot(111, projection="3d")
    vector = radio_1 * pre_vector
    ax_1.scatter3D(vector[0, :], vector[1, :], vector[2, :])
    sin_theta = np.sqrt(1 - cos_theta_coord**2)
    ax_1.scatter3D(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
    )
    for i in np.arange(0, quantity_theta_points):
        for j in np.arange(0, quantity_phi_points):
            ax_1.quiver(
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                sin_theta[i, j] * np.cos(phi_coord[i, j]),
                sin_theta[i, j] * np.sin(phi_coord[i, j]),
                cos_theta_coord[i, j],
                pivot="tail",
                length=r_coord[i, j],
                arrow_length_ratio=0.0625,
                alpha=0.325,
            )
            pass
        pass
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    plt.show()
    pass


def overview_from_sphere_s_cartesian_to_j_spherical_1d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1

    big_l_c = 10
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
    )
    r_coord, phi_coord, cos_theta_coord = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector
        )
    )
    fig = plt.figure(figsize=plt.figaspect(1.0), dpi=68.0, layout="constrained")
    ax_1 = fig.add_subplot(111, projection="3d")
    vector = radio_1 * pre_vector
    ax_1.scatter3D(vector[0, :], vector[1, :], vector[2, :])
    sin_theta = np.sqrt(1 - cos_theta_coord**2)
    ax_1.scatter3D(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
    )
    for i in np.arange(0, final_length):
        ax_1.quiver(
            np.zeros_like(vector[0, i]),
            np.zeros_like(vector[0, i]),
            np.zeros_like(vector[0, i]),
            sin_theta[i] * np.cos(phi_coord[i]),
            sin_theta[i] * np.sin(phi_coord[i]),
            cos_theta_coord[i],
            pivot="tail",
            length=r_coord[i],
            arrow_length_ratio=0.0625,
            alpha=0.325,
        )
        pass
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    plt.show()
    pass


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1
    big_l_c = 10
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord,
        phi_coord,
        cos_theta_coord,
        er_times_n,
        etheta_times_n,
        ephi_times_n,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector,
    )
    fig = plt.figure(figsize=plt.figaspect(0.3), dpi=68.0, layout="constrained")
    ax_1 = fig.add_subplot(131, projection="3d")
    vector = radio_1 * pre_vector
    ax_1.scatter3D(vector[0, :], vector[1, :], vector[2, :])
    sin_theta = np.sqrt(1 - cos_theta_coord**2)
    ax_1.scatter3D(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
    )

    for i in np.arange(0, quantity_theta_points):
        for j in np.arange(0, quantity_phi_points):
            ax_1.quiver(
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                sin_theta[i, j] * np.cos(phi_coord[i, j]),
                sin_theta[i, j] * np.sin(phi_coord[i, j]),
                cos_theta_coord[i, j],
                pivot="tail",
                length=r_coord[i, j],
                arrow_length_ratio=0.0625,
                alpha=0.005,
            )
            pass
        pass
    ax_1.plot_surface(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(er_times_n),
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    ax_1.set_title(
        "Projection of the unitary vector corresponding to the radius"
    )

    ax_1 = fig.add_subplot(132, projection="3d")
    vector = radio_1 * pre_vector
    ax_1.scatter3D(vector[0, :], vector[1, :], vector[2, :])
    sin_theta = np.sqrt(1 - cos_theta_coord**2)
    ax_1.scatter3D(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
    )

    for i in np.arange(0, quantity_theta_points):
        for j in np.arange(0, quantity_phi_points):
            ax_1.quiver(
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                sin_theta[i, j] * np.cos(phi_coord[i, j]),
                sin_theta[i, j] * np.sin(phi_coord[i, j]),
                cos_theta_coord[i, j],
                pivot="tail",
                length=r_coord[i, j],
                arrow_length_ratio=0.0625,
                alpha=0.005,
            )
            pass
        pass
    ax_1.plot_surface(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(etheta_times_n),
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    ax_1.set_title("Projection of the unitary vector $\\theta$")

    ax_1 = fig.add_subplot(133, projection="3d")
    vector = radio_1 * pre_vector
    ax_1.scatter3D(vector[0, :], vector[1, :], vector[2, :])
    sin_theta = np.sqrt(1 - cos_theta_coord**2)
    ax_1.scatter3D(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
    )

    for i in np.arange(0, quantity_theta_points):
        for j in np.arange(0, quantity_phi_points):
            ax_1.quiver(
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                np.zeros_like(vector[0, i, j]),
                sin_theta[i, j] * np.cos(phi_coord[i, j]),
                sin_theta[i, j] * np.sin(phi_coord[i, j]),
                cos_theta_coord[i, j],
                pivot="tail",
                length=r_coord[i, j],
                arrow_length_ratio=0.0625,
                alpha=0.005,
            )
            pass
        pass
    ax_1.plot_surface(
        r_coord * sin_theta * np.cos(phi_coord),
        r_coord * sin_theta * np.sin(phi_coord),
        r_coord * cos_theta_coord,
        rstride=1,
        cstride=1,
        facecolors=cm.coolwarm(ephi_times_n),
    )
    ax_1.set_xlabel("$x$")
    ax_1.set_ylabel("$y$")
    ax_1.set_zlabel("$z$")
    ax_1.set_aspect("equal")
    ax_1.set_title("Projection of the unitary vector $\\phi$")
    plt.show()
    pass


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d() -> None:
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1
    big_l_c = 10
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
    )
    (
        r_coord,
        phi_coord,
        cos_theta_coord,
        er_times_n,
        etheta_times_n,
        ephi_times_n,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        radio_2, p_1, p_2, final_length, pre_vector
    )
    pass


if __name__ == "__main__":
    overview_gauss_legendre_trapezoidal_2d()
    overview_gauss_legendre_trapezoidal_1d()
    overview_gauss_legendre_trapezoidal_real_sh_mapping_2d()
    overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d()
    overview_real_spherical_harmonic_transform_1d()
    overview_complex_spherical_harmonic_transform_1d()
    overview_from_sphere_s_cartesian_to_j_spherical_2d()
    overview_from_sphere_s_cartesian_to_j_spherical_1d()
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d()
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d()
