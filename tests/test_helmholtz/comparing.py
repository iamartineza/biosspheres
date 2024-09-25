import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import biosspheres.helmholtz.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes


def v_1d_vs_2d() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 5
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)

    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t
        )
    )
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        final_length,
        transform,
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_v21_2d - data_v21) / np.abs(data_v21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title("Relative error between $V_{2,1}^0$ with 1d and 2d routines")
    plt.show()
    pass


def v_js_from_the_other() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 3
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_l_2 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_2 * k0)

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    r_coord_2tf_2d, phi_coord_2tf_2d, cos_theta_coord_2tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_1,
            p_2,
            p_1,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    data_v12_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_2,
        radio_1,
        j_l_2,
        r_coord_2tf_2d,
        phi_coord_2tf_2d,
        cos_theta_coord_2tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    data_v21 = crossinteractions.v_0_js_from_v_0_sj(data_v12_2d)
    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_v21_2d - data_v21) / np.abs(data_v21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title(
        "Relative error between $V_{2,1}^0$ with direct and indirect "
        "routines"
    )
    plt.show()
    pass


def k_1d_vs_2d() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 5
    big_l_c = 25

    j_lp_1 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_1 * k0, derivative=True
    )

    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t
        )
    )
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        final_length,
        transform,
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_k21_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_k21_2d - data_k21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title("Absolute error between $K_{2,1}^0$ with 1d and 2d routines")
    plt.show()
    pass


def k_0_sj_from_the_other() -> None:
    radio_1 = 1.3
    radio_2 = 1.7

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 3
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_1 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_1 * k0, derivative=True
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    data_k21_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    data_k21 = crossinteractions.k_0_sj_from_v_0_sj(data_v21_2d, k0, radio_1)
    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_k21_2d - data_k21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title(
        "Absolute error between $K_{2,1}^0$ with direct and indirect "
        "routines"
    )
    plt.show()
    pass


def ka_1d_vs_2d() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 5
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)

    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    (
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        er_1tf,
        eth_1tf,
        ephi_1tf,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        radio_2, p_1, p_2, final_length, pre_vector_t
    )
    data_ka21 = crossinteractions.ka_0_sj_semi_analytic_recurrence_v1d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        er_1tf,
        eth_1tf,
        ephi_1tf,
        final_length,
        transform,
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector_t_2d,
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_ka21_2d - data_ka21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title("Absolute error between $K_{2,1}^{*0}$ with 1d and 2d routines")
    plt.show()
    pass


def ka_js_from_the_other() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 3
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_2 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_2 * k0, derivative=True
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector_t_2d,
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    r_coord_2tf_2d, phi_coord_2tf_2d, cos_theta_coord_2tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_1,
            p_2,
            p_1,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_k12_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l,
        k0,
        radio_2,
        radio_1,
        j_lp_2,
        r_coord_2tf_2d,
        phi_coord_2tf_2d,
        cos_theta_coord_2tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    data_ka21 = crossinteractions.ka_0_sj_from_k_js(data_k12_2d)
    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_ka21_2d - data_ka21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title(
        "Absolute error between $K_{2,1}^{*0}$ with direct and indirect "
        "routines"
    )
    plt.show()
    pass


def w_1d_vs_2d() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 5
    big_l_c = 25

    j_lp_1 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_1 * k0, derivative=True
    )

    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    (
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        er_1tf,
        eth_1tf,
        ephi_1tf,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        radio_2, p_1, p_2, final_length, pre_vector_t
    )
    data_ka21 = crossinteractions.w_0_sj_semi_analytic_recurrence_v1d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        er_1tf,
        eth_1tf,
        ephi_1tf,
        final_length,
        transform,
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector_t_2d,
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.w_0_sj_semi_analytic_recurrence_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_ka21_2d - data_ka21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title("Absolute error between $W_{2,1}^{0}$ with 1d and 2d routines")
    plt.show()
    pass


def w_js_from_the_other() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 3
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_1 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_1 * k0, derivative=True
    )

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector_t_2d,
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )
    data_w21_2d = crossinteractions.w_0_sj_semi_analytic_recurrence_v2d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_lp_1,
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
        weights,
        pre_vector_t_2d[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        pesykus,
        p2_plus_p_plus_q,
        p2_plus_p_minus_q,
    )

    data_w21 = crossinteractions.w_0_sj_from_ka_sj(data_ka21_2d, k0, radio_1)
    plt.figure(dpi=75.0, layout="constrained")
    im = np.abs(data_w21_2d - data_w21)
    plt.imshow(im, cmap="RdBu")
    plt.colorbar()
    plt.title(
        "Absolute error between $W_{2,1}^{0}$ with direct and indirect "
        "routines"
    )
    plt.show()
    pass


def calderon_versions() -> None:
    radio_1 = 3.0
    radio_2 = 2.0

    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1

    k0 = 7.0

    big_l = 2
    big_l_c = 25

    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_1 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_1 * k0, derivative=True
    )
    j_l_2 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_2 * k0)
    j_lp_2 = scipy.special.spherical_jn(
        np.arange(0, big_l + 1), radio_2 * k0, derivative=True
    )
    k0_ratio_j_l_1 = k0 * j_lp_1 / j_l_1
    k0_ratio_j_l_2 = k0 * j_lp_2 / j_l_2

    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    (
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        er_1tf,
        eth_1tf,
        ephi_1tf,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        radio_2, p_1, p_2, final_length, pre_vector_t
    )
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l,
        k0,
        radio_1,
        radio_2,
        j_l_1,
        r_coord_1tf,
        phi_coord_1tf,
        cos_theta_coord_1tf,
        final_length,
        transform,
    )

    gs = auxindexes.giro_sign(big_l)

    a_21, a_12 = crossinteractions.a_0_sj_and_js_from_v_sj(
        big_l, data_v21, k0_ratio_j_l_1, k0_ratio_j_l_2, gs
    )

    data_v_sj, data_k_sj, data_ka_sj, data_w_sj = (
        crossinteractions.v_k_w_0_sj_from_quadratures_1d(
            big_l,
            k0,
            radio_1,
            radio_2,
            j_l_1,
            j_lp_1,
            r_coord_1tf,
            phi_coord_1tf,
            cos_theta_coord_1tf,
            er_1tf,
            eth_1tf,
            ephi_1tf,
            final_length,
            transform,
        )
    )

    a_21_1d, a_12_1d = crossinteractions.a_0_sj_and_js_from_v_k_w(
        data_v_sj, data_k_sj, data_ka_sj, data_w_sj, gs
    )
    aux = np.abs(a_21 - a_21_1d)
    plt.figure()
    plt.imshow(aux, cmap="RdBu", norm=colors.SymLogNorm(linthresh=10 ** (-8)))
    plt.colorbar()
    plt.title("Checking routine A_sj indirect vs 1d.")
    aux = np.abs(a_12 - a_12_1d)
    plt.figure()
    plt.imshow(aux, cmap="RdBu", norm=colors.SymLogNorm(linthresh=10 ** (-8)))
    plt.colorbar()
    plt.title("Checking routine A_js indirect vs 1d.")

    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord_1tf_2d,
        phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d,
        er_1tf_2d,
        eth_1tf_2d,
        ephi_1tf_2d,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector_t_2d,
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)

    data_v_sj, data_k_sj, data_ka_sj, data_w_sj = (
        crossinteractions.v_k_w_0_sj_from_quadratures_2d(
            big_l,
            k0,
            radio_1,
            radio_2,
            j_l_1,
            j_lp_1,
            r_coord_1tf_2d,
            phi_coord_1tf_2d,
            cos_theta_coord_1tf_2d,
            er_1tf_2d,
            eth_1tf_2d,
            ephi_1tf_2d,
            weights,
            pre_vector_t_2d[2, :, 0],
            quantity_theta_points,
            quantity_phi_points,
            pesykus,
            p2_plus_p_plus_q,
            p2_plus_p_minus_q,
        )
    )

    a_21_2d, a_12_2d = crossinteractions.a_0_sj_and_js_from_v_k_w(
        data_v_sj, data_k_sj, data_ka_sj, data_w_sj, gs
    )
    aux = np.abs(a_21 - a_21_2d)
    plt.figure()
    plt.imshow(aux, cmap="RdBu", norm=colors.SymLogNorm(linthresh=10 ** (-8)))
    plt.colorbar()
    plt.title("Checking routine A_sj indirect vs 2d.")
    aux = np.abs(a_12 - a_12_2d)
    plt.figure()
    plt.imshow(aux, cmap="RdBu", norm=colors.SymLogNorm(linthresh=10 ** (-8)))
    plt.colorbar()
    plt.title("Checking routine A_js indirect vs 2d.")
    plt.show()
    pass


def all_cross_versions() -> None:
    n = 3
    big_l = 5
    big_l_c = 28
    radii = np.ones(n) * 1.112
    center_positions = [
        np.asarray([0.0, 0.0, 0]),
        np.asarray([-7.0, -3.0, -2.0]),
        np.asarray([3.0, 5.0, 7.0]),
    ]
    k0 = 7.0

    eles = np.arange(0, big_l + 1)
    j_l = np.empty((n, big_l + 1))
    j_lp = np.empty((n, big_l + 1))

    for j in np.arange(0, n):
        j_l[j, :] = scipy.special.spherical_jn(eles, radii[j] * k0)
        j_lp[j, :] = scipy.special.spherical_jn(
            eles, radii[j] * k0, derivative=True
        )

    almost_big_a0_from_v_1d = (
        crossinteractions.all_cross_interactions_n_spheres_from_v_1d(
            n, big_l, big_l_c, k0, radii, center_positions, j_l, j_lp
        )
    )

    almost_big_a0_from_v_2d = (
        crossinteractions.all_cross_interactions_n_spheres_from_v_2d(
            n, big_l, big_l_c, k0, radii, center_positions, j_l, j_lp
        )
    )

    almost_big_a0_1d = crossinteractions.all_cross_interactions_n_spheres_1d(
        n, big_l, big_l_c, k0, radii, center_positions
    )

    almost_big_a0_2d = crossinteractions.all_cross_interactions_n_spheres_2d(
        n, big_l, big_l_c, k0, radii, center_positions
    )

    print(
        "Difference between all_cross_interaction_routines:\n"
        + "\t- from_v_1d vs from_v_2d\n"
        + str(np.linalg.norm(almost_big_a0_from_v_1d - almost_big_a0_from_v_2d))
        + "\n\t- from_v_2d vs 1d\n"
        + str(np.linalg.norm(almost_big_a0_1d - almost_big_a0_from_v_2d))
        + "\n\t- 1d vs 2d\n"
        + str(np.linalg.norm(almost_big_a0_1d - almost_big_a0_2d))
    )
    aux = np.abs(almost_big_a0_1d - almost_big_a0_from_v_2d)
    plt.figure()
    plt.imshow(aux, cmap="RdBu", norm=colors.SymLogNorm(linthresh=10 ** (-8)))
    plt.colorbar()
    plt.title("Difference between from_v_2d and 1d")
    plt.show()
    pass


if __name__ == "__main__":
    v_1d_vs_2d()
    v_js_from_the_other()
    k_1d_vs_2d()
    k_0_sj_from_the_other()
    ka_1d_vs_2d()
    ka_js_from_the_other()
    w_1d_vs_2d()
    w_js_from_the_other()
    calderon_versions()
    all_cross_versions()
