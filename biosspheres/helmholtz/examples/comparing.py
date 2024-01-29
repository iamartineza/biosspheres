import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import biosspheres.helmholtz.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.auxindexes as auxindexes


def v_1d_vs_2d() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 5
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_v21_2d - data_v21) / np.abs(data_v21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $V_{2,1}^0$ with 1d and 2d routines')
    plt.show()
    pass


def v_js_from_the_other() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 3
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_l_2 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_2 * k0)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    r_coord_2tf_2d, phi_coord_2tf_2d, cos_theta_coord_2tf_2d = \
        (quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_1, p_2, p_1, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v12_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l, k0, radio_2, radio_1, j_l_2, r_coord_2tf_2d, phi_coord_2tf_2d,
        cos_theta_coord_2tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    data_v21 = crossinteractions.v_0_js_from_v_0_sj(data_v12_2d)
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_v21_2d - data_v21) / np.abs(data_v21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $V_{2,1}^0$ with direct and indirect '
              'routines')
    plt.show()
    pass


def k_1d_vs_2d() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 5
    big_l_c = 25
    
    j_lp_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0,
                                        derivative=True)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_lp_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_k21_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l, k0, radio_1, radio_2, j_lp_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_k21_2d - data_k21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Absolute error between $K_{2,1}^0$ with 1d and 2d routines')
    plt.show()
    pass


def k_0_sj_from_the_other() -> None:
    radio_1 = 1.3
    radio_2 = 1.7
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 3
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0,
                                        derivative=True)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    data_k21_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l, k0, radio_1, radio_2, j_lp_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    data_k21 = crossinteractions.k_0_sj_from_v_0_sj(data_v21_2d, k0, radio_1)
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_k21_2d - data_k21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Absolute error between $K_{2,1}^0$ with direct and indirect '
              'routines')
    plt.show()
    pass


def ka_1d_vs_2d() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 5
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    (r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf,
     er_1tf, eth_1tf, ephi_1tf) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    data_ka21 = crossinteractions.ka_0_sj_semi_analytic_recurrence_v1d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, er_1tf, eth_1tf, ephi_1tf, final_length,
        transform)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    (r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d,
     er_1tf_2d, eth_1tf_2d, ephi_1tf_2d) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, er_1tf_2d, eth_1tf_2d, ephi_1tf_2d, weights,
        pre_vector_t_2d[2, :, 0], quantity_theta_points, quantity_phi_points,
        pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_ka21_2d - data_ka21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Absolute error between $K_{2,1}^{*0}$ with 1d and 2d routines')
    plt.show()
    pass


def ka_js_from_the_other() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 3
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_lp_2 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_2 * k0,
                                        derivative=True)
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    (r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d,
     er_1tf_2d, eth_1tf_2d, ephi_1tf_2d) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, er_1tf_2d, eth_1tf_2d, ephi_1tf_2d, weights,
        pre_vector_t_2d[2, :, 0], quantity_theta_points, quantity_phi_points,
        pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q)
    
    r_coord_2tf_2d, phi_coord_2tf_2d, cos_theta_coord_2tf_2d = \
        (quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_1, p_2, p_1, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_k12_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l, k0, radio_2, radio_1, j_lp_2, r_coord_2tf_2d, phi_coord_2tf_2d,
        cos_theta_coord_2tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    
    data_ka21 = crossinteractions.ka_0_sj_from_k_js(data_k12_2d)
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_ka21_2d - data_ka21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Absolute error between $K_{2,1}^{*0}$ with direct and indirect '
              'routines')
    plt.show()
    pass


if __name__ == '__main__':
    v_1d_vs_2d()
    v_js_from_the_other()
    k_1d_vs_2d()
    k_0_sj_from_the_other()
    ka_1d_vs_2d()
    ka_js_from_the_other()
