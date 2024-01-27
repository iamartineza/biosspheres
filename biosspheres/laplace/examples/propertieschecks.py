import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import biosspheres.laplace.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.auxindexes as auxindexes


def v_transpose_check() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 3
    big_l_c = 25
    
    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    r_coord_2tf, phi_coord_2tf, cos_theta_coord_2tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_1, p_2, p_1, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    data_v12 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_2, radio_1, r_coord_2tf, phi_coord_2tf,
        cos_theta_coord_2tf, final_length, transform)
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_v12 - np.transpose(data_v21)) / np.abs(data_v12)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $V_{1,2}^0$ and $V_{2,1}^{0t}$.')
    plt.show()
    pass


def k_with_v_check() -> None:
    radio_1 = 1.2
    radio_2 = 3.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 2
    big_l_c = 25
    
    eles = np.arange(0, big_l + 1)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    el_diagonal_array = np.diag(np.repeat(eles / -radio_1, 2 * eles + 1))
    
    aux = data_v21 @ el_diagonal_array
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_k21 - aux)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Absolute error between $K_{2,1}^0$ and'
              ' a transformation of $V_{2,1}^{0}$.')
    plt.show()
    pass


def w_transpose_check() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 3
    big_l_c = 25
    
    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    (r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf, er_times_n_1tf,
     etheta_times_n_1tf, ephi_times_n_1tf) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    (r_coord_2tf, phi_coord_2tf, cos_theta_coord_2tf, er_times_n_2tf,
     etheta_times_n_2tf, ephi_times_n_2tf) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_1, p_2, p_1, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    data_v12 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_2, radio_1, r_coord_2tf, phi_coord_2tf,
        cos_theta_coord_2tf, final_length, transform)
    
    el_diagonal = auxindexes.diagonal_l_sparse(big_l)
    data_w12 = crossinteractions.w_0_sj_from_v_sj(data_v12, radio_2, radio_1,
                                                  el_diagonal)
    data_w21 = crossinteractions.w_0_sj_from_v_sj(data_v21, radio_1, radio_2,
                                                  el_diagonal)
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_w12 - np.transpose(data_w21))
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $W_{1,2}^0$ and $W_{2,1}^{0t}$.')
    plt.show()
    pass


def w_with_ka_and_v() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 3
    big_l_c = 25
    
    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    (r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf, er_times_n_1tf,
     etheta_times_n_1tf, ephi_times_n_1tf) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    el_diagonal = auxindexes.diagonal_l_sparse(big_l)
    data_ka21 = crossinteractions.ka_0_sj_from_v_sj(data_v21, radio_2,
                                                    el_diagonal)
    
    data_w21 = -crossinteractions.k_0_sj_from_v_0_sj(data_ka21, radio_1,
                                                     el_diagonal)
    data_w21_v2 = crossinteractions.w_0_sj_from_v_sj(data_v21, radio_1,
                                                     radio_2, el_diagonal)
    
    aux_w = np.abs(data_w21 - data_w21_v2)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.show()
    pass


def calderon_build_check() -> None:
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 3
    big_l_c = 50
    
    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    (r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf, er_times_n_1tf,
     etheta_times_n_1tf, ephi_times_n_1tf) = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    el_diagonal = auxindexes.diagonal_l_sparse(big_l)
    data_ka21 = crossinteractions.ka_0_sj_from_v_sj(data_v21, radio_2,
                                                    el_diagonal)
    data_w21 = -crossinteractions.k_0_sj_from_v_0_sj(data_ka21, radio_1,
                                                     el_diagonal)
    
    diagonal = auxindexes.diagonal_l_dense(big_l)
    a_21, a_12 = crossinteractions.a_0_sj_and_js_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform, diagonal)
    a_21_prev = np.concatenate((
        np.concatenate((-data_k21, data_v21), axis=1),
        np.concatenate((data_w21, data_ka21), axis=1)),
        axis=0)
    aux_w = np.abs(a_21 - a_21_prev)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Checking routine A_sj.')
    plt.show()
    pass


if __name__ == '__main__':
    v_transpose_check()
    k_with_v_check()
    w_transpose_check()
    calderon_build_check()
    