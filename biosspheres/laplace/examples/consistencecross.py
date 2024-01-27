import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import biosspheres.laplace.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.auxindexes as auxindexes


def testing_consistence_for_the_routines_for_the_cross_interactions() -> None:
    print('Running '
          'testing_consistence_for_the_routines_for_the_cross_interactions')
    radio_1 = 3.
    radio_2 = 2.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    big_l = 3
    big_l_c = 50
    print('- V operator 1d vs 2d.')
    
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
    
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = \
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    (r_coord_1tf_2d, phi_coord_1tf_2d, cos_theta_coord_1tf_2d,
     er_times_n_1tf_2d, etheta_times_n_1tf_2d, ephi_times_n_1tf_2d) = \
        (quadratures.
         from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
            radio_2, p_1, p_2, quantity_theta_points, quantity_phi_points,
            pre_vector_t_2d))
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data_v21_2d = crossinteractions.v_0_sj_semi_analytic_v2d(
        big_l, radio_1, radio_2, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    print('- K operator 1d vs 2d.')
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    data_v12 = crossinteractions.v_0_js_from_v_0_sj(data_v21_2d)
    
    print('- K_{2,1}^0 from V_{2,1} (1d), checking')
    el_diagonal = auxindexes.diagonal_l_sparse(big_l)
    
    print('- K_{1,2}^0 and K_{2,1}^{*0} (1d), '
          'checking semi-analytic routines')
    data_k12 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_2, radio_1, r_coord_2tf, phi_coord_2tf,
        cos_theta_coord_2tf, final_length, transform)
    
    data_ka21 = crossinteractions.ka_0_sj_from_v_sj(data_v21_2d, radio_2,
                                                    el_diagonal)
    
    print('- W routines, checking.')
    data_w21 = -crossinteractions.k_0_sj_from_v_0_sj(data_ka21, radio_1,
                                                     el_diagonal)
    data_w21_v2 = crossinteractions.w_0_sj_from_v_sj(data_v21, radio_1,
                                                     radio_2, el_diagonal)
    aux_w = np.abs(data_w21 - data_w21_v2)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Checking W routines.')
    print('Plotting W v1 vs W v2.')
    
    print("- Routine checking for A (1d) vs previous.")
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
    print('Plotting A_sj vs the others')
    
    data_ka12 = crossinteractions.ka_0_sj_from_v_sj(data_v12, radio_1,
                                                    el_diagonal)
    data_w12 = crossinteractions.w_0_sj_from_v_sj(data_v12, radio_2, radio_1,
                                                  el_diagonal)
    a_12_prev = np.concatenate((
        np.concatenate((-data_k12, data_v12), axis=1),
        np.concatenate((data_w12, data_ka12), axis=1)),
        axis=0)
    aux_w = np.abs(a_12 - a_12_prev)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Checking routine A_js.')
    print('Plotting A_js vs the others')


if __name__ == '__main__':
    testing_consistence_for_the_routines_for_the_cross_interactions()
    plt.show()
