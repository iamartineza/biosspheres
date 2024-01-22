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
    rs = np.asarray([radio_1, radio_2])
    
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
    aux_tr = np.abs(data_v21_2d - data_v21)
    plt.figure()
    plt.imshow(aux_tr, cmap='RdBu')
    plt.colorbar()
    plt.title('V, 1d vs 2d')
    print('Plotting V, 1d vs 2d')
    
    print('- K operator 1d vs 2d.')
    data_k21_2d = crossinteractions.k_0_sj_semi_analytic_v2d(
        big_l, radio_1, radio_2, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    aux_tr = np.abs(data_k21_2d - data_k21)
    plt.figure()
    plt.imshow(aux_tr, cmap='RdBu')
    plt.colorbar()
    plt.title('K, 1d vs 2d')
    print('Plotting K, 1d vs 2d')
    
    print("- K* operator 1d vs 2d.")
    data_ka21_2d = crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
        big_l, radio_1, radio_2, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, er_times_n_1tf_2d, etheta_times_n_1tf_2d,
        ephi_times_n_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q)
    data_ka21 = crossinteractions.ka_0_sj_semi_analytic_recurrence_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, er_times_n_1tf, etheta_times_n_1tf,
        ephi_times_n_1tf, final_length, transform)
    aux_tr = np.abs(data_ka21_2d - data_ka21)
    plt.figure()
    plt.imshow(aux_tr, cmap='RdBu')
    plt.colorbar()
    plt.title('K*, 1d vs 2d')
    print('Plotting K*, 1d vs 2d')
    
    print('- V_{1,2}^0 from V_{2,1}^{0t} (2d)')
    data_v12 = crossinteractions.v_0_js_from_v_0_sj(data_v21_2d)
    aux_tr = np.abs(data_v12 - np.transpose(data_v21_2d)) / np.abs(data_v12)
    plt.figure()
    plt.imshow(aux_tr, cmap='RdBu')
    plt.colorbar()
    plt.title(
        'Relative error between $V_{1,2}^0$ and $V_{2,1}^{0t}$')
    print('Plotting V_{1,2}^0 vs V_{2,1}^{0t}, with property.')
    
    print('- K_{2,1}^0 from V_{2,1} (1d), checking')
    el_diagonal = auxindexes.diagonal_l_sparse(big_l)
    prop = crossinteractions.k_0_sj_from_v_0_sj(data_v21, radio_1, el_diagonal)
    aux_kv = np.abs(data_k21 - prop)
    plt.figure()
    plt.imshow(aux_kv, cmap='RdBu')
    plt.colorbar()
    plt.title(
        'Difference between $K_{2,1}^0$ and $-V_{2,1}^{0}D_l$, ' +
        'semi-analytically routine.')
    print('Plotting K_{2,1}^0 vs -V_{2,1}^{0}D_l, with property.')
    
    print('- K_{1,2}^0 and K_{2,1}^{*0} (1d), '
          'checking semi-analytic routines')
    data_k12 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, radio_2, radio_1, r_coord_2tf, phi_coord_2tf,
        cos_theta_coord_2tf, final_length, transform)
    aux_k_ka = np.abs(data_k12 - np.transpose(data_ka21))
    plt.figure()
    plt.imshow(aux_k_ka, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Difference between $K^{0}_{1,2}$ and  $K^{0*t}_{2,1}$')
    print('Plotting K_{1,2}^0 vs K_{2,1}, semi-analytic.')
    
    print('- K_{2,1}^{*0} from K_{1,2}^0, checking property')
    data_ka21 = crossinteractions.ka_0_sj_from_k_js(data_k12)
    aux_k_ka = np.abs(data_k12 - np.transpose(data_ka21))
    plt.figure()
    plt.imshow(aux_k_ka, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Difference between $K^{0}_{1,2}$ and  $K^{0*t}_{2,1}$ from K')
    print('Plotting K_{1,2}^0 vs K_{2,1}, property.')
    
    print('- K_{1,2}^0 and K_{2,1}^{*0} (1d), checking property')
    data_ka21 = crossinteractions.ka_0_sj_from_v_sj(data_v21_2d, radio_2,
                                                    el_diagonal)
    aux_k_ka = np.abs(data_k12 - np.transpose(data_ka21))
    plt.figure()
    plt.imshow(aux_k_ka, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Difference between $K^{0}_{1,2}$ and  $K^{0*t}_{2,1}$ from V')
    print('Plotting K_{1,2}^0 vs K_{2,1}, property with V.')
    
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
    
    print("Routines 1d and 2d for A.")
    a_21_2d, a_12_2d = crossinteractions.a_0_sj_and_js_v2d(
        big_l, radio_1, radio_2, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q, diagonal)
    aux_w = np.abs(a_21 - a_21_2d)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Checking routine A_sj.')
    print('Plotting A_sj 1d vs 2d')
    aux_w = np.abs(a_12 - a_12_2d)
    plt.figure()
    plt.imshow(aux_w, cmap='RdBu',
               norm=colors.SymLogNorm(linthresh=10**(-8)))
    plt.colorbar()
    plt.title('Checking routine A_js.')
    print('Plotting A_js 1d vs 2d')
    
    print('Difference between 1d and 2d routines')
    n = 2
    p = [p_1, p_2]
    cross_1 = crossinteractions.all_cross_interactions_n_spheres_v1d(
        n, big_l, big_l_c, rs, p)
    cross_2 = crossinteractions.all_cross_interactions_n_spheres_v2d(
        n, big_l, big_l_c, rs, p)
    print('-Cross 1d vs 2d')
    print(np.linalg.norm(cross_2 - cross_1))
    
    big_l = 10
    big_l_c = 50
    final_length, pre_vector_t, transform = \
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf, \
        er_times_n_1tf, etheta_times_n_1tf, ephi_times_n_1tf = \
        quadratures. \
        from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t)
    diagonal = auxindexes.diagonal_l_dense(big_l)
    ma_1, ma_2 = crossinteractions.a_0_sj_and_js_v1d(
        big_l, radio_1, radio_2, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform, diagonal)
    ma_1_2d, ma_2_2d = crossinteractions.a_0_sj_and_js_v2d(
        big_l, radio_1, radio_2, r_coord_1tf_2d, phi_coord_1tf_2d,
        cos_theta_coord_1tf_2d, weights, pre_vector_t_2d[2, :, 0],
        quantity_theta_points, quantity_phi_points, pesykus, p2_plus_p_plus_q,
        p2_plus_p_minus_q, diagonal)
    print('1 vs 2d')
    aux2 = ma_1 - ma_1_2d
    print(np.linalg.norm(aux2))


if __name__ == '__main__':
    testing_consistence_for_the_routines_for_the_cross_interactions()
    plt.show()
