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
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
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


if __name__ == '__main__':
    v_1d_vs_2d()
