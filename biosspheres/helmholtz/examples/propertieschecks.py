import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import biosspheres.helmholtz.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures


def v_transpose_check() -> None:
    radio_1 = 1.2
    radio_2 = 3.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 2
    big_l_c = 25
    
    j_l_1 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_1 * k0)
    j_l_2 = scipy.special.spherical_jn(np.arange(0, big_l + 1), radio_2 * k0)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    r_coord_2tf, phi_coord_2tf, cos_theta_coord_2tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_1, p_2, p_1, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    data_v12 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, k0, radio_2, radio_1, j_l_2, r_coord_2tf, phi_coord_2tf,
        cos_theta_coord_2tf, final_length, transform)
    
    sign_array = np.diag(
        (-np.ones((big_l + 1)**2))**(np.arange(0, (big_l + 1)**2)))
    giro_array = np.eye((big_l + 1)**2)
    eles = np.arange(0, big_l+1)
    l_square_plus_l = eles * (eles + 1)
    for el in np.arange(1, big_l + 1):
        giro_array[l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1,
                   l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1] = (
            np.fliplr(giro_array[
                      l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1,
                      l_square_plus_l[el] - el:l_square_plus_l[el] + el + 1]))
    aux = giro_array@sign_array@np.transpose(data_v21)@sign_array@giro_array
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_v12 - aux) / np.abs(data_v12)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $V_{1,2}^0$ and'
              ' a transformation of $V_{2,1}^{0t}$.')
    plt.show()
    pass


def k_with_v_check() -> None:
    radio_1 = 1.2
    radio_2 = 3.
    
    p_1 = np.asarray([2., 3., 4.])
    p_2 = -p_1
    
    k0 = 7.
    
    big_l = 2
    big_l_c = 25
    
    eles = np.arange(0, big_l + 1)
    j_l_1 = scipy.special.spherical_jn(eles, radio_1 * k0)
    j_lp_1 = scipy.special.spherical_jn(eles, radio_1 * k0)
    
    final_length, pre_vector_t, transform = \
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    r_coord_1tf, phi_coord_1tf, cos_theta_coord_1tf = (
        quadratures.
        from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector_t))
    
    data_v21 = crossinteractions.v_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_l_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    data_k21 = crossinteractions.k_0_sj_semi_analytic_v1d(
        big_l, k0, radio_1, radio_2, j_lp_1, r_coord_1tf, phi_coord_1tf,
        cos_theta_coord_1tf, final_length, transform)
    
    jeys_array = np.diag(np.repeat(-k0 * (j_lp_1 / j_l_1), 2 * eles + 1))
    
    aux = data_v21@jeys_array
    
    plt.figure(dpi=75., layout='constrained')
    im = np.abs(data_k21 - aux) / np.abs(data_k21)
    plt.imshow(im, cmap='RdBu')
    plt.colorbar()
    plt.title('Relative error between $K_{2,1}^0$ and'
              ' a transformation of $V_{2,1}^{0}$.')
    plt.show()
    pass


if __name__ == '__main__':
    v_transpose_check()
    k_with_v_check()
    