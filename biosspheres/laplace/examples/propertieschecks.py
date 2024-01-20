import numpy as np
import matplotlib.pyplot as plt
import biosspheres.laplace.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures


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


if __name__ == '__main__':
    v_transpose_check()
    