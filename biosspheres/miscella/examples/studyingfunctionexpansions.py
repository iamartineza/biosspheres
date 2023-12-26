import numpy as np
import matplotlib.pyplot as plt
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.mathfunctions as mathfunctions


def b_convergence_in_degree_fixed_quadrature_constant_and_linear_functions(
        max_l: int = 6,
        l_c: int = 6,
        radius: float = 10.,
        cte: float = 3.1,
) -> None:
    """
    Maximum degree for representing a linear and a constant function on the
    surface of a sphere.
    We want to know how much spherical harmonics are needed to aproximate the
    function phi_e.
    We do this for two functions:
    phi_e = cte
    phi_e = -cte z

    Parameters
    ----------
    max_l : int
        >= 0, max degree.
    l_c : int
        >= 0, maximum degree for the quadrature.
    radius : float
        > 0, sphere radius.
    cte: float
        constant function = cte
        linear function = cte * z
    
    Returns
    -------
    None
    
    """
    print(
        '\nMaximum degree for representing a linear and a constant function')
    print('on the surface of a sphere.')
    print('- We want to know how much spherical harmonics are needed')
    print('to aproximate the function phi_e.')
    print('-- phi_e = cte')
    l2_norm = np.zeros(max_l)
    quantity_theta_points, quantity_phi_points, w, pre_vector = \
        quadratures.gauss_legendre_trapezoidal_2d(l_c)
    grid_analytic = np.ones((quantity_theta_points, quantity_phi_points)) * cte
    for el in range(0, max_l):
        dirichlet_expansion = \
            pyshtools.expand.SHExpandGLQ(grid_analytic, w, pre_vector[2, :, 0],
                                         norm=4, csphase=-1, lmax_calc=el)
        grid_expansion = \
            pyshtools.expand.MakeGridGLQ(dirichlet_expansion,
                                         pre_vector[2, :, 0],
                                         norm=4, csphase=-1, lmax=l_c)
        l2_norm[el] = np.sqrt(np.sum((grid_expansion - grid_analytic)**2))
    plt.figure()
    plt.semilogy(l2_norm, marker='x', label='$\\phi_{e_1}(x,y,z) = 3.1$')
    relative_error = l2_norm / (3.1 * 2 * np.sqrt(np.pi))
    
    print('-- phi_e = -cte z')
    l2_norm = np.zeros(max_l)
    grid_analytic = pre_vector[2, :, :] * cte * radius
    for el in range(0, max_l):
        dirichlet_expansion = \
            pyshtools.expand.SHExpandGLQ(grid_analytic, w, pre_vector[2, :, 0],
                                         norm=4, csphase=-1, lmax_calc=el)
        grid_expansion = \
            pyshtools.expand.MakeGridGLQ(dirichlet_expansion,
                                         pre_vector[2, :, 0],
                                         norm=4, csphase=-1, lmax=l_c)
        l2_norm[el] = np.sqrt(np.sum((grid_expansion - grid_analytic)**2))
    plt.semilogy(l2_norm, marker='*', label='$\\phi_{e_2}(x,y,z) = -3.1 z $')
    plt.xlabel('$L$')
    plt.ylabel('$||\\phi_{e} - \\phi_{e}^L ||_{L^2(\\Gamma_1)}$')
    plt.legend()
    
    plt.figure()
    plt.semilogy(relative_error, marker='x',
                 label='$\\phi_{e_1}(x,y,z) = 3.1 $')
    relative_error = l2_norm / (3.1 * np.sqrt(radius * 4. * np.pi / 3.))
    plt.semilogy(relative_error, marker='*',
                 label='$\\phi_{e_2}(x,y,z) = -3.1 z $')
    plt.xlabel('$L$')
    y_label = '$RE2(\\phi_{e},\\phi_{e}^L)_1$'
    plt.ylabel(y_label)
    plt.legend(edgecolor='white')
    pass


def b_convergence_in_degree_point_source_different_distances(
        max_eles: np.ndarray = np.asarray([20, 46, 80, 172], dtype=int),
        radius: float = 10.,
        sigma_e: float = 5.,
        intensity: float = 1.,
        distances: np.ndarray = np.asarray([50., 20., 15, 12])
):
    """
    Maximum degree for representing a point source function on the
    surface of a sphere (given the source).
    We want to know how much spherical harmonics are needed to aproximate the
    function phi_e.
    We do this for a set of point source, assuming a sphere of constant radius
    and centered in the origin.

    Parameters
    ----------
    max_eles : np.ndarray
        of ints, 1 dimension, each entry >= 0, max degree.
    radius : float
        > 0, sphere radius.
    sigma_e : float
        > 0, parameter.
    intensity : float
        > 0, parameter.
    distances : np.ndarray
        of floats, distance of the center of the sphere to the point source,
        1 dimension, each entry >= 0, max degree.
    
    Returns
    -------
    None
    
    """
    eles_c = 2 * max_eles
    markers = ['p', '*', 'x', '.']
    plt.figure()
    plt.xlabel('$L$')
    for counter in np.arange(0, len(max_eles)):
        num = max_eles[counter] + 1
        p_0 = np.asarray([0., 0., distances[counter]])
        l2_norm = np.zeros(max_eles[counter] // 2)
        final_length, total_weights, pre_vector = quadratures. \
            gauss_legendre_trapezoidal_1d(eles_c[counter])
        grid_analytic = np.zeros(final_length)
        legendre_functions = np.zeros((num, final_length))
        full_expansion = np.zeros((num, 1))
        full_expansion[:, 0] = harmonicex. \
            point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
            max_eles[counter], radius, distances[counter], sigma_e,
            intensity)
        for ii in np.arange(0, final_length):
            grid_analytic[ii] = mathfunctions. \
                point_source(radius * pre_vector[:, ii], p_0, sigma_e)
            legendre_functions[:, ii] = \
                pyshtools.legendre.PlON(max_eles[counter], pre_vector[2, ii])
        for el in np.arange(0, max_eles[counter], 2):
            grid_expansion = np.sum(
                legendre_functions[0:el + 1, :] * full_expansion[0:el + 1],
                axis=0)
            l2_norm[el // 2 + np.mod(el, 2)] = np.sqrt(
                np.sum((grid_analytic - grid_expansion)**2 * total_weights))
        phi_norm = np.sqrt(np.sum(grid_analytic**2 * total_weights))
        plt.semilogy(np.arange(0, max_eles[counter], 2), l2_norm / phi_norm,
                     marker=markers[counter],
                     label='$d =$ ' + str(distances[counter]))
    y_label = '$RE2(\\phi_{e_3},\\phi_{e_3}^L)_1$'
    plt.ylabel(y_label)
    plt.legend(edgecolor='white')


if __name__ == '__main__':
    b_convergence_in_degree_fixed_quadrature_constant_and_linear_functions()
    b_convergence_in_degree_point_source_different_distances()
    plt.show()
