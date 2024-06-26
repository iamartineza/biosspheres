import numpy as np
import matplotlib.pyplot as plt
import pyshtools
import biosspheres.quadratures.sphere as quadratures
import biosspheres.miscella.harmonicex as harmonicex
import biosspheres.miscella.mathfunctions as mathfunctions


def sh_expansion_convergence_point_source_different_distances(
    max_eles: np.ndarray = np.asarray([20, 46, 80, 172], dtype=int),
    radius: float = 10.0,
    sigma_e: float = 5.0,
    intensity: float = 1.0,
    distances: np.ndarray = np.asarray([50.0, 20.0, 15, 12]),
) -> None:
    """
    Maximum degree for representing a point source function on the
    surface of a sphere (given the source).
    We want to know how much spherical harmonics are needed to
    aproximate the function phi_e.
    We do this for a set of point source, assuming a sphere of constant
    radius and centered in the origin.

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
        of floats, distance of the center of the sphere to the point
        source, 1 dimension, each entry >= 0, max degree.

    Returns
    -------
    None

    """
    eles_c = 2 * max_eles
    markers = ["p", "*", "x", "."]
    plt.figure()
    plt.xlabel("$L$")
    for counter in np.arange(0, len(max_eles)):
        num = max_eles[counter] + 1
        p_0 = np.asarray([0.0, 0.0, distances[counter]])
        l2_norm = np.zeros(max_eles[counter] // 2)
        final_length, total_weights, pre_vector = (
            quadratures.gauss_legendre_trapezoidal_1d(eles_c[counter])
        )
        grid_analytic = np.zeros(final_length)
        legendre_functions = np.zeros((num, final_length))
        full_expansion = np.zeros((num, 1))
        full_expansion[:, 0] = (
            harmonicex.point_source_coefficients_dirichlet_expansion_azimuthal_symmetry(
                max_eles[counter],
                radius,
                distances[counter],
                sigma_e,
                intensity,
            )
        )
        for ii in np.arange(0, final_length):
            grid_analytic[ii] = mathfunctions.point_source(
                radius * pre_vector[:, ii], p_0, sigma_e
            )
            legendre_functions[:, ii] = pyshtools.legendre.PlON(
                max_eles[counter], pre_vector[2, ii]
            )
            pass
        for el in np.arange(0, max_eles[counter], 2):
            grid_expansion = np.sum(
                legendre_functions[0 : el + 1, :] * full_expansion[0 : el + 1],
                axis=0,
            )
            l2_norm[el // 2 + np.mod(el, 2)] = np.sqrt(
                np.sum((grid_analytic - grid_expansion) ** 2 * total_weights)
            )
            pass
        phi_norm = np.sqrt(np.sum(grid_analytic**2 * total_weights))
        plt.semilogy(
            np.arange(0, max_eles[counter], 2),
            l2_norm / phi_norm,
            marker=markers[counter],
            label="$d =$ " + str(distances[counter]),
        )
        pass
    y_label = "$RE2(\\phi_{e_3},\\phi_{e_3}^L)_1$"
    plt.ylabel(y_label)
    plt.legend(edgecolor="white")
    pass


def sh_expansion_convergence_plane_wave_different_k() -> None:
    max_eles = np.asarray([16, 34, 52, 66, 80], dtype=int)
    radius = 1.0
    a = 1.0
    k = np.asarray([1.0, 11.0, 21.0, 31.0, 41.0])

    eles_c = max_eles + 10
    markers = ["o", "p", "*", "x", "."]
    plt.figure()
    plt.xlabel("$L$")

    k0 = np.asarray([0.0, 0.0, 0.0])
    for counter in np.arange(0, len(max_eles)):
        num = max_eles[counter] + 1
        k0[2] = k[counter]
        l2_norm = np.zeros(max_eles[counter] // 2)
        final_length, total_weights, pre_vector = (
            quadratures.gauss_legendre_trapezoidal_1d(eles_c[counter])
        )
        grid_analytic = np.zeros(final_length, dtype=np.complex128)
        legendre_functions = np.zeros((num, final_length))
        full_expansion = np.zeros((num, 1), dtype=np.complex128)
        full_expansion[:, 0] = (
            harmonicex.plane_wave_coefficients_dirichlet_expansion_0j(
                max_eles[counter], radius, 0.0, k[counter], a, True
            )
        )
        for ii in np.arange(0, final_length):
            grid_analytic[ii] = mathfunctions.plane_wave(
                a, k0, pre_vector[:, ii], np.zeros(3)
            )
            legendre_functions[:, ii] = pyshtools.legendre.PlON(
                max_eles[counter],
                pre_vector[2, ii],
            )
        for el in np.arange(0, max_eles[counter], 2):
            grid_expansion = np.sum(
                legendre_functions[0 : el + 1, :] * full_expansion[0 : el + 1],
                axis=0,
            )
            l2_norm[el // 2 + np.mod(el, 2)] = np.sqrt(
                np.sum(
                    np.abs(grid_analytic - grid_expansion) ** 2 * total_weights
                )
            )
        phi_norm = np.sqrt(np.sum(np.abs(grid_analytic) ** 2 * total_weights))
        plt.semilogy(
            np.arange(0, max_eles[counter], 2),
            l2_norm / phi_norm,
            marker=markers[counter],
            label="$k =$ " + str(k[counter]),
        )
    y_label = "Approximated relative error in $L^2(\mathbb{S})$"
    plt.ylabel(y_label)
    plt.legend(edgecolor="white")
    plt.show()
    pass


if __name__ == "__main__":
    sh_expansion_convergence_plane_wave_different_k()
    sh_expansion_convergence_point_source_different_distances()
    plt.show()
