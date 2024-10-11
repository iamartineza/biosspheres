"""
This module contains the test for biosspheres.quadratures.sphere.py
"""

import pytest
import numpy as np
import biosspheres.quadratures.sphere as quadratures
from biosspheres.utils.auxindexes import pes_y_kus


def test_gauss_legendre_trapezoidal_2d() -> None:
    """
    First test for the routine
    biosspheres.quadratures.gauss_legendre_trapezoidal_2d

    """
    try:
        for big_l_c in [0, 1, 5]:
            quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    except Exception as e:
        print(
            "biosspheres.quadratures.gauss_legendre_trapezoidal_2d"
            + " did not work with big_l_c = ",
            big_l_c,
            ". Exception: ",
            e,
        )
    list_big_l_c = np.array([0, 1, 2, 3, 5, 10])
    for big_l_c in list_big_l_c:
        quantity_theta_points, quantity_phi_points, weights, pre_vector = (
            quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
        )
        assert len(weights) == quantity_theta_points
        assert len(pre_vector[:, 0, 0]) == 3
        assert len(pre_vector[0, :, 0]) == quantity_theta_points
        assert len(pre_vector[0, 0, :]) == quantity_phi_points
        assert len(weights) == quantity_theta_points
        for ii in np.arange(0, quantity_theta_points):
            for jj in np.arange(0, quantity_phi_points):
                assert np.linalg.norm(pre_vector[:, ii, jj]) == pytest.approx(
                    1.0, rel=1e-5
                ), "Vector should be unitary"
                pass
            pass
        pass
    pass


def test_gauss_legendre_trapezoidal_1d() -> None:
    """
    First test for the routine
    biosspheres.quadratures.gauss_legendre_trapezoidal_1d

    """
    list_big_l_c = np.array([0, 1, 2, 3, 5, 10])
    for big_l_c in list_big_l_c:
        final_length, total_weights, pre_vector = (
            quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
        )
        assert len(pre_vector[0, :]) == final_length
        assert len(pre_vector[:, 0]) == 3
        assert len(total_weights) == final_length
        for ii in np.arange(0, final_length):
            assert np.linalg.norm(pre_vector[:, ii]) == pytest.approx(
                1.0, rel=1e-5
            ), "Vector should be unitary"
            pass
        pass
    pass


def test_gauss_legendre_trapezoidal_real_sh_mapping_2d() -> None:
    big_l = 1
    big_l_c = 50
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(big_l,
                                                                  big_l_c)

    quantity_theta_points_2, quantity_phi_points_2, weights_2, pre_vector_2 = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    assert quantity_phi_points == quantity_phi_points_2
    assert quantity_theta_points == quantity_theta_points_2
    assert len(weights) == quantity_theta_points
    assert len(pre_vector[:, 0, 0]) == 3
    assert len(pre_vector[0, :, 0]) == quantity_theta_points
    assert len(pre_vector[0, 0, :]) == quantity_phi_points
    for ii in np.arange(0, quantity_theta_points):
        assert weights == pytest.approx(weights_2)
        for jj in np.arange(0, quantity_phi_points):
            assert np.linalg.norm(pre_vector[:, ii, jj]) == pytest.approx(
                np.linalg.norm(pre_vector_2[:, ii, jj]), rel=1e-5
            ), "Vector should be unitary"
            pass
        pass
    pass
    pass


def overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d() -> None:
    big_l = 1
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    big_l_c = 50
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )
    pass


def overview_real_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.real_spherical_harmonic_transform_1d(big_l, big_l_c)
    )

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(big_l,
                                                                  big_l_c)

    integral = np.sum(
        spherical_harmonics[0, :, :].flatten("F") * transform[0, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten("F") * transform[1, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten("F") * transform[2, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten("F") * transform[3, :]
    )
    print(integral)
    pass


def overview_complex_spherical_harmonic_transform_1d() -> None:
    big_l = 1
    big_l_c = 2
    final_length, pre_vector, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )

    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = pes_y_kus(big_l)
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_complex_sh_mapping_2d(
        big_l, big_l_c, pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q
    )

    integral = np.sum(
        spherical_harmonics[0, :, :].flatten("F") * transform[0, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[1, :, :].flatten("F") * transform[1, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[2, :, :].flatten("F") * transform[2, :]
    )
    print(integral)
    integral = np.sum(
        spherical_harmonics[3, :, :].flatten("F") * transform[3, :]
    )
    print(integral)
    pass


def overview_from_sphere_s_cartesian_to_j_spherical_2d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1

    big_l_c = 10
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    r_coord, phi_coord, cos_theta_coord = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_2d(
            radio_2,
            p_1,
            p_2,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector,
        )
    )
    pass


def overview_from_sphere_s_cartesian_to_j_spherical_1d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1

    big_l_c = 10
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
    )
    r_coord, phi_coord, cos_theta_coord = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_1d(
            radio_2, p_1, p_2, final_length, pre_vector
        )
    )
    pass


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d() -> None:
    radio_1 = 1.2
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1
    big_l_c = 10
    quantity_theta_points, quantity_phi_points, weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    (
        r_coord,
        phi_coord,
        cos_theta_coord,
        er_times_n,
        etheta_times_n,
        ephi_times_n,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
        radio_2,
        p_1,
        p_2,
        quantity_theta_points,
        quantity_phi_points,
        pre_vector,
    )
    pass


def from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d() -> None:
    radio_2 = 1.7
    p_1 = np.asarray([2.0, 1.0, 2.5])
    p_2 = -p_1
    big_l_c = 10
    final_length, total_weights, pre_vector = (
        quadratures.gauss_legendre_trapezoidal_1d(big_l_c)
    )
    (
        r_coord,
        phi_coord,
        cos_theta_coord,
        er_times_n,
        etheta_times_n,
        ephi_times_n,
    ) = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
        radio_2, p_1, p_2, final_length, pre_vector
    )
    pass


if __name__ == "__main__":
    test_gauss_legendre_trapezoidal_2d()
    test_gauss_legendre_trapezoidal_1d()
    test_gauss_legendre_trapezoidal_real_sh_mapping_2d()
    overview_gauss_legendre_trapezoidal_complex_sh_mapping_2d()
    overview_real_spherical_harmonic_transform_1d()
    overview_complex_spherical_harmonic_transform_1d()
    overview_from_sphere_s_cartesian_to_j_spherical_2d()
    overview_from_sphere_s_cartesian_to_j_spherical_1d()
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d()
    from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d()
