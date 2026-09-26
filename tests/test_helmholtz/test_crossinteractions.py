"""
Tests for biosspheres.helmholtz.crossinteractions.

They check transposition identities between the operators of two
spheres, the relations between V, K, K* and W, the agreement between
the routines with 1d and 2d quadratures, and the agreement between the
routines that build all cross interactions of several spheres.

"""

import numpy as np
import pytest
import scipy.special
import biosspheres.helmholtz.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes

CONFIGURATIONS = [
    (3.0, 2.0, 7.0, 5, 25, 1e-10),
    (1.3, 1.7, 7.0, 3, 25, 3e-12),
    pytest.param((0.9, 0.7, 2.0, 10, 30, 2e-12), marks=pytest.mark.slow),
    (1.0, 1.0, 7.0, 3, 25, 5e-11),
]
ROUNDING = 1e-13


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def setup(
    radio_1: float,
    radio_2: float,
    k0: float,
    big_l: int,
    big_l_c: int,
    tolerance: float,
) -> dict:
    """
    Operators between two spheres of radii radio_1 and radio_2 centered
    at p_1 = (2, 3, 4) and p_2 = -p_1, computed with the routines with
    1d and 2d quadratures. Key "21" is sphere 1 to sphere 2, and "12"
    the other way.

    Parameters
    ----------
    radio_1 : float
        > 0, radius of sphere 1.
    radio_2 : float
        > 0, radius of sphere 2.
    k0 : float
        wave number.
    big_l : int
        >= 0, max degree.
    big_l_c : int
        >= 0, parameter of the quadratures.
    tolerance : float
        for the identities limited by the quadratures.

    Returns
    -------
    data : dict

    """
    p_1 = np.asarray([2.0, 3.0, 4.0])
    p_2 = -p_1
    eles = np.arange(0, big_l + 1)
    final_length, pre_vector_t, transform = (
        quadratures.complex_spherical_harmonic_transform_1d(big_l, big_l_c)
    )
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t_2d = (
        quadratures.gauss_legendre_trapezoidal_2d(big_l_c)
    )
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    data = {
        "gs": auxindexes.giro_sign(big_l),
        "big_l": big_l,
        "k0": k0,
        "radio_1": radio_1,
        "radio_2": radio_2,
        "tolerance": tolerance,
    }
    for key, r_j, r_s, p_j, p_s in [
        ("21", radio_1, radio_2, p_1, p_2),
        ("12", radio_2, radio_1, p_2, p_1),
    ]:
        j_l = scipy.special.spherical_jn(eles, r_j * k0)
        j_lp = scipy.special.spherical_jn(eles, r_j * k0, derivative=True)
        coordinates_1d = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            r_s, p_j, p_s, final_length, pre_vector_t
        )
        coordinates_2d = quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
            r_s,
            p_j,
            p_s,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t_2d,
        )
        tail_2d = (
            weights,
            pre_vector_t_2d[2, :, 0],
            quantity_theta_points,
            quantity_phi_points,
            pesykus,
            p2_plus_p_plus_q,
            p2_plus_p_minus_q,
        )
        data["ratio" + key] = k0 * j_lp / j_l
        data["v" + key + "_1d"] = crossinteractions.v_0_sj_semi_analytic_v1d(
            big_l,
            k0,
            r_j,
            r_s,
            j_l,
            *coordinates_1d[0:3],
            final_length,
            transform,
        )
        data["k" + key + "_1d"] = crossinteractions.k_0_sj_semi_analytic_v1d(
            big_l,
            k0,
            r_j,
            r_s,
            j_lp,
            *coordinates_1d[0:3],
            final_length,
            transform,
        )
        data["ka" + key + "_1d"] = (
            crossinteractions.ka_0_sj_semi_analytic_recurrence_v1d(
                big_l,
                k0,
                r_j,
                r_s,
                j_l,
                *coordinates_1d,
                final_length,
                transform,
            )
        )
        data["w" + key + "_1d"] = (
            crossinteractions.w_0_sj_semi_analytic_recurrence_v1d(
                big_l,
                k0,
                r_j,
                r_s,
                j_lp,
                *coordinates_1d,
                final_length,
                transform,
            )
        )
        data["v" + key + "_2d"] = crossinteractions.v_0_sj_semi_analytic_v2d(
            big_l, k0, r_j, r_s, j_l, *coordinates_2d[0:3], *tail_2d
        )
        data["k" + key + "_2d"] = crossinteractions.k_0_sj_semi_analytic_v2d(
            big_l, k0, r_j, r_s, j_lp, *coordinates_2d[0:3], *tail_2d
        )
        data["ka" + key + "_2d"] = (
            crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
                big_l, k0, r_j, r_s, j_l, *coordinates_2d, *tail_2d
            )
        )
        data["w" + key + "_2d"] = (
            crossinteractions.w_0_sj_semi_analytic_recurrence_v2d(
                big_l, k0, r_j, r_s, j_lp, *coordinates_2d, *tail_2d
            )
        )
        data["quadratures" + key + "_1d"] = (
            crossinteractions.v_k_w_0_sj_from_quadratures_1d(
                big_l,
                k0,
                r_j,
                r_s,
                j_l,
                j_lp,
                *coordinates_1d,
                final_length,
                transform,
            )
        )
        data["quadratures" + key + "_2d"] = (
            crossinteractions.v_k_w_0_sj_from_quadratures_2d(
                big_l, k0, r_j, r_s, j_l, j_lp, *coordinates_2d, *tail_2d
            )
        )
    return data


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def data(request) -> dict:
    return setup(*request.param)


@pytest.mark.parametrize("operator", ["v", "k", "ka", "w"])
@pytest.mark.parametrize("key", ["21", "12"])
def test_1d_vs_2d(data: dict, operator: str, key: str) -> None:
    assert (
        relative_error(
            data[operator + key + "_2d"], data[operator + key + "_1d"]
        )
        < ROUNDING
    )
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_v_transpose(data: dict, version: str) -> None:
    v21 = data["v21_" + version]
    v12 = data["v12_" + version]
    gs = data["gs"]
    assert relative_error(gs @ v21.T @ gs, v12) < data["tolerance"]
    assert relative_error(crossinteractions.v_0_js_from_v_0_sj(v21), v12) < (
        data["tolerance"]
    )
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_k_from_v(data: dict, version: str) -> None:
    eles = np.arange(0, len(data["ratio21"]))
    jeys = np.repeat(-data["ratio21"], 2 * eles + 1)
    assert relative_error(
        data["v21_" + version] * jeys, data["k21_" + version]
    ) < (ROUNDING)
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_ka_from_k(data: dict, version: str) -> None:
    gs = data["gs"]
    ka21 = data["ka21_" + version]
    k12 = data["k12_" + version]
    assert relative_error(gs @ ka21.T @ gs, k12) < data["tolerance"]
    assert relative_error(crossinteractions.ka_0_sj_from_k_js(k12), ka21) < (
        data["tolerance"]
    )
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_w_transpose(data: dict, version: str) -> None:
    gs = data["gs"]
    w21 = data["w21_" + version]
    w12 = data["w12_" + version]
    assert relative_error(gs @ w21.T @ gs, w12) < data["tolerance"]
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_w_from_ka(data: dict, version: str) -> None:
    w21 = crossinteractions.w_0_sj_from_ka_sj(
        data["ka21_" + version], data["k0"], data["radio_1"]
    )
    assert relative_error(w21, data["w21_" + version]) < ROUNDING
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_quadratures_routines(data: dict, version: str) -> None:
    for computed, operator in zip(
        data["quadratures21_" + version], ["v", "k", "ka", "w"]
    ):
        assert relative_error(computed, data[operator + "21_" + version]) < (
            ROUNDING
        )
        pass
    pass


@pytest.mark.parametrize("version", ["1d", "2d"])
def test_calderon_from_v_vs_blocks(data: dict, version: str) -> None:
    radio_1, radio_2, k0 = data["radio_1"], data["radio_2"], data["k0"]
    gs = data["gs"]
    big_l = data["big_l"]
    v21 = data["v21_" + version]
    a_21, a_12 = crossinteractions.a_0_sj_and_js_from_v_sj(
        big_l, v21, data["ratio21"], data["ratio12"], gs
    )
    k21 = crossinteractions.k_0_sj_from_v_0_sj(v21, k0, radio_1)
    v12 = crossinteractions.v_0_js_from_v_0_sj(v21)
    k12 = crossinteractions.k_0_sj_from_v_0_sj(v12, k0, radio_2)
    ka21 = crossinteractions.ka_0_sj_from_k_js(k12)
    w21 = crossinteractions.w_0_sj_from_ka_sj(ka21, k0, radio_1)
    a_21_blocks = np.block([[-k21, v21], [w21, ka21]])
    assert relative_error(a_21, a_21_blocks) < ROUNDING
    a_21_direct, a_12_direct = crossinteractions.a_0_sj_and_js_from_v_k_w(
        *data["quadratures21_" + version], gs
    )
    assert relative_error(a_21, a_21_direct) < data["tolerance"]
    assert relative_error(a_12, a_12_direct) < data["tolerance"]
    pass


N_SPHERES_CONFIGURATIONS = [
    (5, 28, 7.0, 1.112, 5e-12),
    (4, 20, 2.0, 0.8, 2e-12),
]


@pytest.mark.parametrize(
    "big_l, big_l_c, k0, radius, tolerance", N_SPHERES_CONFIGURATIONS
)
def test_all_cross_interactions_versions(
    big_l: int, big_l_c: int, k0: float, radius: float, tolerance: float
) -> None:
    n = 3
    radii = np.ones(n) * radius
    center_positions = [
        np.asarray([0.0, 0.0, 0]),
        np.asarray([-7.0, -3.0, -2.0]),
        np.asarray([3.0, 5.0, 7.0]),
    ]
    eles = np.arange(0, big_l + 1)
    j_l = np.empty((n, big_l + 1))
    j_lp = np.empty((n, big_l + 1))
    for j in np.arange(0, n):
        j_l[j, :] = scipy.special.spherical_jn(eles, radii[j] * k0)
        j_lp[j, :] = scipy.special.spherical_jn(
            eles, radii[j] * k0, derivative=True
        )
        pass
    from_v_1d = crossinteractions.all_cross_interactions_n_spheres_from_v_1d(
        n, big_l, big_l_c, k0, radii, center_positions, j_l, j_lp
    )
    from_v_2d = crossinteractions.all_cross_interactions_n_spheres_from_v_2d(
        n, big_l, big_l_c, k0, radii, center_positions, j_l, j_lp
    )
    direct_1d = crossinteractions.all_cross_interactions_n_spheres_1d(
        n, big_l, big_l_c, k0, radii, center_positions
    )
    direct_2d = crossinteractions.all_cross_interactions_n_spheres_2d(
        n, big_l, big_l_c, k0, radii, center_positions
    )
    assert relative_error(from_v_2d, from_v_1d) < 1e-13
    assert relative_error(direct_1d, from_v_1d) < tolerance
    assert relative_error(direct_2d, direct_1d) < 1e-13
    pass
