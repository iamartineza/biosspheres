"""
Tests for biosspheres.laplace.crossinteractions: transposition
identities, relations between V, K, K* and W, the assembly of the
Calderón cross blocks, and 1D against 2D quadrature routines.
"""

import functools
import pytest
import numpy as np
import biosspheres.laplace.crossinteractions as crossinteractions
import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes

BIG_L_C = 25
GEOMETRIES = [
    (3, 3.0, 2.0, (2.0, 3.0, 4.0)),
    (2, 1.2, 3.0, (2.0, 3.0, 4.0)),
    (3, 1.3, 1.7, (2.0, 3.0, 4.0)),
    (5, 3.0, 2.0, (2.0, 3.0, 4.0)),
    (5, 1.0, 2.0, (0.0, 0.0, 2.5)),
]


def relative_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    """
    Returns the max-norm of the difference relative to the max-norm of
    the reference.
    """
    return np.max(np.abs(approximation - reference)) / np.max(np.abs(reference))


def one_direction_1d(
    big_l: int,
    r_j: float,
    r_s: float,
    p_j: np.ndarray,
    p_s: np.ndarray,
) -> dict:
    """
    V_{s,j}^0, K_{s,j}^0, K_{s,j}^{*0} and A_{s,j}^0, A_{j,s}^0 with the
    1D quadrature routines.
    """
    final_length, pre_vector_t, transform = (
        quadratures.real_spherical_harmonic_transform_1d(big_l, BIG_L_C)
    )
    r_coord, phi_coord, cos_theta_coord, er, eth, ephi = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_1d(
            r_s, p_j, p_s, final_length, pre_vector_t
        )
    )
    arguments = (
        big_l,
        r_j,
        r_s,
        r_coord,
        phi_coord,
        cos_theta_coord,
    )
    a_sj, a_js = crossinteractions.a_0_sj_and_js_v1d(
        *arguments,
        final_length,
        transform,
        auxindexes.diagonal_l_dense(big_l),
    )
    return {
        "v": crossinteractions.v_0_sj_semi_analytic_v1d(
            *arguments, final_length, transform
        ),
        "k": crossinteractions.k_0_sj_semi_analytic_v1d(
            *arguments, final_length, transform
        ),
        "ka": crossinteractions.ka_0_sj_semi_analytic_recurrence_v1d(
            *arguments, er, eth, ephi, final_length, transform
        ),
        "a_sj": a_sj,
        "a_js": a_js,
    }


def one_direction_2d(
    big_l: int,
    r_j: float,
    r_s: float,
    p_j: np.ndarray,
    p_s: np.ndarray,
) -> dict:
    """
    V_{s,j}^0, K_{s,j}^0, K_{s,j}^{*0} and A_{s,j}^0 with the 2D
    quadrature routines.
    """
    quantity_theta_points, quantity_phi_points, weights, pre_vector_t = (
        quadratures.gauss_legendre_trapezoidal_2d(BIG_L_C)
    )
    r_coord, phi_coord, cos_theta_coord, er, eth, ephi = (
        quadratures.from_sphere_s_cartesian_to_j_spherical_and_spherical_vectors_2d(
            r_s,
            p_j,
            p_s,
            quantity_theta_points,
            quantity_phi_points,
            pre_vector_t,
        )
    )
    geometry = (big_l, r_j, r_s, r_coord, phi_coord, cos_theta_coord)
    quadrature = (
        weights,
        pre_vector_t[2, :, 0],
        quantity_theta_points,
        quantity_phi_points,
        *auxindexes.pes_y_kus(big_l),
    )
    a_sj, a_js = crossinteractions.a_0_sj_and_js_v2d(
        *geometry, *quadrature, auxindexes.diagonal_l_dense(big_l)
    )
    return {
        "v": crossinteractions.v_0_sj_semi_analytic_v2d(*geometry, *quadrature),
        "k": crossinteractions.k_0_sj_semi_analytic_v2d(*geometry, *quadrature),
        "ka": crossinteractions.ka_0_sj_semi_analytic_recurrence_v2d(
            *geometry, er, eth, ephi, *quadrature
        ),
        "a_sj": a_sj,
        "a_js": a_js,
    }


@functools.lru_cache(maxsize=None)
def setup(big_l: int, r_1: float, r_2: float, p_1: tuple) -> dict:
    """
    Operators between sphere 1 (radius r_1, center p_1) and sphere 2
    (radius r_2, center -p_1). Key "21" holds s = 2, j = 1 and key "12"
    holds s = 1, j = 2.
    """
    p_1 = np.asarray(p_1)
    p_2 = -p_1
    return {
        "big_l": big_l,
        "r_1": r_1,
        "r_2": r_2,
        "el": auxindexes.diagonal_l_sparse(big_l),
        "1d": {
            "21": one_direction_1d(big_l, r_1, r_2, p_1, p_2),
            "12": one_direction_1d(big_l, r_2, r_1, p_2, p_1),
        },
        "2d": {
            "21": one_direction_2d(big_l, r_1, r_2, p_1, p_2),
            "12": one_direction_2d(big_l, r_2, r_1, p_2, p_1),
        },
    }


def calderon_blocks(
    data: dict, r_j: float, r_s: float, el_diagonal
) -> np.ndarray:
    """
    Assembles [[-K, V], [W, K*]] from V_{s,j}^0 and K_{s,j}^0, with K*
    and W obtained from V.
    """
    ka = crossinteractions.ka_0_sj_from_v_sj(data["v"], r_s, el_diagonal)
    w = -crossinteractions.k_0_sj_from_v_0_sj(ka, r_j, el_diagonal)
    return np.block([[-data["k"], data["v"]], [w, ka]])


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
@pytest.mark.parametrize("version", ["1d", "2d"])
def test_v_transpose(big_l, r_1, r_2, p_1, version) -> None:
    """
    V_{1,2}^0 = (V_{2,1}^0)^t, and v_0_js_from_v_0_sj applies it.
    """
    data = setup(big_l, r_1, r_2, p_1)[version]
    v_21 = crossinteractions.v_0_js_from_v_0_sj(data["12"]["v"])
    assert relative_error(v_21, data["21"]["v"]) < 3e-13
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
@pytest.mark.parametrize("version", ["1d", "2d"])
def test_k_from_v(big_l, r_1, r_2, p_1, version) -> None:
    """
    K_{2,1}^0 = V_{2,1}^0 diag(-l / r_1), explicitly and through
    k_0_sj_from_v_0_sj.
    """
    all_data = setup(big_l, r_1, r_2, p_1)
    data = all_data[version]["21"]
    eles = np.arange(0, big_l + 1)
    explicit = data["v"] @ np.diag(np.repeat(eles / -r_1, 2 * eles + 1))
    routine = crossinteractions.k_0_sj_from_v_0_sj(
        data["v"], r_1, all_data["el"]
    )
    assert relative_error(explicit, data["k"]) < 5e-14
    assert relative_error(routine, data["k"]) < 5e-14
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
@pytest.mark.parametrize("version", ["1d", "2d"])
def test_ka_from_k_and_from_v(big_l, r_1, r_2, p_1, version) -> None:
    """
    K_{2,1}^{*0} = (K_{1,2}^0)^t and K_{2,1}^{*0} = diag(-p / r_2)
    V_{2,1}^0.
    """
    all_data = setup(big_l, r_1, r_2, p_1)
    data = all_data[version]
    from_k = crossinteractions.ka_0_sj_from_k_js(data["12"]["k"])
    from_v = crossinteractions.ka_0_sj_from_v_sj(
        data["21"]["v"], r_2, all_data["el"]
    )
    assert relative_error(from_k, data["21"]["ka"]) < 1e-12
    assert relative_error(from_v, data["21"]["ka"]) < 1e-11
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
def test_w_transpose(big_l, r_1, r_2, p_1) -> None:
    """
    W_{1,2}^0 = (W_{2,1}^0)^t, with both obtained from V.
    """
    all_data = setup(big_l, r_1, r_2, p_1)
    data = all_data["1d"]
    el_diagonal = all_data["el"]
    w_12 = crossinteractions.w_0_sj_from_v_sj(
        data["12"]["v"], r_2, r_1, el_diagonal
    )
    w_21 = crossinteractions.w_0_sj_from_v_sj(
        data["21"]["v"], r_1, r_2, el_diagonal
    )
    assert relative_error(w_12, w_21.T) < 5e-12
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
def test_w_from_ka_and_from_v(big_l, r_1, r_2, p_1) -> None:
    """
    W_{2,1}^0 obtained from K_{2,1}^{*0} equals the one obtained
    directly from V_{2,1}^0.
    """
    all_data = setup(big_l, r_1, r_2, p_1)
    v_21 = all_data["1d"]["21"]["v"]
    el_diagonal = all_data["el"]
    ka_21 = crossinteractions.ka_0_sj_from_v_sj(v_21, r_2, el_diagonal)
    from_ka = -crossinteractions.k_0_sj_from_v_0_sj(ka_21, r_1, el_diagonal)
    from_v = crossinteractions.w_0_sj_from_v_sj(v_21, r_1, r_2, el_diagonal)
    assert relative_error(from_ka, from_v) < 2e-14
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
@pytest.mark.parametrize("version", ["1d", "2d"])
def test_calderon_build(big_l, r_1, r_2, p_1, version) -> None:
    """
    A_{2,1}^0 and A_{1,2}^0 returned by a_0_sj_and_js equal the blocks
    assembled from the V and K of each direction.
    """
    all_data = setup(big_l, r_1, r_2, p_1)
    data = all_data[version]
    el_diagonal = all_data["el"]
    a_21 = calderon_blocks(data["21"], r_1, r_2, el_diagonal)
    a_12 = calderon_blocks(data["12"], r_2, r_1, el_diagonal)
    assert relative_error(data["21"]["a_sj"], a_21) < 3e-15
    assert relative_error(data["21"]["a_js"], a_12) < 1e-12
    pass


@pytest.mark.parametrize("big_l, r_1, r_2, p_1", GEOMETRIES)
@pytest.mark.parametrize("name", ["v", "k", "ka", "a_sj", "a_js"])
def test_1d_vs_2d(big_l, r_1, r_2, p_1, name) -> None:
    """
    The 1D and 2D quadrature routines give the same matrices.
    """
    data = setup(big_l, r_1, r_2, p_1)
    assert (
        relative_error(data["2d"]["21"][name], data["1d"]["21"][name]) < 5e-14
    )
    pass


@pytest.mark.parametrize("big_l, big_l_c", [(3, 25), (3, 50)])
def test_all_cross_1d_vs_2d(big_l, big_l_c) -> None:
    """
    all_cross_interactions_n_spheres with 1D and 2D quadratures agree.
    """
    radii = np.asarray([3.0, 2.0, 1.0])
    p_1 = np.asarray([2.0, 3.0, 4.0])
    centers = [p_1, -p_1, np.asarray([6.0, -5.0, 0.0])]
    cross_1d = crossinteractions.all_cross_interactions_n_spheres_v1d(
        len(radii), big_l, big_l_c, radii, centers
    )
    cross_2d = crossinteractions.all_cross_interactions_n_spheres_v2d(
        len(radii), big_l, big_l_c, radii, centers
    )
    assert relative_error(cross_2d, cross_1d) < 2e-14
    pass
