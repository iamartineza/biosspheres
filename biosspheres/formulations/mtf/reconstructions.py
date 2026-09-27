from typing import Callable
import numpy as np
import scipy.special as special
import pyshtools


def rf_laplace_one_sphere_azimuthal_symmetry(
    vector: np.ndarray,
    rs: np.ndarray,
    ps: list[np.ndarray],
    big_l: int,
    coefficients: np.ndarray,
) -> float:
    """
    Reconstruction of the function u using the representation formula
    and the coefficients from solving the Laplace transmission problem
    with the mtf formulation.

    Parameters
    ----------
    vector: np.ndarray
        Length 3, representing the x, y and z coordinates in the Cartesian
        coordinate system.
    rs: np.ndarray
        with the radii of the sphere. Should be of length one.
    ps: list
        of np.ndarray
    big_l: int
    coefficients: np.ndarray
        traces of the function.

    Returns
    -------
    u: float
        evaluation of the function u in the given vector.
    """
    n = 1

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1

    u = 0.0
    while num < n:
        aux = vector - ps[num]
        r_aux = np.linalg.norm(aux)
        if r_aux < rs[num]:
            sphere = num + 1
            num = n
        pass
        num = num + 1
        pass

    if sphere == 0:
        for num in np.arange(0, n):
            aux = vector - ps[num]
            r = np.linalg.norm(aux)
            z = aux[2]

            cos_theta = z / r

            legendre_function = pyshtools.legendre.PlON(big_l, cos_theta)
            ratio = rs[num] / r
            u_temp = np.sum(
                ratio**ele_plus_1
                * (
                    eles * coefficients[eles + 2 * (big_l + 1) * num]
                    + rs[num] * coefficients[eles + (big_l + 1) * (2 * num + 1)]
                )
                * legendre_function
                / eles_times_2_plus_1
            )
            u = u + u_temp
            pass
        return u
    else:
        aux = vector - ps[sphere - 1]
        r = np.linalg.norm(aux)
        z = aux[2]

        cos_theta = z / r

        legendre_function = pyshtools.legendre.PlON(big_l, cos_theta)
        ratio = r / rs[sphere - 1]
        u = np.sum(
            ratio**eles
            * (
                ele_plus_1
                * coefficients[eles + 2 * (big_l + 1) * (sphere - 1 + n)]
                + rs[sphere - 1]
                * coefficients[eles + (big_l + 1) * (2 * (sphere - 1 + n) + 1)]
            )
            * legendre_function
            / eles_times_2_plus_1
        )
        return u
    return u


def rf_laplace_n_spheres(
    vector: np.ndarray,
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    big_l: int,
    coefficients: np.ndarray,
) -> float:

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    l_square_plus_l = ele_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    big_el_plus_1_square = (big_l + 1) ** 2

    u = 0.0

    sphere = 0
    num = 0

    while num < n:
        aux = vector - positions[num]
        r_aux = np.linalg.norm(aux)
        if r_aux < radii[num]:
            sphere = num + 1
            num = n
        pass
        num += 1
        pass

    if sphere == 0:
        for num in np.arange(0, n):
            aux = vector - positions[num]
            r = np.linalg.norm(aux)
            xx = aux[0]
            yy = aux[1]
            zz = aux[2]

            cos_theta = zz / r
            phi = np.arctan2(yy, xx)

            legendre_function = pyshtools.legendre.PlmON(
                big_l, cos_theta, csphase=-1, cnorm=0
            )
            cos_m_phi = np.cos(eles[1 : len(eles)] * phi)
            sin_m_phi = np.sin(eles[1 : len(eles)] * phi)

            ratio = radii[num] / r
            u_temp = 0.0
            for el in np.arange(0, big_l + 1):
                temp = (
                    el
                    * coefficients[
                        l_square_plus_l[el] + 2 * big_el_plus_1_square * num
                    ]
                    + radii[num]
                    * coefficients[
                        l_square_plus_l[el]
                        + big_el_plus_1_square * (2 * num + 1)
                    ]
                )
                temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
                for m in np.arange(1, el + 1):
                    temp_plus_m = (
                        el
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + radii[num]
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_plus_m *= cos_m_phi[m - 1]
                    temp_minus_m = (
                        el
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + radii[num]
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_minus_m *= sin_m_phi[m - 1]
                    temp += (temp_minus_m + temp_plus_m) * legendre_function[
                        l_times_l_plus_l_divided_by_2[el] + m
                    ]
                    pass
                temp *= ratio ** ele_plus_1[el] / eles_times_2_plus_1[el]
                u_temp += temp
                pass
            u += u_temp
            pass
        return u
    else:
        aux = vector - positions[sphere - 1]
        r = np.linalg.norm(aux)
        xx = aux[0]
        yy = aux[1]
        zz = aux[2]

        cos_theta = zz / r
        phi = np.arctan2(yy, xx)

        legendre_function = pyshtools.legendre.PlmON(
            big_l, cos_theta, csphase=-1, cnorm=0
        )
        cos_m_phi = np.cos(eles[1 : len(eles)] * phi)
        sin_m_phi = np.sin(eles[1 : len(eles)] * phi)

        ratio = r / radii[sphere - 1]
        u = 0.0
        for el in np.arange(0, big_l + 1):
            temp = (
                ele_plus_1[el]
                * coefficients[
                    l_square_plus_l[el]
                    + 2 * big_el_plus_1_square * (sphere - 1 + n)
                ]
                + radii[sphere - 1]
                * coefficients[
                    l_square_plus_l[el]
                    + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                ]
            )
            temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
            for m in np.arange(1, el + 1):
                temp_plus_m = (
                    ele_plus_1[el]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + radii[sphere - 1]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_minus_m = (
                    ele_plus_1[el]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + radii[sphere - 1]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_plus_m *= cos_m_phi[m - 1]
                temp_minus_m *= sin_m_phi[m - 1]
                temp += (temp_minus_m + temp_plus_m) * legendre_function[
                    l_times_l_plus_l_divided_by_2[el] + m
                ]
                pass
            temp *= ratio**el / eles_times_2_plus_1[el]
            u += temp
            pass
        return u
    return u


def rf_laplace_n_spheres_plus_ex_function(
    vector: np.ndarray,
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    big_l: int,
    coefficients: np.ndarray,
    exterior: Callable[[np.ndarray], float],
) -> float:

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    l_square_plus_l = ele_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    big_el_plus_1_square = (big_l + 1) ** 2

    u = 0.0

    sphere = 0
    num = 0

    while num < n:
        aux = vector - positions[num]
        r_aux = np.linalg.norm(aux)
        if r_aux < radii[num]:
            sphere = num + 1
            num = n
        pass
        num += 1
        pass

    if sphere == 0:
        for num in np.arange(0, n):
            aux = vector - positions[num]
            r = np.linalg.norm(aux)
            xx = aux[0]
            yy = aux[1]
            zz = aux[2]

            cos_theta = zz / r
            phi = np.arctan2(yy, xx)

            legendre_function = pyshtools.legendre.PlmON(
                big_l, cos_theta, csphase=-1, cnorm=0
            )
            cos_m_phi = np.cos(eles[1 : len(eles)] * phi)
            sin_m_phi = np.sin(eles[1 : len(eles)] * phi)

            ratio = radii[num] / r
            u_temp = 0.0
            for el in np.arange(0, big_l + 1):
                temp = (
                    el
                    * coefficients[
                        l_square_plus_l[el] + 2 * big_el_plus_1_square * num
                    ]
                    + radii[num]
                    * coefficients[
                        l_square_plus_l[el]
                        + big_el_plus_1_square * (2 * num + 1)
                    ]
                )
                temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
                for m in np.arange(1, el + 1):
                    temp_plus_m = (
                        el
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + radii[num]
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_plus_m *= cos_m_phi[m - 1]
                    temp_minus_m = (
                        el
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + radii[num]
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_minus_m *= sin_m_phi[m - 1]
                    temp += (temp_minus_m + temp_plus_m) * legendre_function[
                        l_times_l_plus_l_divided_by_2[el] + m
                    ]
                    pass
                temp *= ratio ** ele_plus_1[el] / eles_times_2_plus_1[el]
                u_temp += temp
                pass
            u += u_temp
            pass
        return u + exterior(vector)
    else:
        aux = vector - positions[sphere - 1]
        r = np.linalg.norm(aux)
        xx = aux[0]
        yy = aux[1]
        zz = aux[2]

        cos_theta = zz / r
        phi = np.arctan2(yy, xx)

        legendre_function = pyshtools.legendre.PlmON(
            big_l, cos_theta, csphase=-1, cnorm=0
        )
        cos_m_phi = np.cos(eles[1 : len(eles)] * phi)
        sin_m_phi = np.sin(eles[1 : len(eles)] * phi)

        ratio = r / radii[sphere - 1]
        u = 0.0
        for el in np.arange(0, big_l + 1):
            temp = (
                ele_plus_1[el]
                * coefficients[
                    l_square_plus_l[el]
                    + 2 * big_el_plus_1_square * (sphere - 1 + n)
                ]
                + radii[sphere - 1]
                * coefficients[
                    l_square_plus_l[el]
                    + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                ]
            )
            temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
            for m in np.arange(1, el + 1):
                temp_plus_m = (
                    ele_plus_1[el]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + radii[sphere - 1]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_minus_m = (
                    ele_plus_1[el]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + radii[sphere - 1]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_plus_m *= cos_m_phi[m - 1]
                temp_minus_m *= sin_m_phi[m - 1]
                temp += (temp_minus_m + temp_plus_m) * legendre_function[
                    l_times_l_plus_l_divided_by_2[el] + m
                ]
                pass
            temp *= ratio**el / eles_times_2_plus_1[el]
            u += temp
            pass
        return u
    return u


def rf_laplace_n_spheres_call(
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    big_l: int,
    coefficients: np.ndarray,
) -> Callable[[np.ndarray], float]:
    def u_rf(vector):
        return rf_laplace_n_spheres(
            vector, n, radii, positions, big_l, coefficients
        )

    return u_rf


def rf_laplace_n_spheres_plus_ex_function_call(
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    big_l: int,
    coefficients: np.ndarray,
    exterior: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    def u_rf(vector):
        return rf_laplace_n_spheres(
            vector, n, radii, positions, big_l, coefficients
        ) + exterior(vector)

    return u_rf


def complex_spherical_harmonics_at_points(
    big_l: int, cos_theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Returns the complex spherical harmonics of degree <= big_l, the basis
    of the Helmholtz routines, evaluated at the points with the given
    spherical coordinates.

    Returns
    -------
    spherical_harmonics : np.ndarray
        Shape ((big_l + 1)**2, len(phi)).
        spherical_harmonics[l(l+1) + m] = Y_{l,m}, with
        Y_{l,-m} = (-1)**m conj(Y_{l,m}).
    """
    legendre = np.empty(((big_l + 1) * (big_l + 2) // 2, len(phi)))
    for i in np.arange(0, len(phi)):
        legendre[:, i] = pyshtools.legendre.PlmON(
            big_l, cos_theta[i], csphase=-1, cnorm=1
        )
    spherical_harmonics = np.empty(
        ((big_l + 1) ** 2, len(phi)), dtype=np.complex128
    )
    for el in np.arange(0, big_l + 1):
        for m in np.arange(0, el + 1):
            y_lm = legendre[el * (el + 1) // 2 + m] * np.exp(1j * m * phi)
            spherical_harmonics[el * (el + 1) + m] = y_lm
            spherical_harmonics[el * (el + 1) - m] = (-1) ** m * np.conj(y_lm)
    return spherical_harmonics


def rf_helmholtz_n_spheres(
    points: np.ndarray,
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    kii: np.ndarray,
    big_l: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Evaluates with the representation formula the solution of a
    Helmholtz transmission problem solved with the MTF.

    Outside the spheres it returns the scattered field u_0; inside the
    sphere j it returns u_j.

    Parameters
    ----------
    points : np.ndarray
        Shape (number of points, 3).
    n : int
        >= 1, number of spheres.
    radii : np.ndarray
        Length n.
    positions : list[np.ndarray]
        Centers of the spheres.
    kii : np.ndarray
        Length n + 1. kii[0] is the exterior wave number, kii[j] the one
        of sphere j.
    big_l : int
        >= 0, max degree.
    coefficients : np.ndarray
        Solution of the MTF system, length 4 * n * (big_l + 1)**2, in
        the order of mtf.mtf_n_matrix.

    Returns
    -------
    u : np.ndarray
        Complex, length number of points.
    """
    num = (big_l + 1) ** 2
    eles = np.arange(0, big_l + 1)
    eles_repeated = np.repeat(eles, 2 * eles + 1)
    points = np.atleast_2d(points)
    owner = np.full(len(points), -1)
    for j in np.arange(0, n):
        rho = np.linalg.norm(points - np.asarray(positions[j]), axis=1)
        owner[rho < radii[j]] = j
    u = np.zeros(len(points), dtype=np.complex128)
    for j in np.arange(0, n):
        for interior in [False, True]:
            region = owner == j if interior else owner == -1
            if not np.any(region):
                continue
            wave_number = kii[j + 1] if interior else kii[0]
            x = points[region] - np.asarray(positions[j])
            r = np.linalg.norm(x, axis=1)
            cos_theta = np.divide(x[:, 2], r, out=np.ones_like(r), where=r > 0)
            sh = complex_spherical_harmonics_at_points(
                big_l, cos_theta, np.arctan2(x[:, 1], x[:, 0])
            )
            kr = wave_number * radii[j]
            j_l = special.spherical_jn(eles, kr)[eles_repeated]
            j_lp = special.spherical_jn(eles, kr, derivative=True)[
                eles_repeated
            ]
            y_l = special.spherical_yn(eles, kr)[eles_repeated]
            y_lp = special.spherical_yn(eles, kr, derivative=True)[
                eles_repeated
            ]
            start = 2 * num * (n + j) if interior else 2 * num * j
            d = coefficients[start : start + num]
            d_n = coefficients[start + num : start + 2 * num]
            kr_points = wave_number * r[np.newaxis, :]
            radial = special.spherical_jn(eles[:, np.newaxis], kr_points)
            if interior:
                weights = (j_l + 1j * y_l) * d_n - wave_number * (
                    j_lp + 1j * y_lp
                ) * d
            else:
                weights = wave_number * j_lp * d + j_l * d_n
                radial = radial + 1j * special.spherical_yn(
                    eles[:, np.newaxis], kr_points
                )
            u[region] += (
                1j
                * wave_number
                * radii[j] ** 2
                * np.sum(
                    weights[:, np.newaxis] * radial[eles_repeated] * sh, axis=0
                )
            )
    return u
