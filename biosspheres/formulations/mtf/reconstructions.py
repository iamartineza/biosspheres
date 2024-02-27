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


def rf_helmholtz_n_spheres(
    vector: np.ndarray,
    n: int,
    radii: np.ndarray,
    positions: list[np.ndarray],
    kii: np.ndarray,
    big_l: int,
    coefficients: np.ndarray,
) -> np.complex128:

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    l_square_plus_l = ele_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    big_el_plus_1_square = (big_l + 1) ** 2

    u = 0.0 * 1j

    sphere = 0
    num = 0

    j_l = np.empty(big_l + 1)
    j_lp = np.empty_like(j_l)
    h_l = np.empty(big_l + 1, dtype=np.complex128)
    h_lp = np.empty_like(h_l)

    def leg_exp(cos_the, ph):
        leg = pyshtools.legendre.PlmON(big_l, cos_the, csphase=-1, cnorm=1)
        exp_m_p = np.exp(eles[1 : len(eles)] * ph)
        exp_m_n = (-1.0) ** eles[1 : len(eles)] / exp_m_pos
        return leg, exp_m_p, exp_m_n

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

            legendre_function, exp_m_pos, exp_m_neg = leg_exp(cos_theta, phi)

            h_l[:] = special.spherical_jn(
                eles, kii[0] * r
            ) + 1j * special.spherical_yn(eles, kii[0] * r)
            j_l[:] = special.spherical_jn(eles, kii[0] * radii[num])
            j_lp[:] = special.spherical_jn(
                eles, kii[0] * radii[num], derivative=True
            )
            u_temp = 0.0 * 1j
            for el in np.arange(0, big_l + 1):
                temp = (
                    kii[0]
                    * j_lp[el]
                    * coefficients[
                        l_square_plus_l[el] + 2 * big_el_plus_1_square * num
                    ]
                    + j_l[el]
                    * coefficients[
                        l_square_plus_l[el]
                        + big_el_plus_1_square * (2 * num + 1)
                    ]
                )
                temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
                for m in np.arange(1, el + 1):
                    temp_plus_m = (
                        kii[0]
                        * j_lp[el]
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + j_l[el]
                        * coefficients[
                            l_square_plus_l[el]
                            + m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_plus_m *= exp_m_pos[m - 1]
                    temp_minus_m = (
                        kii[0]
                        * j_lp[el]
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + 2 * big_el_plus_1_square * num
                        ]
                        + j_l[el]
                        * coefficients[
                            l_square_plus_l[el]
                            - m
                            + big_el_plus_1_square * (2 * num + 1)
                        ]
                    )
                    temp_minus_m *= exp_m_neg[m - 1]
                    temp += (temp_minus_m + temp_plus_m) * legendre_function[
                        l_times_l_plus_l_divided_by_2[el] + m
                    ]
                    pass
                temp *= h_l[el]
                u_temp += temp
                pass
            u += 1j * kii[0] * radii[num] ** 2 * u_temp
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

        legendre_function, exp_m_pos, exp_m_neg = leg_exp(cos_theta, phi)

        j_l[:] = special.spherical_jn(eles, kii[sphere] * r)
        h_l[:] = special.spherical_jn(
            eles, kii[sphere] * radii[sphere - 1]
        ) + 1j * special.spherical_yn(eles, kii[sphere] * radii[sphere - 1])
        h_lp[:] = special.spherical_jn(
            eles, kii[sphere] * radii[sphere - 1], derivative=True
        ) + 1j * special.spherical_yn(
            eles, kii[sphere] * radii[sphere - 1], derivative=True
        )
        for el in np.arange(0, big_l + 1):
            temp = (
                -kii[sphere]
                * h_lp[el]
                * coefficients[
                    l_square_plus_l[el]
                    + 2 * big_el_plus_1_square * (sphere - 1 + n)
                ]
                + h_l[el]
                * coefficients[
                    l_square_plus_l[el]
                    + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                ]
            )
            temp *= legendre_function[l_times_l_plus_l_divided_by_2[el]]
            for m in np.arange(1, el + 1):
                temp_plus_m = (
                    -kii[sphere]
                    * h_lp[el]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + h_l[el]
                    * coefficients[
                        l_square_plus_l[el]
                        + m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_minus_m = (
                    -kii[sphere]
                    * h_lp[el]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + 2 * big_el_plus_1_square * (sphere - 1 + n)
                    ]
                    + h_l[el]
                    * coefficients[
                        l_square_plus_l[el]
                        - m
                        + big_el_plus_1_square * (2 * (sphere - 1 + n) + 1)
                    ]
                )
                temp_plus_m *= exp_m_pos[m - 1]
                temp_minus_m *= exp_m_neg[m - 1]
                temp += (temp_minus_m + temp_plus_m) * legendre_function[
                    l_times_l_plus_l_divided_by_2[el] + m
                ]
                pass
            temp *= j_l[el]
            u += temp
            pass
        return u
    return u
