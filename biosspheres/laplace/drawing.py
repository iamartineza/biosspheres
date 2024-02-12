from typing import Callable
import numpy as np
import pyshtools


def draw_cut_representation_formula_one_sphere_azimuthal_symmetry(
    cut: int,
    center: np.ndarray,
    horizontal: float,
    vertical: float,
    inter_horizontal: int,
    inter_vertical: int,
    coefficients: np.ndarray,
    radius: float,
    big_l: int,
    exterior: Callable[[np.ndarray], float],
) -> np.ndarray:
    """

    Parameters
    ----------
    cut : int
        1, 2 or 3. Indicates if the drawing is a parallel cut of the
        plane: xy if = 1, xz if = 2, yz = 3 if other.
    center : np.ndarray
        array of floats of length 2. Coordinates of the center of the
        drawing.
    horizontal : float
        horizontal length of the rectangle that is going to be drawn.
    vertical : float
        vertical length of the rectangle that is going to be drawn.
    inter_horizontal : int
        > 0, quantity of points in the horizontal axis.
    inter_vertical : int
        > 0, quantity of points in the vertical axis.
    coefficients : np.ndarray
        coefficients of the spherical harmonics expansion.
    radius : float
        radius of the sphere.
    big_l : int
        > 0, maximum order of the spherical harmonics used to discretize
        the traces.
    exterior : Callable[[np.ndarray], float]
        exterior function, to be evaluated in each point.

    Returns
    -------
    data_for_plotting : np.ndarray
        2D of floats for plotting.
    """
    n = 1
    ps = [np.asarray([0.0, 0.0, 0.0])]
    rs = np.asarray([radius])

    x1 = np.linspace(-horizontal / 2, horizontal / 2, inter_horizontal)
    y1 = np.linspace(-vertical / 2, vertical / 2, inter_vertical)
    data_for_plotting = np.zeros((len(y1), len(x1)))

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    if cut == 3:
        z = 0.0 + center[2]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                y = y1[jj] + center[1]

                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - ps[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < rs[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num = num + 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r

                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta
                        )
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio**ele_plus_1
                            * (
                                eles
                                * coefficients[eles + 2 * (big_l + 1) * num]
                                + rs[num]
                                * coefficients[
                                    eles + (big_l + 1) * (2 * num + 1)
                                ]
                            )
                            * legendre_function
                            / eles_times_2_plus_1
                        )
                        u = u + u_temp
                        pass
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r

                    legendre_function = pyshtools.legendre.PlON(
                        big_l, cos_theta
                    )
                    ratio = r / rs[sphere - 1]
                    u = np.sum(
                        ratio**eles
                        * (
                            ele_plus_1
                            * coefficients[
                                eles + 2 * (big_l + 1) * (sphere - 1 + n)
                            ]
                            + rs[sphere - 1]
                            * coefficients[
                                eles + (big_l + 1) * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        * legendre_function
                        / eles_times_2_plus_1
                    )
                    data_for_plotting[jj, ii] = u
                pass
            pass
        return x1, y1, data_for_plotting
    elif cut == 2:
        y = center[1]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - ps[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < rs[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num = num + 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r

                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta
                        )
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio**ele_plus_1
                            * (
                                eles
                                * coefficients[eles + 2 * (big_l + 1) * num]
                                + rs[num]
                                * coefficients[
                                    eles + (big_l + 1) * (2 * num + 1)
                                ]
                            )
                            * legendre_function
                            / eles_times_2_plus_1
                        )
                        u = u + u_temp
                        pass
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r

                    legendre_function = pyshtools.legendre.PlON(
                        big_l, cos_theta
                    )
                    ratio = r / rs[sphere - 1]
                    u = np.sum(
                        ratio**eles
                        * (
                            ele_plus_1
                            * coefficients[
                                eles + 2 * (big_l + 1) * (sphere - 1 + n)
                            ]
                            + rs[sphere - 1]
                            * coefficients[
                                eles + (big_l + 1) * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        * legendre_function
                        / eles_times_2_plus_1
                    )
                    data_for_plotting[jj, ii] = u
                pass
            pass
        return x1, y1, data_for_plotting
    else:
        x = center[0]
        for ii in np.arange(0, len(x1)):
            y = x1[ii] + center[1]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - ps[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < rs[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num = num + 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r
                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta
                        )
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio**ele_plus_1
                            * (
                                eles
                                * coefficients[eles + 2 * (big_l + 1) * num]
                                + rs[num]
                                * coefficients[
                                    eles + (big_l + 1) * (2 * num + 1)
                                ]
                            )
                            * legendre_function
                            / eles_times_2_plus_1
                        )
                        u += u_temp
                        pass
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r
                    legendre_function = pyshtools.legendre.PlON(
                        big_l, cos_theta
                    )
                    ratio = r / rs[sphere - 1]
                    u = np.sum(
                        ratio**eles
                        * (
                            ele_plus_1
                            * coefficients[
                                eles + 2 * (big_l + 1) * (sphere - 1 + n)
                            ]
                            + rs[sphere - 1]
                            * coefficients[
                                eles + (big_l + 1) * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        * legendre_function
                        / eles_times_2_plus_1
                    )
                    data_for_plotting[jj, ii] = u
                pass
            pass
        return x1, y1, data_for_plotting


def draw_cut_representation_formula_n_sphere(
    cut: int,
    center: np.ndarray,
    horizontal: float,
    vertical: float,
    inter_horizontal: int,
    inter_vertical: int,
    coefficients: np.ndarray,
    radii: np.ndarray,
    positions: list[np.ndarray],
    big_l: int,
    exterior: Callable[[np.ndarray], float],
) -> np.ndarray:
    """

    Parameters
    ----------
    cut : int
        1, 2 or 3. Indicates if the drawing is a parallel cut of the
        plane: xy if = 1, xz if = 2, yz = 3 if other.
    center : np.ndarray
        array of floats of length 2. Coordinates of the center of the
        drawing.
    horizontal : float
        horizontal length of the rectangle that is going to be drawn.
    vertical : float
        vertical length of the rectangle that is going to be drawn.
    inter_horizontal : int
        > 0, quantity of points in the horizontal axis.
    inter_vertical : int
        > 0, quantity of points in the vertical axis.
    coefficients : np.ndarray
        coefficients of the spherical harmonics expansion.
    radii : np.ndarray
        numpy array of floats with the radii of the spheres.
    positions : list
        with numpy arrays that represents the position vector of the
        center of the spheres.
    big_l : int
        > 0, maximum order of the spherical harmonics used to discretize
        the traces.
    exterior : Callable[[np.ndarray], float]
        exterior function, to be evaluated in each point.

    Returns
    -------
    data_for_plotting : np.ndarray
        2D of floats for plotting.
    """
    n = len(radii)

    x1 = np.linspace(-horizontal / 2, horizontal / 2, inter_horizontal)
    y1 = np.linspace(-vertical / 2, vertical / 2, inter_vertical)
    data_for_plotting = np.zeros((len(y1), len(x1)))

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    l_square_plus_l = ele_plus_1 * eles
    l_times_l_plus_l_divided_by_2 = l_square_plus_l // 2
    big_el_plus_1_square = (big_l + 1) ** 2
    if cut == 3:
        z = 0.0 + center[2]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                y = y1[jj] + center[1]

                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - positions[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < radii[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num += 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - positions[num]
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
                                    l_square_plus_l[el]
                                    + 2 * big_el_plus_1_square * num
                                ]
                                + radii[num]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + big_el_plus_1_square * (2 * num + 1)
                                ]
                            )
                            temp *= legendre_function[
                                l_times_l_plus_l_divided_by_2[el]
                            ]
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
                                temp += (
                                    temp_minus_m + temp_plus_m
                                ) * legendre_function[
                                    l_times_l_plus_l_divided_by_2[el] + m
                                ]
                                pass
                            temp *= (
                                ratio ** ele_plus_1[el]
                                / eles_times_2_plus_1[el]
                            )
                            u_temp += temp
                            pass
                        u += u_temp
                        pass
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - positions[sphere - 1]
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
                    u_temp = 0.0
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
                                + big_el_plus_1_square
                                * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        temp *= legendre_function[
                            l_times_l_plus_l_divided_by_2[el]
                        ]
                        for m in np.arange(1, el + 1):
                            temp_plus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_minus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_plus_m *= cos_m_phi[m - 1]
                            temp_minus_m *= sin_m_phi[m - 1]
                            temp += (
                                temp_minus_m + temp_plus_m
                            ) * legendre_function[
                                l_times_l_plus_l_divided_by_2[el] + m
                            ]
                            pass
                        temp *= ratio**el / eles_times_2_plus_1[el]
                        u_temp += temp
                        pass
                    data_for_plotting[jj, ii] = u_temp
                pass
            pass
        return x1, y1, data_for_plotting
    elif cut == 2:
        y = center[1]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - positions[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < radii[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num += 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - positions[num]
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
                                    l_square_plus_l[el]
                                    + 2 * big_el_plus_1_square * num
                                ]
                                + radii[num]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + big_el_plus_1_square * (2 * num + 1)
                                ]
                            )
                            temp *= legendre_function[
                                l_times_l_plus_l_divided_by_2[el]
                            ]
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
                                temp += (
                                    temp_minus_m + temp_plus_m
                                ) * legendre_function[
                                    l_times_l_plus_l_divided_by_2[el] + m
                                ]
                                pass
                            temp *= (
                                ratio ** ele_plus_1[el]
                                / eles_times_2_plus_1[el]
                            )
                            u_temp += temp
                            pass
                        u += u_temp
                        pass
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - positions[sphere - 1]
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
                    u_temp = 0.0
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
                                + big_el_plus_1_square
                                * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        temp *= legendre_function[
                            l_times_l_plus_l_divided_by_2[el]
                        ]
                        for m in np.arange(1, el + 1):
                            temp_plus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_minus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_plus_m *= cos_m_phi[m - 1]
                            temp_minus_m *= sin_m_phi[m - 1]
                            temp += (
                                temp_minus_m + temp_plus_m
                            ) * legendre_function[
                                l_times_l_plus_l_divided_by_2[el] + m
                            ]
                            pass
                        temp *= ratio**el / eles_times_2_plus_1[el]
                        u_temp += temp
                        pass
                    data_for_plotting[jj, ii] = u_temp
                pass
            pass
        return x1, y1, data_for_plotting
    elif cut == 1:
        x = center[0]
        for ii in np.arange(0, len(x1)):
            y = x1[ii] + center[1]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                cart_vector = np.asarray([x, y, z])
                sphere = 0
                num = 0

                while num < n:
                    aux = cart_vector - positions[num]
                    r_aux = np.linalg.norm(aux)
                    if r_aux < radii[num]:
                        sphere = num + 1
                        num = n
                    pass
                    num += 1
                    pass

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.0
                    for num in np.arange(0, n):
                        aux = cart_vector - positions[num]
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
                                    l_square_plus_l[el]
                                    + 2 * big_el_plus_1_square * num
                                ]
                                + radii[num]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + big_el_plus_1_square * (2 * num + 1)
                                ]
                            )
                            temp *= legendre_function[
                                l_times_l_plus_l_divided_by_2[el]
                            ]
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
                                temp += (
                                    temp_minus_m + temp_plus_m
                                ) * legendre_function[
                                    l_times_l_plus_l_divided_by_2[el] + m
                                ]
                                pass
                            temp *= (
                                ratio ** ele_plus_1[el]
                                / eles_times_2_plus_1[el]
                            )
                            u_temp += temp
                            pass
                        u += u_temp
                    data_for_plotting[jj, ii] = u + ex
                else:
                    aux = cart_vector - positions[sphere - 1]
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
                    u_temp = 0.0
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
                                + big_el_plus_1_square
                                * (2 * (sphere - 1 + n) + 1)
                            ]
                        )
                        temp *= legendre_function[
                            l_times_l_plus_l_divided_by_2[el]
                        ]
                        for m in np.arange(1, el + 1):
                            temp_plus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    + m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_minus_m = (
                                ele_plus_1[el]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + 2
                                    * big_el_plus_1_square
                                    * (sphere - 1 + n)
                                ]
                                + radii[sphere - 1]
                                * coefficients[
                                    l_square_plus_l[el]
                                    - m
                                    + big_el_plus_1_square
                                    * (2 * (sphere - 1 + n) + 1)
                                ]
                            )
                            temp_plus_m *= cos_m_phi[m - 1]
                            temp_minus_m *= sin_m_phi[m - 1]
                            temp += (
                                temp_minus_m + temp_plus_m
                            ) * legendre_function[
                                l_times_l_plus_l_divided_by_2[el] + m
                            ]
                            pass
                        temp *= ratio**el / eles_times_2_plus_1[el]
                        u_temp += temp
                        pass
                    data_for_plotting[jj, ii] = u_temp
                pass
            pass
        return x1, y1, data_for_plotting
