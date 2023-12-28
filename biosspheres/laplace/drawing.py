import numpy as np
import pyshtools


def draw_cut_laplace_one_sphere_azimuthal_symmetry(
        cut, center, horizontal, vertical, inter_horizontal, inter_vertical,
        coefficients, radius, big_l, exterior):
    """

    Parameters
    ----------
    cut: int . 1, 2 or 3. Indicates if the drawing is a parallel cut of the
        plane: xy if = 1, xz if = 2, yz if other.
    center: array of floats of length 2. Coordinates of the center of the
        drawing.
    horizontal: float, horizontal length of the rectangle that is going to be
        drawn.
    vertical: float, vertical length of the rectangle that is going to be
        drawn.
    inter_horizontal: int > 0, quantity of points in the horizontal axis.
    inter_vertical: int > 0, quantity of points in the vertical axis.
    coefficients: coefficients of the spherical harmonics expansion.
    radius: radius of the sphere.
    big_l: int > 0, maximum order of the spherical harmonics used to discretize
        the traces.
    exterior: exterior function, to be evaluated in each point

    Returns
    -------
    data_for_plotting: array 2D of floats for plotting.
    """
    n = 1
    ps = [np.asarray([0., 0., 0.])]
    rs = np.asarray([radius])

    x1 = np.linspace(-horizontal / 2, horizontal / 2, inter_horizontal)
    y1 = np.linspace(-vertical / 2, vertical / 2, inter_vertical)
    data_for_plotting = np.zeros((len(y1), len(x1)))

    eles = np.arange(0, big_l + 1)
    ele_plus_1 = eles + 1
    eles_times_2_plus_1 = 2 * eles + 1
    if cut == 3:
        z = 0. + center[2]
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
                    num = num + 1

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r

                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta)
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio ** ele_plus_1 *
                            (eles * coefficients[eles + 2 * (big_l + 1) * num]
                             + rs[num] * coefficients[eles + (big_l + 1)
                                                      * (2 * num + 1)])
                            * legendre_function / eles_times_2_plus_1)
                        u = u + u_temp
                    data_for_plotting[jj, ii] = \
                        u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r

                    legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta)
                    ratio = r / rs[sphere - 1]
                    u = np.sum(
                        ratio ** eles * (ele_plus_1 *
                                         coefficients[eles + 2 * (big_l + 1) *
                                                      (sphere - 1 + n)] +
                                         rs[sphere - 1] *
                                         coefficients[eles + (big_l + 1) *
                                                      (2 * (sphere - 1 + n)
                                                       + 1)])
                        * legendre_function / eles_times_2_plus_1)
                    data_for_plotting[jj, ii] = u
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
                    num = num + 1

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r

                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta)
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio ** ele_plus_1 *
                            (eles * coefficients[eles + 2 * (big_l + 1) * num]
                             + rs[num] * coefficients[eles + (big_l + 1)
                                                      * (2 * num + 1)])
                            * legendre_function / eles_times_2_plus_1)
                        u = u + u_temp

                    data_for_plotting[jj, ii] = \
                        u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r

                    legendre_function = pyshtools.legendre.PlON(
                        big_l, cos_theta)
                    ratio = r / rs[sphere - 1]
                    u = np.sum(
                        ratio ** eles * (ele_plus_1 *
                                         coefficients[eles + 2 * (big_l + 1) *
                                                      (sphere - 1 + n)] +
                                         rs[sphere - 1] *
                                         coefficients[eles + (big_l + 1) *
                                                      (2 * (sphere - 1 + n)
                                                       + 1)])
                        * legendre_function / eles_times_2_plus_1)
                    data_for_plotting[jj, ii] = u
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
                    num = num + 1

                if sphere == 0:
                    ex = exterior(np.asarray([x, y, z]))
                    u = 0.
                    for num in np.arange(0, n):
                        aux = cart_vector - ps[num]
                        r = np.linalg.norm(aux)
                        z = aux[2]

                        cos_theta = z / r
                        legendre_function = pyshtools.legendre.PlON(
                            big_l, cos_theta)
                        ratio = rs[num] / r
                        u_temp = np.sum(
                            ratio ** ele_plus_1 *
                            (eles * coefficients[eles + 2 * (big_l + 1) * num]
                             + rs[num] * coefficients[eles + (big_l + 1)
                                                      * (2 * num + 1)])
                            * legendre_function / eles_times_2_plus_1)
                        u += u_temp
                    data_for_plotting[jj, ii] = \
                        u + ex
                else:
                    aux = cart_vector - ps[sphere - 1]
                    r = np.linalg.norm(aux)
                    z = aux[2]

                    cos_theta = z / r
                    legendre_function = pyshtools.legendre.PlON(
                        big_l, cos_theta)
                    ratio = r / rs[sphere-1]
                    u = np.sum(
                        ratio**eles * (ele_plus_1 *
                                       coefficients[eles + 2 * (big_l+1) *
                                                    (sphere - 1 + n)] +
                                       rs[sphere-1] *
                                       coefficients[eles + (big_l+1) *
                                                    (2 * (sphere - 1 + n)
                                                     + 1)])
                        * legendre_function / eles_times_2_plus_1)
                    data_for_plotting[jj, ii] = u
        return x1, y1, data_for_plotting
