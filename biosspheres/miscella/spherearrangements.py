import numpy as np


def cube_vertex_positions(number: int, r: float, d: float):
    """
    To obtain an arrangement of number**3 spheres.
    
    Parameters
    ----------
    number: int
        the total number of spheres is number**3.
    r: float
        radius of each sphere.
    d: float
        distance between two surfaces of spheres.
    
    Returns
    -------
    ps
        position of the center of the spheres.
    
    """
    ps = []
    p = np.array([0., 0., 0.])
    two_r_plus_d = 2 * r + d
    for ii in np.arange(0, number):
        p[2] = 0.
        p = p + np.array([0., 0., two_r_plus_d * ii])
        for jj in np.arange(0, number):
            p[1] = 0.
            p = p + np.array([0., two_r_plus_d * jj, 0.])
            for zz in np.arange(0, number):
                p[0] = 0.
                ps.append(p + np.array([two_r_plus_d * zz, 0., 0.]))
                p = p + np.array([two_r_plus_d * zz, 0., 0.])
    return ps
