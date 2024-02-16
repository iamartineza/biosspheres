import numpy as np


def cube_vertex_positions(number: int, r: float, d: float) -> list[np.ndarray]:
    """
    To obtain an arrangement of number**3 spheres.

    Notes
    -----
    This functions assumes the radii of all spheres are the same, and
    orders them in the corners of an array of cubes, starting in the
    origin of the coordinate system and placing them in the first
    quadrant.
    We need three parameters, number, r and d. These are translated to
    number**3 spheres, with r the radius of each one, and the length of
    the edge of each cube is 2r + d. If d=0 the spheres will touch each
    other.

    Parameters
    ----------
    number: int
        the total number of spheres is number**3.
    r: float
        radius of each sphere.
    d: float
        distance between two surfaces of spheres. The length of the edge
        of each cube 2r + d. If d=0 the spheres will touch each other.

    Returns
    -------
    ps: list[np.ndarray]
        position of the center of the spheres.

    """
    ps = []
    p = np.array([0.0, 0.0, 0.0])
    two_r_plus_d = 2 * r + d
    for ii in np.arange(0, number):
        p[2] = 0.0
        p = p + np.array([0.0, 0.0, two_r_plus_d * ii])
        for jj in np.arange(0, number):
            p[1] = 0.0
            p = p + np.array([0.0, two_r_plus_d * jj, 0.0])
            for zz in np.arange(0, number):
                p[0] = 0.0
                ps.append(p + np.array([two_r_plus_d * zz, 0.0, 0.0]))
                p = p + np.array([two_r_plus_d * zz, 0.0, 0.0])
                pass
            pass
        pass
    return ps
