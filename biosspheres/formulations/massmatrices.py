"""
This module has implementation for routines that return mass matrices
for one or several spheres when using spherical harmonics basis as
test and trial functions in a boundary integral formulation setting.
All routines return the diagonal of the matrix (the rest of the entries
are zero).

Routine listings
----------------
j_block
two_j_blocks
n_j_blocks
n_two_j_blocks
"""

import numpy as np
import biosspheres.utils.validation.inputs as val


def j_block(big_l: int, r: float, azimuthal: bool = True) -> np.ndarray:
    """
    Returns a numpy array with the boundary integral operator
    corresponding to the identity evaluated and tested with spherical
    harmonics of order 0 in a sphere of radius r.
    (Only returns the diagonal).

    Notes
    -----
    mass_matrix[l] = ( I Y_l,0 ; Y_l,0 )_L^2(surface sphere radius r).
    = r**2
    for each l such that 0 <= l <= big_l, and with
    ( ., .)_L^2(surface sphere radius r): inner product where indicated.

    Parameters
    ----------
    big_l : int
        >= 0, max degree. For big_l over 3000 a warning will be given.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    mass_matrix : np.ndarray
        with real values.
        If azimuthal = True
            Shape ((big_l+1), (big_l+1))
        Else
            Shape ((big_l+1)**2, (big_l+1)**2)

    See Also
    --------
    two_j_blocks
    n_j_blocks
    n_two_j_blocks

    """
    # Input validation
    val.big_l_validation(big_l, "big_l")
    val.radius_validation(r, "r")
    val.bool_validation(azimuthal, "azimuthal")

    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = r**2 * np.ones(num)

    assert np.isfinite(mass_matrix).all(), "Array contains NaN or Inf values."

    return mass_matrix


def two_j_blocks(big_l: int, r: float, azimuthal: bool = True) -> np.ndarray:
    """
    Returns a numpy array with the result of
    j_block(big_l, r, azimuthal) concatenated with itself.

    Parameters
    ----------
    big_l : int
        >= 0, max degree. For big_l over 3000 a warning will be given.
    r : float
        > 0, radius.
    azimuthal : bool
        Default True.

    Returns
    -------
    mass_matrix : np.ndarray
        with real values.
        If azimuthal = True
            Shape (2 * (big_l+1), 2 * (big_l+1))
        Else
            Shape (2 * (big_l+1)**2, 2 * (big_l+1)**2)

    See Also
    --------
    j_block
    n_j_blocks
    n_two_j_blocks

    """
    # Input validation
    val.big_l_validation(big_l, "big_l")
    val.radius_validation(r, "r")
    val.bool_validation(azimuthal, "azimuthal")

    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = r**2 * np.ones(2 * num)

    assert np.isfinite(mass_matrix).all(), "Array contains NaN or Inf values."

    return mass_matrix


def n_j_blocks(
    big_l: int, radii: np.ndarray, azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array with the result concatenated of the
    j_block(big_l, r, azimuthal) for each value r in radii.

    Parameters
    ----------
    big_l : int
        >= 0, max degree. For big_l over 3000 a warning will be given.
    radii : np.ndarray
        Array with the radii of the spheres.
    azimuthal : bool
        Default True.

    Returns
    -------
    mass_matrix : np.ndarray
        with real values.
        If azimuthal = True
            Shape (len(radii) * (big_l+1), len(radii) * (big_l+1))
        Else
            Shape (len(radii) * (big_l+1)**2, len(radii) * (big_l+1)**2)

    See Also
    --------
    j_block
    j_blocks
    n_two_j_blocks

    """
    # Input validation
    val.big_l_validation(big_l, "big_l")
    val.radii_validation(radii, "radii")
    val.bool_validation(azimuthal, "azimuthal")

    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = np.ones(len(radii) * num)
    for j in np.arange(0, len(radii)):
        mass_matrix[j * num : (j + 1) * num] = (
            radii[j] ** 2 * mass_matrix[j * num : (j + 1) * num]
        )

    assert np.isfinite(mass_matrix).all(), "Array contains NaN or Inf values."

    return mass_matrix


def n_two_j_blocks(
    big_l: int, radii: np.ndarray, azimuthal: bool = True
) -> np.ndarray:
    """
    Returns a numpy array with the result concatenated of the
    two_j_blocks(big_l, r, azimuthal) for each value r in radii.

    Parameters
    ----------
    big_l : int
        >= 0, max degree. For big_l over 3000 a warning will be given.
    radii : np.ndarray
        Array with the radii of the spheres. Should be one dimensional.
    azimuthal : bool
        Default True.

    Returns
    -------
    mass_matrix : np.ndarray
        with real values.
        If azimuthal = True
            Shape (2 * len(radii) * (big_l+1),
                    2 * len(radii) * (big_l+1))
        Else
            Shape (2 * len(radii) * (big_l+1)**2,
                    2 * len(radii) * (big_l+1)**2)

    See Also
    --------
    j_block
    j_blocks
    n_j_blocks

    """
    # Input validation
    val.big_l_validation(big_l, "big_l")
    val.radii_validation(radii, "radii")
    val.bool_validation(azimuthal, "azimuthal")

    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = np.ones(2 * len(radii) * num)
    for j in np.arange(0, len(radii)):
        mass_matrix[2 * j * num : 2 * (j + 1) * num] = (
            radii[j] ** 2 * mass_matrix[2 * j * num : 2 * (j + 1) * num]
        )

    assert np.isfinite(mass_matrix).all(), "Array contains NaN or Inf values."

    return mass_matrix
