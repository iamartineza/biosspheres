import numpy as np


def j_block(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> np.ndarray:
    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = r**2 * np.ones(num)
    return mass_matrix


def two_j_blocks(
        big_l: int,
        r: float,
        azimuthal: bool = True
) -> np.ndarray:
    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = r**2 * np.ones(2 * num)
    return mass_matrix


def n_j_blocks(
        big_l: int,
        radii: np.ndarray,
        azimuthal: bool = True
) -> np.ndarray:
    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = np.ones(len(radii) * num)
    for j in np.arange(0, len(radii)):
        mass_matrix[j*num:(j+1)*num] = (
                radii[j]**2 * mass_matrix[j*num:(j+1)*num])
    return mass_matrix


def n_two_j_blocks(
        big_l: int,
        radii: np.ndarray,
        azimuthal: bool = True
) -> np.ndarray:
    num = big_l + 1
    if not azimuthal:
        num = num**2
    mass_matrix = np.ones(2 * len(radii) * num)
    for j in np.arange(0, len(radii)):
        mass_matrix[2*j*num:2*(j+1)*num] = (
                radii[j]**2 * mass_matrix[2*j*num:2*(j+1)*num])
    return mass_matrix
