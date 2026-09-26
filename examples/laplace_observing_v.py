import matplotlib.pyplot as plt
from biosspheres.laplace.selfinteractions import v_jj_azimuthal_symmetry


def observing_v(big_l: int, r: float) -> None:
    """
    Plots the V operator.

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    r : float
        > 0, radius.

    Returns
    -------
    None

    See Also
    --------
    biosspheres.laplace.selfinteractions.v_jj_azimuthal_symmetry

    """
    v = v_jj_azimuthal_symmetry(big_l, r)
    plt.figure()
    plt.plot(v, marker="x")
    plt.title("V")
    plt.xlabel("l")
    pass


if __name__ == "__main__":
    observing_v(50, 2.5)
    plt.show()
