import matplotlib.pyplot as plt
from biosspheres.laplace.selfinteractions import v_jj_azimuthal_symmetry


def observing_v(
        big_l: int = 50,
        r: float = 2.5
) -> None:
    """
    Plots the V operator. It shows the plot.

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
    plt.plot(
        v,
        marker='x'
    )
    plt.title('V')
    plt.xlabel('l')
    pass


if __name__ == '__main__':
    observing_v()
    plt.show()
