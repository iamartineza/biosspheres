import numpy as np
import matplotlib.pyplot as plt
from biosspheres.helmholtz.selfinteractions import v_jj_azimuthal_symmetry


def observing_v() -> None:
    """
    Plots the real and imaginary parts of the V operator. It shows the plot.
    
    Returns
    -------
    None
    
    """
    r = 2.5
    big_l = 50
    k = 7.
    
    v = v_jj_azimuthal_symmetry(big_l, r, k)
    plt.figure()
    plt.plot(
        np.real(v),
        marker='x',
        label='Real part of V'
    )
    plt.plot(
        np.imag(v),
        marker='x',
        label='Imaginary part of V'
    )
    plt.title('V')
    plt.xlabel('l')
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    observing_v()
