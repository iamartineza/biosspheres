import numpy as np


def azimuthal_trace_to_general_with_zeros(
    big_l: int, trace: np.ndarray
) -> np.ndarray:
    """
    Returns the extension by 0, for all m, of an array that has entries
    for all m = 0

    Parameters
    ----------
    big_l : int
        >= 0, max degree.
    trace : np.ndarray
        Vector that only has entries for m = 0.

    Returns
    -------
    general_trace : np.ndarray

    """
    num = big_l + 1
    eles = np.arange(0, num)
    eles_times_2_plus_one = eles * (eles + 1)
    general_trace = np.zeros(num**2, dtype=trace.dtype)
    general_trace[eles_times_2_plus_one] = trace[eles]
    return general_trace
