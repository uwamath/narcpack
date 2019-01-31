import numpy as np

def noapprox(f, interval=[-1,1]):
    """
    A perfect method for function approximation. Returns original function.
    Should only be used for testing tests.

    Parameters
    ----------
    f : 1-D function
        Function to be approximated. Should be real-valued and accept real
        inputs.

    interval : array-like, optional
        Left and right endpoints of interval over which to approximate f.
        f is approximated over interval[0] <= x <= interval[1].

    Returns
    ----------
    fapprox : function
        Approximation of f.
    """

    approxf = f

    return approxf
