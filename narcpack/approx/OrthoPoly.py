import numpy as np

class OrthoPoly:
    """
    Orthogonal Polynomial approximation
    on given data, interval

    Implements Gram-Schmidt orthogonalization
    to construct the polynomial basis

    Parameters
    func: function to approximate
    interval: tuple (start, end)
    n: number of polynomials to use to approximate
    m: number of points to use

    """
    def __init__(self, func, interval, n, m):
        self.func = func
        self.interval = interval
        self.n = n
        self.m = m
        self.data = func(np.linspace(interval(0), interval(-1), m))

        # gram-schmidt
        p0 = 1 # base
        for i in range(n):
            break

# YOLO