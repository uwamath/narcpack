import numpy as np
from .Poly import Poly

class LLS(Poly):
    """Linear least-squares function approximation

    This class inherits our Poly class to approximate a given function as a linear function.
    """

    def __init__(self, func, interval=[-1,1]):
        """Inits LLS with a function to approximate and an interval on which to approximate"""
        self.func = func
        self.interval = np.array(interval)

        # Sample the given function at m points and evaluate
        m = 1000
        x = np.linspace(interval[0], interval[1], m)
        y = func(x)

        # Create the linear least-squares fit c0+c1*x**2
        c0 = (np.sum(x**2)*np.sum(y)-np.sum(x*y)*np.sum(x))/(m*np.sum(x**2)-np.sum(x)**2)
        c1 = (m*np.sum(x*y)-np.sum(x)*np.sum(y))/(m*np.sum(x**2)-np.sum(x)**2)

        self.coeffs = np.array([c0, c1])
