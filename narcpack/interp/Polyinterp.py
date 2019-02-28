import numpy as np
from .Poly import Poly

class Polyinterp(Poly):
    """Simple polynomial interpolation"""

    def __init__(self, x, y):
        """Initialize interpolation class

        Parameters:
        x : one-dimensional real array
        y : one-dimensional real array of same length as x"""

        V = np.zeros([len(x),len(y)])
        for i in range(len(y)):
            V[:,i] = x**i

        self.coeffs = np.linalg.solve(V,y)
