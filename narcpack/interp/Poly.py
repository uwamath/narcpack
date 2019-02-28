import numpy as np

class Poly:
    """Very simple polynomial class

    This is an extremely simple polynomial class for use in approximation classes.
    """

    def __init__(self, coeffs):
        """Init Poly with the given coefficients (beginning with degree 0)"""
        self.coeffs = np.array(coeffs)

    def __add__(self, other):
        """Override the + operator"""
        return Poly(self.coeffs+other.coeffs)

    def __sub__(self, other):
        """Override the - operator"""
        return Poly(self.coeffs-other.coeffs)

    def __call__(self, x):
        """
        A function to evaluate the polynomial at given point or points x.
        Uses Horner's method for efficient evaluation.
        """
        b = self.coeffs[-1]
        for a in reversed(self.coeffs[:-1]):
            b = a + b * x
        return(b)

    def deriv(self):
        """A function to return the derivative"""
        return(Poly(self.coeffs[1:]*np.arange(1,len(self.coeffs))))
