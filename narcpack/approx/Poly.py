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

    def eval(self, x):
        """A function to evaluate the polynomial at given point or points x."""
        a = 0.0
        for deg, coeff in enumerate(self.coeffs):
            a += coeff*x**deg
        return(a)

    def deriv(self):
        """A function to return the derivative"""
        return(Poly(self.coeffs[1:]*np.arange(1,len(self.coeffs))))
