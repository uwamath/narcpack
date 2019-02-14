import numpy as np

class Poly:
    """Very simple polynomial class

    This is an extremely simple polynomial class for use in approximation classes.
    """

    def __init__(self, coeffs):
        """Init Poly with the given coefficients (beginning with degree 0).
        The optional last argument gives the base."""
        self.coeffs = np.array(coeffs)
        self.phi0 = lambda x : 0*x + 1.0
        self.phi1 = lambda x : x
        self.alpha = lambda x : x
        self.beta = lambda x : 0*x

    def __add__(self, other):
        """Override the + operator"""
        return(type(self)(self.coeffs+other.coeffs))

    def __sub__(self, other):
        """Override the - operator"""
        return(type(self)(self.coeffs-other.coeffs))

    def __call__(self, x):
        """Allow calling class to evaluate"""
        return(self.eval(x))

    def eval(self, x):
        """
        A function to evaluate the polynomial at given point or points x.
        Uses Horner's method for efficient evaluation.
        """
        b2 = x*0
        b1 = x*0
        for a in reversed(self.coeffs[:-1]):
            btmp = b1
            b1 = a + self.alpha(x)*b1 + self.beta(x)*b2
            b2 = btmp
        return(self.phi0(x)*self.coeffs[0]+self.phi1(x)*b1+self.beta(x)*self.phi0(x)*b2)

    def deriv(self):
        """A function to return the derivative"""
        return(Poly(self.coeffs[1:]*np.arange(1,len(self.coeffs))))
