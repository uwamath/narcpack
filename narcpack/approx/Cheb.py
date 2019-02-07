import numpy as np

class Cheb:
    """Very simple polynomial class

    This is an extremely simple Chebyshev polynomial class for use in approximation classes.
    """

    def __init__(self, f, interval=[-1,1], n=10):
        """
        Parameters
        ----------
        f : scalar valued function or coefficients for basis
        interval : array_like, optional
               Left and right endpoints of interval over which to approximate f.
               f is approximated over interval[0] <= x <= interval[1]
        n : integer
            degree of approximation
        """
        
        # check if f is a callable function or list
        if callable(f):
            a,b = interval
            
            x = (np.arange(n+1)+0.5)*np.pi/(n+1)
            
            T = np.cos(np.outer(np.arange(n+1),x))
            
            c = (2/(n+1))*T@f(np.cos(x)) # could implement as DCT instead of matrix multiply
            c[0] /= 2
            
            self.coeffs = c
            self.n = n
            
        elif isinstance(f, (list, tuple, np.ndarray)):
            self.coeffs = np.array(f)
            self.n = len(self.coeffs)

        else:
            raise ValueError("Cheb class must be initialized on scalar function or list like object")
        
        # need j=j because of python 'late binding closures'
        self.basis = [(lambda x, j=j: np.cos(j*np.arccos(x))) for j in range(n)]
        
    def __add__(self, other):
        """Override the + operator"""
        return Cheb(self.coeffs+other.coeffs)

    def __sub__(self, other):
        """Override the - operator"""
        return Cheb(self.coeffs-other.coeffs)

    def eval(self, x):
        """
        A function to evaluate the polynomial at given point or points x.
        
        Notes
        -----
        There are better algorithms for evaluating Chebyshev polynomials
        """
        
        a = 0.0
        for j in range(self.n):
            a += self.coeffs[j]*self.basis[j](x)
        return(a)

    def deriv(self):
        """A function to return the derivative"""
        return None
