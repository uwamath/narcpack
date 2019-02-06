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
	    
	    c = (2/(n+1))*T.dot(f(np.cos(x))) # could implement as DCT instead of matrix multiply
	    c[0] /= 2

            self.coeffs = c

        elif isinstance(f, (list, tuple, np.ndarray)):
            self.coeffs = np.array(f)

        else:
            except ValueError:
                print("Cheb class must be initialized on scalar function or list like object")

    def get_nth_basis(n):
        """
        return n-th Chebyshev polynomial
        
        Parameters
        ----------
        n : integer

        Returns
        -------
        callable function (n-th Chebyshev polynomial)
        """
        return lambda x: np.cos(n*np.arccos(x))

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
