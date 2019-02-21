import numpy as np
from scipy.integrate import quad
from numpy.polynomial import chebyshev as C
        

class Rational:
    """
    Approximation of a function by a rational function.

    This class uses rational functions whose numerators and denominators are Chebyshev
    polynomials to approximate a given function on a given interval. The user may specify
    the degrees of the polynomials in both the numerator and denominator.
    """

    def __init__(self, f, interval=[-1,1], degs=[3,3], coeffs=None):
        """
        Find the rational approximation of a given function.

        Parameters
        ----------
        f : 1-D function
        Function to be approximated. Should be real-valued and accept real inputs.

        interval : array-like, optional (not quite working yet--best to leave as [-1,1])
            Left and right endpoints of interval over which to approximate f.
            f is approximated over interval[0] <= x <= interval[1].
            Modifies the interval variable of the Rational_approximation object to match interval

        degs : array-like, optional
            Degrees of the polynomials in the numerator and denominator.
            deg(numerator) = degs[0], deg(denominator) = degs[1]

        coeffs : array-like, optional
            Array-like object with two entries. The first should contain the Chebyshev cofficients
            of the numerator and the second the Chebyshev coefficients of the denominator of
            the rational approximation. coeffs may also be a dict with keys num and denom with
            corresponding values the Chebyshev coefficients of the numerator and denominator.
            Note: passing in coefficients will cause the function f to be ignored

        
        Example
        ----------
        import matplotlib.pyplot as plt
        
        f = lambda x: np.exp(-x)
        r = Rational(f, degs=[3,3])

        x = np.linspace(-1,3)
        plt.plot(x,f(x),'b-',x,r.eval(x),'r--')
        plt.legend(["True function","Rational approximation"])
        plt.show()
        """

        # Map the interval to [-1,1]
        assert len(interval) == 2, "interval must consist of two points"
        assert interval[0] < interval[1], "interval[0] must be less than interval[1]"
        self.interval = interval
        a, b = interval[0], interval[1]
        # f_norm = lambda x: f((2. * x - a - b) / (b - a))
        f_norm = lambda x: f((x * (b - a) + a + b) / 2.)

        # Get the coefficients, if given
        if coeffs is None:
            self.coeffs = {'num':C.Chebyshev(0), 'denom':C.Chebyshev(0)}
        else:
            if isinstance(coeffs, dict):
                self.coeffs=coeffs
            else:
                assert len(coeffs) == 2, "coeffs must have exactly two entries"
                self.coeffs = {'num':C.Chebyshev(coeffs[0],domain=interval), 'denom':C.Chebyshev(coeffs[1],domain=interval)}
            return

        # Find Chebyshev representation of f_norm * q
        assert len(degs) == 2, "degs must consist of two numbers"
        assert type(degs[0]) == int and type(degs[1]) == int, "entries of degs must be integers"
        assert degs[0] > -1 and degs[1] > -1, "entries of degs must be nonnegative"

        n, m = degs[0], degs[1]
        N = n + m                # Total degree of p and q
        f_cheb_coeffs = np.zeros(N+m+1)

        # (Could be made more efficient by using fixed points and precomputing f_norm(cos(theta)))
        f_cheb_coeffs[0] = (2 / np.pi) * quad(lambda theta: f_norm(np.cos(theta)) , 0, np.pi)[0]

        for k in range(1,N + m + 1):
            f_cheb_coeffs[k] = (2 / np.pi) * quad(lambda theta: f_norm(np.cos(theta)) * np.cos(k * theta), 0, np.pi)[0]
        
        # Form linear system to be solved for Chebyshev coefficients of p and q
        A = np.zeros((N+1,N+2))

        for i in range(N+1):
            if i <= n:
                A[i,i] = 1
            for j in range(n+1,N+1):
                if i > 0:
                    A[i,j] = -(f_cheb_coeffs[i+j-n]+f_cheb_coeffs[np.abs(i-j+n)]) / 2.
                else:        # Can vectorize this
                    A[i,j] = - f_cheb_coeffs[j-n]/ 2.
            
        A[:,N+1] = f_cheb_coeffs[:N+1]
        A[0,N+1] /= 2
        
        # Solve linear system and interpret coefficients as Chebyshev polynomials
        x = np.linalg.solve(A[:,:-1],A[:,-1])
        self.coeffs['num'] = C.Chebyshev(x[:n+1],domain=interval)
        self.coeffs['denom'] = C.Chebyshev(np.concatenate(([1],x[n+1:]),axis=None),domain=interval)


    
    def __call__(self,x):
        return self.coeffs['num'](np.array(x)) / self.coeffs['denom'](np.array(x))
