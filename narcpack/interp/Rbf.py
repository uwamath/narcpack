import numpy as np

class Rbf:
    """
    Interpolation by radial basis functions.
    """

    def __init__(self, x, y, function='multiquadric'):
        """Initialize interpolation class

        Parameters:
        x : one-dimensional real array
        y : one-dimensional real array of same length as x
        function (optional) : string specifying the kernel to use. Options include
            'multiquadric'         : 
            'inverse multiquadric' : 
            'inverse quadratic'    :
            'gaussian'             : 
            'linear'               : 
            'cubic'                : 
            'quintic'              : 
            'thin_plate'           : 
        """

        # Estimate parameter for multiquadric, inverse, and gaussian functions
        if function in ['multiquadric', 'inverse multiquadric', 'inverse quadratic', 'gaussian']:
            # Set epsilon to approximate average distance between points
            sample_1 = np.random.choice(x, size=len(x), replace=True)
            sample_2 = np.random.choice(x, size=len(x), replace=True)

            self.epsilon = np.mean(np.abs(sample_1-sample_2))

        # Get the basis function to use
        functions = {
            'multiquadric': lambda r: np.sqrt(1 + (r * self.epsilon)**2) ,
            'inverse multiquadric': lambda r: 1 / np.sqrt(1 + (self.epsilon * r)**2) ,
            'inverse quadratic': lambda r: 1 / (1 + (self.epsilon * r)**2) ,
            'gaussian': lambda r: np.exp(-(self.epsilon * r)**2) ,
            'linear': lambda r: r ,
            'cubic': lambda r: r**3 ,
            'quintic': lambda r: r**5 ,
            'thin_plate': lambda r: r**2 * np.log(r)
        }

        if function in functions:
            self.function = functions[function]
        else:
            raise ValueError('Function passed to Rbf constructor is not supported.')


        if len(x) != len(y):
            raise ValueError('x and y must have the same size')
        n = len(x)
        X = np.empty((n,n))

        # Somewhat inefficient because X is symmetric
        for i in range(n):
            X[:,i] = self.function(np.abs(x[i] - x))

        self.weights = np.linalg.solve(X,y)
        self.x = x


    def __call__(self, x):
        """Evaluate interpolated function using the Bulirsch-Stoer algorithm

        Parameters:
        x : one-dimensional real array or real scalar

        This function should evaluate the interpolated function at a point or points x.
        The special __call__ function will let us evaluate using Rational(x)."""


        F = np.empty((len(x),len(self.x)))

        for i in range(len(x)):
            F[i,:] = self.function(np.abs(self.x - x[i]))

        return np.dot(F,self.weights)

