import numpy as np

class Bspline:
    """
    Elements of a b-spline implementation
    """
    def __init__(self, f, interval=[-1, 1], nControl=10, deg=3):
        """
        Invokes low-level method evaluating b-spline at specified parameters.

        Parameters
        ---------
        f : 1-d function
        Approximant mapping reals to reals.

        interval :  real pair.
        Domain over which to approximate.

        nControl : positive integer.
        Number of ("control") points at which to sample f within interval.

        deg : integer
        Degree of spline approximator.
        """
        self.nControl = nControl
        self.deg = deg
        self.nKnots = self.nControl + self.deg + 1
        self.knots = np.zeros(self.nKnots)

        """
        End knots repeated deg-many times for differentiability.
        Central knots evenly spaced, for now.
        """
        a, b = interval[0], interval[1]
        for i in range(0, deg):
            self.knots[i] = a
        for i in range(self.nKnots-self.deg-1, self.nKnots):
            self.knots[i] = b
        self.knots[self.deg:self.nKnots-self.deg-1] = np.linspace(a,b,self.nKnots-2*self.deg-1)

        """
        Samples evenly-space control values, for now.
        """
        self.fSample = np.zeros(nControl)
        for i in range(0, nControl - 1):
             self.fSample[i] = f(i * ((b - a) / nControl))
             
    def __call__(self, x):
        ans = []
        for xpoint in x:
            if xpoint < self.knots[self.deg]:
                ans.append(self.onePoint(self.knots[self.deg], self.deg))
            else:
                i = self.deg
                while (self.knots[i+1] <= xpoint) & (i < self.nKnots-self.deg-3):
                    i += 1
                ans.append(self.onePoint(xpoint, i))
        return(ans)

    def onePoint(self, x, i):
        """
        Pointwise evaluation of b-spline using deBoor's algorithm.
        Cribbed from Wikipedia.

        'x' is the position at which to evaluate the spline.
        'i' is the nearest knot index <= x.
        """

        s = [self.fSample[j + i - self.deg] for j in range(0, self.deg+1)]
        for r in range(1, self.deg+1):
            for j in range(self.deg, r-1, -1):
                alpha = (x - self.knots[j+i-self.deg]) / (self.knots[j+1+i-r] - self.knots[j+i-self.deg])
                s[j] = (1.0 - alpha) * s[j-1] + alpha * s[j]
        return s[self.deg]
