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
        for i in range(0, deg-1):
            self.knots[i] = a
        for i in range(self.nKnots - self.deg, self.nKnots - 1):
            self.knots[i] = b
        for i in range(1, self.nKnots - 2 * self.deg):
            self.knots[self.deg + i] = i * ((b - a) / (self.nKnots - 2 * self.deg))

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
                ans.append(self.onePoint(self.knots[self.deg], self.deg, self.deg, self.knots, self.fSample))
            else:
                i = self.deg
                while self.knots[i] <= xpoint:
                    i += 1
                ans.append(self.onePoint(xpoint, self.deg+i, self.deg, self.knots, self.fSample))
        return(ans)

    def onePoint(self, x, i, d, k, y):
        """
        Pointwise evaluation of b-spline using deBoor's algorithm.
        Cribbed from Wikipedia.

        'x' is the position at which to evaluate the spline.
        'i' is the nearest knot index <= x.
        'd' is the polynomial degree.
        'k' is are the d-padded knot positions.
        'y' are the control points.
        """

        s = [y[j + i - d] for j in range(0, d+1)]
        for r in range(1, d+1):
            for j in range(d, r-1, -1):
                alpha = (x - k[j+i-d]) / (k[j+1+i-r] - k[j+i-d])
                s[j] = (1.0 - alpha) * s[j-1] + alpha * s[j]
        return s[d]
