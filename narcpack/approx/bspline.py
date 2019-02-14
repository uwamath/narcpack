import numpy as np

class BSpline:
    """
    Elements of a b-spline implementation
    """
    def fit(self, f, interval=[-1, 1], cPts = 10, deg=3):
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
        nKnots = nControl + deg + 1
        knots = np.zeros(nKnots)

        """
        End knots repeated deg-many times for differentiability.
        Central knots evenly spaced, for now.
        """
        a, b = interval[0], interval[1]
        for i in range(0, deg-1):
            knots[i] = a
        for i in range(nKnots - deg, nKnots - 1):
            knots[i] = b

        for i in range(1, nKnots - 2 * deg):
            knots[deg + i] = i * ((b - a) / (nKnots - 2 * deg))

        """
        Samples evenly-space control values, for now.
        """
        fSample = np.zeros(nControl)
        for i in range(0, nControl - 1):
             fSample[i] = f(i * ((b - a) / nControl))
             
        yApprox = np.zeros(nControl)
        for i in range(0, nControl-1):
            yApprox[i] = onePoint(fSample, deg + i, deg, knots, cPts)
        

    def onePoint(x, i, d, k, y):
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
