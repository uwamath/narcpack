import numpy as np

class OrthoPoly:
    """
    Orthogonal Polynomial approximation
    on given data, interval

    Implements Gram-Schmidt orthogonalization
    to construct the polynomial basis

    Parameters
    func: function to approximate
    interval: tuple (start, end)

    """
    def __init__(self, func, interval):
        self.func = func
        self.interval = interval

    def approximate(self, n, m):
        """
        Gram-Schmidt to construct orthogonal polynomial basis
        Returns the m approximated points using n degree polynomial

        Parameters
        n: number of polynomials to use as basis
        m: number of points to evaluate the function at
        """
        self.x = np.linspace(self.interval[0], self.interval[-1], m)
        self.data = self.func(self.x)

        # gram-schmidt
        self.poly = [np.ones(m)]
        if n > 1:
            for i in range(1,n):
                tempx = self.x**i
                for j in range(len(self.poly)):
                    tempx -= (tempx @ self.poly[j] / (self.poly[j] @ self.poly[j])) * self.poly[j]
                self.poly.append(tempx.copy())

        # calculate coefficients
        ck = []
        ak = []
        for polys in self.poly:
            ck.append(np.trapz(polys**2, self.x))
            ak.append(np.trapz(polys * self.data, self.x) / ck[-1])

        # output
        self.out = np.zeros(m)
        for i, polys in enumerate(self.poly):
            self.out += ak[i] * polys
        return self.x, self.out, ak, self.poly
        

