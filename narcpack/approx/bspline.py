
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
