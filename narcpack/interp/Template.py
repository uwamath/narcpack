class Template:
    """Template for interpolation class """

    def __init__(self, x, y, optArg1=None, optArg2=None, optArg3=None):
        """Initialize interpolation class

        Parameters:
        x : one-dimensional real array
        y : one-dimensional real array of same length as x
        optArg1, ... : optional arguments

        This function should perform the interpolation once Template is initialized."""

    def __call__(self, x):
        """Evaluate interpolated function

        Parameters:
        x : one-dimensional real array or real scalar

        This function should evaluate the interpolated function at a point or points x.
        The special __call__ function will let us evaluate using Template(x)."""
