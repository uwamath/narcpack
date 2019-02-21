from unittest import TestCase
import numpy as np

from narcpack.approx import LLS, Cheb, Rational, RemezPoly, OrthoPoly

classes = [LLS,Cheb,Rational,RemezPoly,OrthoPoly]
optargs = [{},{},{'degs':[1,0]},{},{}]

class TestConstant(TestCase):
    def test_constant(self):
        g = lambda x : 0*x
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
            x = np.linspace(0,1,1000)
            self.assertTrue((a(x) == g(x)).all())

