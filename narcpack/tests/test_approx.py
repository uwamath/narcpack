from unittest import TestCase
import numpy as np

from narcpack.approx import LLS
classes = [LLS]

class TestConstant(TestCase):
    def test_constant(self):
        g = lambda x : 0
        for f in classes:
            a = f(g, [0,1])

    def test_abs(self):
        g = lambda x : np.abs(x)
        for f in classes:
            a = f(g, [-1,1])

    def test_step(self):
        g = lambda x : np.sign(x)
        for f in classes:
            a = f(g, [-1,1])

    def test_runge(self):
        g = lambda x : 1.0/(1.0+x**2)
        for f in classes:
            a = f(g, [-1,1])

    def test_exp(self):
        g = lambda x : np.exp(x)
        for f in classes:
            a = f(g, [0,8])

    def test_sing(self):
        g = lambda x : np.exp(1.0/(1.0-x))
        for f in classes:
            a = f(g, [-1,1])
