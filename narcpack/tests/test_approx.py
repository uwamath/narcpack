from unittest import TestCase
import numpy as np

from narcpack.approx import noapprox
funcs = [noapprox]

class TestConstant(TestCase):
    def test_constant(self):
        g = lambda x : 0
        for f in funcs:
            a = f(g, np.array([0,1]))
            x = np.linspace(0,1,100)
            d = np.max(np.abs(a(x)-g(x)))
            self.assertTrue(d < 1)

    def test_abs(self):
        g = lambda x : np.abs(x)
        for f in funcs:
            a = f(g, np.array([-1,1]))
            x = np.linspace(-1,1,100)
            d = np.max(np.abs(a(x)-g(x)))
            self.assertTrue(d < 1)

    def test_step(self):
        g = lambda x : np.sign(x)
        for f in funcs:
            a = f(g, np.array([-1,1]))
            x = np.linspace(-1,1,100)
            d = np.max(np.abs(a(x)-g(x)))
            self.assertTrue(d < 1)

    def test_runge(self):
        g = lambda x : 1.0/(1.0+x**2)
        for f in funcs:
            a = f(g, np.array([-1,1]))
            x = np.linspace(-1,1,100)
            d = np.max(np.abs(a(x)-g(x)))
            self.assertTrue(d < 1)

    def test_exp(self):
        g = lambda x : np.exp(x)
        for f in funcs:
            a = f(g, np.array([0,8]))
            x = np.linspace(0,8,400)
            d = np.max(np.abs(a(x)-g(x)))
            self.assertTrue(d < 1)

    def test_sing(self):
        g = lambda x : np.exp(1.0/(x-1.0))
        for f in funcs:
            a = f(g, np.array([-1,1]))
            x = np.linspace(-1,1,100)
            d = np.max(np.abs(a(x)-np.append(g(x)[:-1],0.0)))
            self.assertTrue(d < 1)
