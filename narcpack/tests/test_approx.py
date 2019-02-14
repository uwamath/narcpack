from unittest import TestCase
import numpy as np
import warnings

from narcpack.approx import LLS
from narcpack.approx import Cheb

classes = [LLS,Cheb]
optargs = [{}, {'n':100}]

class TestApprox(TestCase):
    def test_constant(self):
        g = lambda x : 0*x
        for n, f in enumerate(classes):
            a = f(g, [0,1], **optargs[n])

    def test_abs(self):
        g = lambda x : np.abs(x)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])

    def test_step(self):
        g = lambda x : np.sign(x)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])

    def test_runge(self):
        g = lambda x : 1.0/(1.0+x**2)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])

    def test_exp(self):
        g = lambda x : np.exp(x)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])

    def test_sing(self):
        g = lambda x : np.exp(1.0/(1.0-x))
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
