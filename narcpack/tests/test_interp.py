from unittest import TestCase
import numpy as np

from narcpack.interp import Rbf,Polyinterp

class TestConstant(TestCase):
    def test_Rbf(self):
        g = lambda x : 0*x
        x = np.linspace(0,1,100)
        a = Rbf(x, g(x))
        self.assertTrue((a(x) == g(x)).all())

    def test_Polyinterp(self):
        g = lambda x : 0*x
        x = np.linspace(0,1,100)
        a = Polyinterp(x, g(x))
        self.assertTrue((a(x) == g(x)).all())
