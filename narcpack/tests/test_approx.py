from unittest import TestCase
import numpy as np
import warnings

from narcpack.approx import LLS
from narcpack.approx import Cheb

classes = [LLS,Cheb]
optargs = [{}, {'n':100}]

warnings.filterwarnings('ignore')

class TestConstant(TestCase):
    def test_constant(self):
        g = lambda x : 0*x
        for n, f in enumerate(classes):
            a = f(g, [0,1], **optargs[n])
            x = np.linspace(0,1,1000)
            print('test_constant '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))

    def test_abs(self):
        g = lambda x : np.abs(x)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
            x = np.linspace(-1,1,1000)
            print('test_abs '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))

    def test_step(self):
        g = lambda x : np.sign(x)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
            x = np.linspace(-1,1,1000)
            print('test_step '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))

    def test_runge(self):
        g = lambda x : 1.0/(1.0+x**2)
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
            x = np.linspace(-1,1,1000)
            print('test_runge '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))

    def test_exp(self):
        g = lambda x : np.exp(x)
        for n, f in enumerate(classes):
            a = f(g, [0,8], **optargs[n])
            x = np.linspace(0,8,1000)
            print('test_exp '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))

    def test_sing(self):
        g = lambda x : np.exp(1.0/(1.0-x))
        for n, f in enumerate(classes):
            a = f(g, [-1,1], **optargs[n])
            x = np.linspace(-1,1,1000)
            print('test_sing '+f.__name__+' '+str(np.max(np.abs(a.eval(x)-g(x)))))
