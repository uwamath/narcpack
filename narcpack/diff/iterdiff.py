import numpy as np

def iterdiff(x, f, iterations=0):
    """Simple first-order finite difference iterated with trapezoidal-rule
    integration to hopefully smooth out derivative (with nothing done at the
    boundaries, so result will get shorter with each iteration)"""
    x, f = fdiff(x, f)
    for i in range(iterations):
        x, f = antider(x, f)
        x, f = fdiff(x, f)
    return(x, f)

def fdiff(x, f):
    return(x[:-1], np.diff(f)/np.diff(x))

def antider(x, f):
    ff = 0.0*f[1:]
    for i in range(1,len(f)):
        ff[i-1] = trule(x[:-i], f[:-i])
    return(x[1:], ff)

def trule(x, f):
    return(np.sum(0.5*np.diff(x)*(f[:-1]+f[1:])))
