# Utilities to analyze/solve Linear Programs in the standard inequality form:
# min c^T x
# s.t. A x <= b
import numpy as np

"""
TODO
1. LP seed
Function to get a corner of the polyhedron.
Be carefull of transform A&b to use method of Vandenberghe's slides
"""

def active_constraints(A, b, x):
    """Return a set of active constraints
    """
    gx = np.dot(A,x)
    return np.nonzero(gx == b)[0]

def get_dual_vars(A, c, J):
    """Compute dual variables given a set of active constraints
    """
    z_J = np.linalg.solve(A[J, :].T, -c)
    z = np.zeros((A.shape[0], 1))
    z[J] = z_J
    return z

def lp_seed(A, b):
    """Find a corner for simplex method
    """
    raise('Sorry, not deployed yet!!!')

def simplex(A, b, c, x_0=None, deg_alg='bland', verbose=False):
    """Solve a linear program by simplex
    """
    it = 0

    if x_0 is None:
        x = lp_seed(A, b)
    else:
        x = x_0

    while True:
        print 'Iter:', it
        J = active_constraints(A, b, x)
        z = get_dual_vars(A, c, J)
        if (z[J]>0).sum():
            break
        else:
    return x
