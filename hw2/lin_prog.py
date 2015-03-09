# Utilities to analyze/solve Linear Programs in the standard inequality form:
# min c^T x
# s.t. A x <= b
"""
TODO
1. LP seed
Function to get a corner of the polyhedron.
Be carefull of transform A&b to use method of Vandenberghe's slides

2. Use LU factorization
"""
import logging

import numpy as np

logger = logging.getLogger('lin_prog module:')
np.set_printoptions(precision=4)

def active_constraints(A, b, x):
    """Return a set of active constraints
    """
    dx = b - np.dot(A,x)
    return np.nonzero(dx == 0)[0]

def check_unboundness(A_times_delta_x):
    """Check if LP is unbounded
    """
    if np.all(A_times_delta_x <= 0):
        logger.info('LP is unbounded')
        return True
    else:
        return False

def compute_delta_x(A, J, k):
    """Compute direction to get next adjacent extreme point
    """
    rst = np.zeros((A.shape[0], 1))
    rst[k] = -1
    delta_x = np.linalg.solve(A[J, :], rst[J, :])
    A_times_delta_x = np.dot(A, delta_x)
    return delta_x, A_times_delta_x

def correct_degeneracy(J, A, n):
    """Return a non-degenerate set if it isn't
    """
    card_J = J.size
    if card_J == n:
        return J, False
    elif card_J < n:
        return None, True
    else:
        while True:
            Jnd = np.random.choice(J, n, replace=False)
            r = np.linalg.matrix_rank(A[Jnd, :])
            if r == n:
                break
        return Jnd, False

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

def pivoting_add(ratio, rule):
    """Choose constraint to get an adjacent extreme with lower cost
    """
    j = np.argmin(ratio)
    check_tie = ratio == ratio.min()
    if rule.lower() == 'bland' and check_tie.sum() > 1:
       j = check_tie.nonzero()[0].min() 
    return j

def pivoting_remove(z, rule):
    """Choose which active constraint will be replaced
    """
    if rule is None:
        k = np.argmin(z)
    elif rule.lower() == 'bland': 
        k = np.min(np.nonzero(z < 0)[0])
    else:
        raise('Undefined pivoting rule')
    return k

def simplex(A, b, c, x_i=None, p_rule=None):
    """Solve a linear program by simplex
    """
    if x_i is None:
        x = lp_seed(A, b)
    else:
        x = x_i

    it, m, n = 1, A.shape[0], A.shape[1]
    fx = np.dot(c.T, x_i)
    J = active_constraints(A, b, x)
    J, stop = correct_degeneracy(J, A, n)
    if stop:
        logger.info('Insufficient active constraints')

    while True:
        logger.info('Iteration: {0}, f(x): {1}'.format(it, fx))
        z = get_dual_vars(A, c, J)
        logger.info('x =\n' + np.array_str(x))
        logger.info('J =\n' + np.array_str(J))
        logger.info('z =\n' + np.array_str(z))

        if (z[J]>0).sum() == n:
            logger.info('An optimum value was found')
            break
        else:
            b_Ax = b - np.dot(A, x)
            k = pivoting_remove(z, rule=p_rule)
            d_x, Ad_x = compute_delta_x(A, J, k)
            stop = check_unboundness(Ad_x)
            if stop:
                x[:] = -np.inf
                break

            idx, ratio = Ad_x <= 0, b_Ax / Ad_x
            ratio[idx] = np.inf
            j = pivoting_add(ratio, rule=p_rule)
            J[J==k] = j

            x = x + ratio[j]*d_x
            fx = np.dot(c.T, x)
            it += 1

            logger.info('delta_x=\n' + np.array_str(d_x))
            logger.info('b - Ax=\n' + np.array_str(b_Ax))
            logger.info('Ad_x=\n' + np.array_str(Ad_x))
            logger.info('alpha: {0}'.format(ratio[j]))
            logger.info('k: {0}, j: {1}'.format(k, j))

    return x

