import logging

import numpy as np

import lin_prog as lp

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG)

# 6.1
A = np.array([[  -1,  -6,   1,   3],
              [  -1,  -2,   7,   1],
              [   0,   3, -10,  -1],
              [  -6, -11,  -2,  12],
              [   1,   6,  -1,  -3],
             ])
b = np.array([[ -3],
              [  5],
              [ -8],
              [ -7],
              [  4],
             ])
c = np.array([[  47],
              [  93],
              [  17],
              [ -93],
             ])
x = np.array([[ 1],
              [ 1],
              [ 1],
              [ 1],
             ])

J = lp.active_constraints(A, b, x)
z = lp.get_dual_vars(A, c, J)
print "b - Ax =\n", b - np.dot(A, x)
print "Dual variables:\n", z

# 6.2
A = np.array([[ -1,  0,  0],
              [  0, -1,  0],
              [  0,  0, -1],
              [  1,  0,  0],
              [  0,  1,  0],
              [  0,  0,  1],
              [  1,  1,  1]
             ])
b = np.array([[ 0],
              [ 0],
              [ 0],
              [ 2],
              [ 2],
              [ 2],
              [ 4],
             ])
c = np.array([[  1],
              [  1],
              [ -1],
             ])
x_1 = np.array([[ 2],
                [ 2],
                [ 0],
               ])

x_s = lp.simplex(A, b, c, x_i=x_1, p_rule='bland', verbose=True)
