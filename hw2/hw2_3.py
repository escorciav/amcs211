import numpy as np

def assert_extreme_point(A, b, C, d, x):
    """Check if x is an extrem point
    """
    if np.any(d != np.dot(C, x)):
        print "This point is infeasible. Don't satisfy EQ constraints"
        return False
    active_set = b - np.dot(A, x)
    if np.any(active_set < 0):
        print "This point is infeasible. Don't satisfy IEQ constraints"
        return False
    J = np.nonzero(active_set == 0)[0]
    print "Active constraints:\n", J
    r = np.linalg.matrix_rank(np.concatenate((A[J, :], C)))
    print "rank([A_J; C]) =", r
    if r == n:
        return True
    else:
        return False

# 6.1
A = np.array([[   0,   2,   2,   2,  -4],
              [   0,  -2,   2,  -2,   0],
              [   4,   0,   2,   0,   2],
             ], dtype=np.float)
b = np.array([[  2],
              [  2],
              [  2],
             ], dtype=np.float)
p, n = A.shape[0], A.shape[1]
C = np.concatenate((np.eye(n), -np.eye(n)))
m = C.shape[0]
d = np.ones((m, 1))
x_1 = np.array([[    1],
                [ -0.5],
                [    0],
                [ -0.5],
                [   -1],
               ])
x_2 = np.array([[  0],
                [  0],
                [  1],
                [  0],
                [  0],
               ], dtype=np.float)
x_3 = np.array([[  0],
                [  1],
                [  1],
                [ -1],
                [  0],
               ], dtype=np.float)

# (a)
print "Part (a):"
print "b - Ax_1 =\n", b - np.dot(A, x_1)
print "b - Ax_2 =\n", b - np.dot(A, x_2)
print "b - Ax_2 =\n", b - np.dot(A, x_3)
print "Part (b):"
print "x_1 is EP:\n", assert_extreme_point(C, d, A, b, x_1)
print "x_2 is EP:\n", assert_extreme_point(C, d, A, b, x_2)
print "x_3 is EP:\n", assert_extreme_point(C, d, A, b, x_3)

