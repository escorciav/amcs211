from numpy import any, array, array_str, concatenate, dot, intersect1d
from numpy import nonzero, ravel, setdiff1d, zeros
from numpy.linalg import lstsq, norm

def active_constraints(A, b, x):
    """Return a set of active constraints
    """
    dx = b - dot(A, x)
    if any(dx > 0):
        raise ValueError('x is infeasible')
    return nonzero(dx == 0)[0]

def get_starting_point(A, b):
    raise NotImplementedError('You need to provide x_0')

def get_dual_vars(G, f, A, x, W):
    """Compute lagrange multiplies for W"""
    mu, res, rank, s = lstsq(A[W, :].T, dot(G, x) + f)
    return mu

def quadprog(G, f, A, b, x_0=None, max_iter=1e6, tol=1e-4, verbose=True):
    """Solve convex QP via active set method

    min 0.5 * x^T * G * x + f^T * x
    s.t. A * x >= b

    """
    # Compute a feasible starting point
    if x_0 is None:
        x = get_starting_point(A, b)
    else:
        x = x_0.copy()
    m, n, k = x.size, A.shape[0], 0
    # Set initial active-set of contraints
    W = active_constraints(A, b, x)
    while k < max_iter:
       f_obj = dot(x.T, dot(G, x)) + dot(f.T, x)
       if verbose:
           print 'Iter: {0}\tObj: {1}\tW: {2:5}\tx: {3}'.format(k,
               array_str(ravel(f_obj), precision=3), array_str(W),
               array_str(ravel(x), precision=3))
       p = solve_qp_eq(G, f, A[W, :], b[W], x)
       if norm(p) <= tol * m**2:
           lmbda = get_dual_vars(G, f, A, x, W)
           if all(lmbda >= 0):
               # Global minimizer was found
               if verbose:
                   print 'Optimization finished\n'
               break
           else:
               # Remove most-violated constraint
               j = lmbda.argmin()
               W = setdiff1d(W, [W[j]])
       else:
           alpha, j = step_size(A, b, x, p, W)
           x = x + alpha*p
           if j is not None:
               # Add a blocking constraint
               W = concatenate((W, [j]), axis=0)
       k += 1
    if k == max_iter and verbose:
        print 'Max number of iterations reached'
    return x

def solve_qp_eq(G, f, C, d, x):
    """Solve QP with equality constraint"""
    m, n = x.size, C.shape[0]
    A = concatenate((concatenate((G, C.T), axis=1),
                     concatenate((C, zeros((n, n))), axis=1)), axis=0)
    b = concatenate((f + dot(G, x), dot(C, x) - d), axis=0)
    lmbda, res, rank, s = lstsq(A, b)
    return -lmbda[:m, :]

def step_size(A, b, x, p, W):
    """Compute minimum step size such that x + p is still in the polyhedron"""
    num, den, alpha_min, block = b - dot(A, x), dot(A, p), 1, None
    alpha = num / den
    J = intersect1d(setdiff1d(range(A.shape[0]), W, assume_unique=True),
                    nonzero(den < 0)[0])
    if J.size != 0:
        alpha_min = alpha[J].min()
    alpha_hat = min(1, alpha_min)
    # Check blocking constraints
    if alpha_hat < 0:
        raise ValueError('Got a negative alpha (step length)')
    elif alpha_hat < 1:
        block = nonzero(alpha_hat == alpha)[0][0]
    return alpha_hat, block
 
def hw5_1b():
    """
    Solve the following qp with active set method
    min x_1^2 + 2*x_2^2 - 2*x_1 - 6*x_2 - 2*x_1*x_2
    s.t. 0.5*x_1 + 0.5*x_2 <= 1; -1*x_1 + 2*x_2 <= 2; x_1>=0; x_2>= 0

    Choose x_0 inside, in the boundary and as extrem point.
    """
    G = array([[2, -2],[-2, 4]])
    F = array([[-2],[-6]])
    A = -1 * array([[0.5, 0.5], [-1, 2], [-1, 0], [0, -1]])
    b = -1 * array([[1],[2],[0],[0]])
    x_seed = [array([[1], [0.5]]), array([[0], [0]]), array([[0], [0.5]])]
    for x_0 in x_seed:
        x = quadprog(G, F, A, b, x_0)

if __name__ == '__main__':
    hw5_1b()
