import matplotlib.pyplot as plt
from numpy import array, dot, eye, ones, zeros
from numpy.linalg import inv, norm

def hw4_1():
    """
    Use
    1. steepest descent
    2. Newton algorithm
    3. Cauchy point
    4. Conjugate Gradient
    5. Quasi-newton method

    to minimize the Rosenbrock function (2.22). First try the initial point
    x0 = (1.2, 1.2)T and then the more difficult starting point x0 = (1.2, 1)T
    """
    alpha_max , rho, c = 1, 0.5, 0.5
    delta_max, eta = 2, 1.0/8
    x_p, f_obj, alpha, iter = [[]] * 5, [[]] * 5, [[]] * 5, [[]] * 5
    leg = ['Trust Region (Cauchy Point)', 'Steepest Descent',
           'Newton', 'Conjugate Gradient', 'Quasi-Newton']
    f = (rb_function, rb_gradient, rb_hessian)

    x_0, i = array([[1.2], [1.2]]), 0
    x_p[i], f_obj[i], alpha[i], iter[i] = trust_region_min(f, x_0, delta_max,
        eta)
    i += 1
    x_p[i], f_obj[i], alpha[i], iter[i] = backtracking_min(f, x_0,
        'steepest descent', alpha_max, rho, c)
    i += 1
    x_p[i], f_obj[i], alpha[i], iter[i] = backtracking_min(f, x_0,
        'newton', alpha_max, rho, c)
    i += 1
    f = (rb_function, rb_gradient, [[], False])
    x_p[i], f_obj[i], alpha[i], iter[i] =  backtracking_min(f, x_0, 'cg-fr',
        alpha_max, rho, c)
    i += 1
    f = (rb_function, rb_gradient, [rb_hessian(x_0), x_0.copy(), [], False])
    x_p[i], f_obj[i], alpha[i], iter[i] =  backtracking_min(f, x_0,
        'quasi-newton', alpha_max, rho, c)
    plot_results('out1.pdf', x_p, f_obj, alpha, iter, '[1.2, 1.2].T', leg)


    x_0, i = array([[1.2], [1.0]]), 0
    f = (rb_function, rb_gradient, rb_hessian)
    x_p[i], f_obj[i], alpha[i], iter[i] = trust_region_min(f, x_0, delta_max,
        eta)
    i += 1
    x_p[i], f_obj[i], alpha[i], iter[i] = backtracking_min(f, x_0,
        'steepest descent', alpha_max, rho, c)
    i += 1
    x_p[i], f_obj[i], alpha[i], iter[i] = backtracking_min(f, x_0,
        'newton', alpha_max, rho, c)
    i += 1
    f = (rb_function, rb_gradient, [[], False])
    x_p[i], f_obj[i], alpha[i], iter[i] =  backtracking_min(f, x_0, 'cg-fr',
        alpha_max, rho, c)
    i += 1
    f = (rb_function, rb_gradient, [rb_hessian(x_0), x_0.copy(), [], False])
    x_p[i], f_obj[i], alpha[i], iter[i] =  backtracking_min(f, x_0,
        'quasi-newton', alpha_max, rho, c)
    plot_results('out2.pdf', x_p, f_obj, alpha, iter, '[1.2, 1.0].T', leg)
    return None

def backtracking_min(f, x_0, method, alpha_max=1, rho=0.5, c=0.5, iter=0,
                     tol=1e-4, max_iter=1e6):
    # Minimize f using line search
    f_obj = zeros((max_iter, 1))
    alpha = alpha_max * ones((max_iter, 1))
    alpha[0] = 0

    x, f_obj[iter] = x_0.copy(), f[0](x_0)
    while norm(f[1](x)) > tol and iter < max_iter:
        iter += 1
        p = step_dir(f, x, method)
        alpha[iter], f_obj[iter] = step_length(f, x, p, alpha[iter], rho, c)
        x += alpha[iter] * p

    f_obj, alpha = f_obj[0:iter], alpha[0:iter]
    return x, f_obj, alpha, iter

def step_dir(f, x_k, method):
    # Return a unit direction of search
    if method == 'newton' or method == 'n':
        p = - dot(inv(f[2](x_k)), f[1](x_k))
    elif method == 'quasi-newton' or method == 'qn':
        if not f[2][3]:
            f[2][0] = inv(f[2][0])
            f[2][2], f[2][3] = f[1](x_k), True
        else:
            n = x_k.ndim
            x_k_1, g_x_k_1, g_x_k = f[2][1], f[2][2], f[1](x_k)
            s, y = x_k - x_k_1, g_x_k - g_x_k_1
            rho = 1 / dot(y.T, s)
            A, B = eye(n) - rho*dot(s, y.T), eye(n) - rho*dot(y, s.T)
            f[2][0] = dot(A, dot(f[2][0], B)) + rho*dot(s, s.T)
            f[2][1], f[2][2] = x_k.copy(), g_x_k
        p = - dot(f[2][0], f[2][2])
    elif method == 'cg-fr' or method == 'cg':
        if not f[2][1]:
            f[2][0], f[2][1] = -f[1](x_k), True
            p = f[2][0].copy()
        else:
            g_x_k, g_x_k_1 = f[1](x_k), f[2][0]
            beta = norm(g_x_k) / norm(g_x_k_1)
            f[2][0] = -g_x_k + beta*g_x_k_1
            p = f[2][0].copy()
    else:
        p = - f[1](x_k)
    p = p / norm(p)
    return p

def step_length(f, x_k, p_k, alpha_max=1, rho=0.5, c=0.5):
    # Return the step length based on first Wolfe condition
    alpha = alpha_max
    f_x_k = f[0](x_k)
    while (f[0](x_k + alpha * p_k) > 
           f_x_k + c * alpha * dot(f[1](x_k).T, p_k)):
        alpha = rho * alpha
    return alpha, f_x_k

def trust_region_min(f, x_0, method, delta_max=1, eta=1.0/8, iter=0,
                     tol=1e-4, max_iter=1e6):
    # Minimize f with trust region algorithm
    f_obj = zeros((max_iter, 1))
    delta = delta_max * ones((max_iter, 1))

    x, f_obj[iter] = x_0.copy(), f[0](x_0)
    while norm(f[1](x)) > tol and iter < max_iter:
        iter += 1
        p = cauchy_point(f, x, delta[iter])
        approx_at_p = (f_obj[iter - 1] + dot(f[1](x).T, p) +
                       0.5*dot(p.T, dot(f[2](x), p)))
        rho = (f_obj[iter - 1] - f[0](x + p)) / (f_obj[iter - 1] - approx_at_p)
        if rho < 0.25:
            delta[iter] = 0.25*delta[iter - 1]
        else:
            if rho > 0.75 and norm(p) == delta[iter - 1]:
                delta[iter] = min(2*delta[iter - 1], delta_max)
            else:
                delta[iter] = delta[iter - 1]
        if rho > eta:
            x += p
            f_obj[iter] = f[0](x)
        else:
            f_obj[iter] = f_obj[iter - 1]

    f_obj, delta = f_obj[0:iter], delta[0:iter]
    return x, f_obj, delta, iter

def cauchy_point(f, x_k, delta):
    # Compute step direction for trust-region optimization
    g_k, B_k = f[1](x_k), f[2](x_k)
    p_s = -delta*g_k/norm(g_k)
    D = dot(g_k.T, dot(B_k, g_k))
    if D <= 0:
        tao = 1
    else:
        tao = min(1, norm(g_k)**3 / delta / D)
    p_c = tao*p_s
    return p_c

# Visualize iteration
def plot_results(filename, x_p, f_obj, alpha, iter, x_0, legend, tol=1e-4):
    marker = 'ovsx><'
    color = 'bgrcm'
    for i in range(len(legend)):
        plt.loglog(range(iter[i]), f_obj[i], lw=2.5, marker=marker[i],
                     fillstyle='full', ms=8.0, color=color[i])
    plt.title('Minimizing Rosenbrock Function, tol %.0e and x_0 ' % tol + x_0, fontsize=12)
    plt.xlabel('Num Iters', fontsize=12)
    plt.ylabel('Objective function', fontsize=12)
    plt.legend(legend, fontsize=10)
    plt.savefig(filename)
    plt.close()

# Function to minimize, its gradient and hessian

def rb_function(x):
    return array([(100*(x[1, 0] - x[0, 0]**2)**2 + (1 - x[0, 0])**2)])

def rb_gradient(x):
    return array([[2*x[0, 0] - 400*x[0, 0] * (-x[0, 0]**2 + x[1, 0]) - 2],
                  [200*(x[1, 0] - x[0, 0]**2)]])

def rb_hessian(x):
    return array([[(2 + 1200*x[0, 0]**2 - 400*x[1, 0]), (-400*x[0, 0])],
                  [(-400*x[0]), 200]])

hw4_1()
