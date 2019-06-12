from numpy import dot, matmul, ones, zeros, copy
from numpy.linalg import norm
import numpy as np

ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
SEARCH_MAX_ITER = 10
X_SIZE = 200


def get_projection(w):
    return np.vectorize(lambda x: max(x, 0))(w)


def line_search(f, A, b, x, d, gk, alpha0, beta, c, lam):
    alpha_j = alpha0
    for j in range(0, SEARCH_MAX_ITER):
        x_temp = get_projection(x + alpha_j * d)
        if f(A, b, x_temp, lam) <= f(A, b, x, lam) + alpha_j * c * dot(d, gk):
            break
        else:
            alpha_j = alpha_j * beta
    return alpha_j


def steepest_decent(A, b, x_0, lam, max_iter, epsilon):
    objs = list()
    x_sd = copy(x_0)
    for k in range(max_iter):
        gk = g(A, b, x_sd, lam)
        d = -1 * gk
        alpha = line_search(x_func, A, b, x_sd, d, gk, ALPHA_0, BETA, C, lam)
        x_sd = get_projection(x_sd + alpha * d)
        objs.append(x_func(A, b, x_sd, lam))
        if norm(gk) < epsilon:
            print('******************* converged *************************')
            break
    return x_sd[0:X_SIZE] - x_sd[X_SIZE:2 * X_SIZE], objs


def x_func(A, b, x, lam):
    return norm(matmul(A, x) - b, 2) + lam * dot(ones(x.shape), x)


def g(A, b, x, lam):
    g_x = 2 * matmul(A.transpose(), matmul(A, x) - b) + lam * ones(x.shape)
    return g_x / norm(g_x)
