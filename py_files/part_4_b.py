from numpy import dot, array, matmul, ones, zeros
from numpy.linalg import norm
import numpy as np
from random import randint

LAM_VALUES = [0.0001, 0.001, 0.01, 1.0, 10.0]
MAX_ITER = 1000
ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
SEARCH_MAX_ITER = 10
EPSILON = 0.01


def get_projection(x, lam):
    x = np.vectorize(lambda x_i: x_i - lam if x_i > lam else(0 if -lam <= x_i <= lam else x_i + lam))(x)
    return x


def line_search(f, A, b, x, d_x, gk_x, alpha0, beta, c, lam):
    alpha_j = alpha0
    for j in range(0, SEARCH_MAX_ITER):
        x_temp = x + alpha_j * d_x
        x_temp = get_projection(x_temp, lam)
        if f(A, b, x_temp, lam) <= f(A, b, x_temp, lam) + alpha_j * c * dot(d_x, gk_x):
            break
        else:
            alpha_j = alpha_j * beta
    return alpha_j


def steepest_decent(A, b, x_sd, lam, max_iter, epsilon):
    objs = list()
    for k in range(max_iter):
        gk_x = g(A, b, x_sd)
        d_x = -1 * gk_x
        alph = line_search(x_func, A, b, x_sd, d_x, gk_x, ALPHA_0, BETA, C, lam)
        x_sd = x_sd + alph * d_x
        x_sd = get_projection(x_sd, lam)
        r = b - matmul(A, x_sd)
        objs.append(x_func(A, b, x_sd, lam))
        if norm(r) / norm(b) < epsilon:
            break
    return x_sd, objs


def x_func(A, b, x, lam):
    return norm(matmul(A, x) - b, 2) + lam * norm(x, 1)


def g(A, b, x):
    return -1 * matmul(A.transpose(), b - matmul(A, x))


def main():
    A = np.random.rand(100, 200)
    x = zeros(200)
    indices = np.random.choice(200, 20)
    x[indices] = np.random.rand(20)
    print(f'x={x}')
    b = matmul(A, x) + np.random.normal(0, 0.1, 100)

    x_0 = zeros(200)
    for lam in LAM_VALUES:
        x_sol, objs = steepest_decent(A, b, x_0, lam, MAX_ITER, EPSILON)
        print(f'lam={lam}, x_sd={x_sol}')


if __name__ == '__main__':
    main()
