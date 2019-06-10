from numpy import dot, matmul, ones, zeros, copy
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

LAM_VALUES = [2.0, 4.0, 8.0, 16.0]
MAX_ITER = 1000
ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
SEARCH_MAX_ITER = 10
EPSILON = 0.01
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


def save_plot(results, x_label, y_label, name):
    plt.figure()
    for lambda_res in results:
        plt.plot(results[lambda_res], label=lambda_res)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{name}.pdf', bbox_inches="tight")


def main():
    # measurements
    results_objs = dict()

    # init params
    A_orig = np.random.normal(0, 1, (100, X_SIZE))
    x = zeros(X_SIZE)
    indices = np.random.choice(X_SIZE, 20)
    x[indices] = np.random.normal(1, 1, 20)
    b = matmul(A_orig, x) + np.random.normal(0, 0.05, 100)

    # params for sd
    x_0 = np.concatenate((ones(X_SIZE), zeros(X_SIZE)), axis=0)#np.random.normal(0, 1, X_SIZE * 2)
    A = np.concatenate((A_orig, -A_orig), axis=1)

    for lam in LAM_VALUES:
        print(f'starting sd for lambda={lam}...')
        x_sol, objs = steepest_decent(A, b, x_0, lam, MAX_ITER, EPSILON)
        print(f'lam={lam}, {np.count_nonzero(x_sol)} non zero, norm(x_sol-x)={norm(x_sol-x)}')
        results_objs[f'lam={lam}'] = objs

    save_plot(results_objs, 'Iteration', 'Objective', 'Objectives')


if __name__ == '__main__':
    main()
