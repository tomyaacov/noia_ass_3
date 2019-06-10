from numpy import dot, matmul, ones, zeros
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

LAM_VALUES = [10.0, 20.0, 30.0, 40.0, 50.0]
MAX_ITER = 5000
ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
SEARCH_MAX_ITER = 10
EPSILON = 0.01


def get_projection(u, v):
    u = np.vectorize(lambda x: max(x, 0))(u)
    v = np.vectorize(lambda x: max(x, 0))(v)
    return u, v


def line_search(f, A, b, u, v, d_u, d_v, gk_u, gk_v, alpha0, beta, c, lam):
    alpha_j = alpha0
    for j in range(0, SEARCH_MAX_ITER):
        u_temp = u + alpha_j * d_u
        v_temp = v + alpha_j * d_v
        u_temp, v_temp = get_projection(u_temp, v_temp)
        #  TODO validate this part
        if f(A, b, u_temp, v_temp, lam) <= f(A, b, u, v, lam) + alpha_j * c * dot(d_u - d_v, gk_u - gk_v):
            break
        else:
            alpha_j = alpha_j * beta
    return alpha_j


def steepest_decent(A, b, u_sd, v_sd, lam, max_iter, epsilon):
    objs, norms = list(), list()
    for k in range(max_iter):
        gk_u, gk_v = g(A, b, u_sd, v_sd, lam)
        d_u, d_v = -1 * gk_u, -1 * gk_v
        alpha = line_search(x_func, A, b, u_sd, v_sd, d_u, d_v, gk_u, gk_v, ALPHA_0, BETA, C, lam)
        u_sd = u_sd + alpha * d_u
        v_sd = v_sd + alpha * d_v
        u_sd, v_sd = get_projection(u_sd, v_sd)
        r = b - matmul(A, u_sd - v_sd)
        objs.append(x_func(A, b, u_sd, v_sd, lam))
        r_norm = norm(r)
        b_norm = norm(b)
        norms.append(r_norm)
        if r_norm / b_norm < epsilon:
            print('******************* converged *************************')
            break
    return u_sd - v_sd, objs, norms


def x_func(A, b, u, v, lam):
    return norm(matmul(A, u - v) - b, 2) + lam * matmul(ones(u.shape), u + v)


def g(A, b, u, v, lam):
    # TODO add condition for size of norm
    g_u = 2 * matmul(A.transpose(), matmul(A, u - v) - b) + lam * ones(u.shape)
    g_v = -2 * matmul(A.transpose(), matmul(A, u - v) - b) + lam * ones(v.shape)
    return g_u / norm(g_u), g_v / norm(g_v)
    # return g_u, g_v


def save_plot(results, x_label, y_label, name):
    plt.figure()
    for lambda_res in results:
        plt.plot(results[lambda_res], label=lambda_res)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{name}.pdf', bbox_inches="tight")


def main():
    results_objs, results_norms = dict(), dict()
    # init params
    u_sd, v_sd = np.random.normal(0, 1, 200), np.random.normal(0, 1, 200)
    A = np.random.normal(0, 1, (100, 200))
    x = zeros(200)
    indices = np.random.choice(200, 20)
    x[indices] = np.random.normal(0, 1, 20)
    b = matmul(A, x) + np.random.normal(0, 0.1, 100)

    for lam in LAM_VALUES:
        print(f'starting sd for lambda={lam}...')
        x_sol, objs, norms = steepest_decent(A, b, u_sd, v_sd, lam, MAX_ITER, EPSILON)
        print(f'lam={lam}, {np.count_nonzero(x_sol)} non zero')
        results_objs[f'lam={lam}'], results_norms[f'lam={lam}'] = objs, norms

    save_plot(results_objs, 'Iteration', 'Objective', 'Objectives')
    save_plot(results_norms, 'Iteration', 'Residual Norm', 'Norms')


if __name__ == '__main__':
    main()
