from numpy import dot, array
from random import randint

M_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER = 10000
ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
X_STAR = [150/89, 84/89]
SEARCH_MAX_ITER = 100


def line_search(f, x, d, gk, alpha0, beta, c, m):
    alpha_j = alpha0
    for j in range(0, SEARCH_MAX_ITER):
        x_temp = x + alpha_j * d
        if f(x_temp, m) <= f(x, m) + alpha_j * c * dot(d, gk):
            break
        else:
            alpha_j = alpha_j * beta
    return alpha_j


def steepest_decent(f, x_sd, max_iter, m):
    print('starting sd....')
    for k in range(max_iter):
        gk = g(x_sd, m)
        d_sd = -1 * gk
        alpha_sd = line_search(f, x_sd, d_sd, gk, ALPHA_0, BETA, C, m)
        x_sd = x_sd + alpha_sd * d_sd
        # TODO fix the convergence condition?
        if round(x_sd[0], 2) == round(X_STAR[0], 2) and round(x_sd[1], 2) == round(X_STAR[1], 2):
            print('found optimum!!!')
            break
    return x_sd


def x_func(x, m):
    return pow(x[0] + x[1], 2) - 10 * (x[0] + x[1]) + m * (
           + pow(3 * x[0] + x[1] - 6, 2)
           + pow(max(pow(x[0], 2) + pow(x[1], 2) - 5, 0), 2)
           + pow(max(-1 * x[0], 0), 2)
    )


def x_func_orig(x):
    return pow(x[0] + x[1], 2) - 10 * (x[0] + x[1])


def check_constraint(x):
    # return pow(3 * x[0] + x[1] - 6, 2)
    # return pow(max(pow(x[0], 2) + pow(x[1], 2) - 5, 0), 2)
    # return pow(max(-1 * x[0], 0), 2)
    return pow(x[0] + x[1], 2) - 10 * (x[0] + x[1])


def g(x, m):
    x_g = array([0.0, 0.0])

    x_g[0] = 2 * (x[0] + x[1]) - 10 + m * (
        + 6 * (3 * x[0] + x[1] - 6)
        + 0 if (pow(x[0], 2) + pow(x[1], 2) - 5) <= 0 else 4 * x[0] * (pow(x[0], 2) + pow(x[1], 2) - 5)
        + 0 if (-1 * x[0]) <= 0 else 2 * x[0]
    )

    x_g[1] = 2 * (x[0] + x[1]) - 10 + m * (
        + 2 * (3 * x[0] + x[1] - 6)
        + 0 if (pow(x[0], 2) + pow(x[1], 2) - 5) <= 0 else 4 * x[1] * (pow(x[0], 2) + pow(x[1], 2) - 5)
    )
    return x_g


def main():
    x_0 = array([float(randint(0, 9)), float(randint(0, 9))])  # X_STAR
    for _m in M_VALUES:
        x_sol = steepest_decent(x_func, x_0, MAX_ITER, _m)
        print(f'x_sd={x_sol}')
        print(f'f(x)={x_func_orig(x_sol)}')


if __name__ == '__main__':
    main()
