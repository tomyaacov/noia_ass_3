from numpy import dot, array, copy, round, array_equal
from random import randint

M_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_ITER = 10000
ALPHA_0 = 1.0
BETA = 0.5
C = 0.0001
X_STAR = [1.42, 1.72]
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


def steepest_decent(f, x_0, max_iter, m):
    x_sd = copy(x_0)
    for k in range(max_iter):
        gk = g(x_sd, m)
        d_sd = -1 * gk
        alpha_sd = line_search(f, x_sd, d_sd, gk, ALPHA_0, BETA, C, m)
        x_last = copy(x_sd)
        x_sd = x_sd + alpha_sd * d_sd
        if array_equal(round(x_sd, 2), round(x_last, 2)):
            break
    return round(x_sd, 2)


def x_func(x, m):
    return pow(x[0] + x[1], 2) - 10 * (x[0] + x[1]) + m * (
           + pow(3 * x[0] + x[1] - 6, 2)
           + pow(max(pow(x[0], 2) + pow(x[1], 2) - 5, 0), 2)
           + pow(max(-1 * x[0], 0), 2)
    )


def x_func_orig(x):
    return round(pow(x[0] + x[1], 2) - 10 * (x[0] + x[1]), 2)


def first_constraint(x):
    return round(3 * x[0] + x[1] - 6, 2)


def second_constraint(x):
    return round(pow(x[0], 2) + pow(x[1], 2) - 5, 2)


def third_constraint(x):
    return round(-1 * x[0], 2)


def check_constraint(x):
    print(f'**** values for x={x}')
    print(f'function value is {pow(x[0] + x[1], 2) - 10 * (x[0] + x[1])}')
    print(f'first constraint is {3 * x[0] + x[1] - 6}')
    print(f'second constraint is {pow(x[0], 2) + pow(x[1], 2) - 5}')
    print(f'third constraint is {-1 * x[0]}')
    print('')


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
        print(f'm={_m}, x*={x_sol}, f(x)={x_func_orig(x_sol)}, c1={first_constraint(x_sol)},'
              f' c2={second_constraint(x_sol)}, c3={third_constraint(x_sol)}')


if __name__ == '__main__':
    main()
