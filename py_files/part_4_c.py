from numpy import matmul, ones, zeros
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
from py_files.part_4_b import steepest_decent

LAM_VALUES = [2.0, 4.0, 8.0, 16.0]
X_SIZE = 200
MAX_ITER = 1000
EPSILON = 0.01


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
    x_0 = np.concatenate((ones(X_SIZE), zeros(X_SIZE)), axis=0)
    A = np.concatenate((A_orig, -A_orig), axis=1)

    for lam in LAM_VALUES:
        print(f'starting sd for lambda={lam}...')
        x_sol, objs = steepest_decent(A, b, x_0, lam, MAX_ITER, EPSILON)
        print(f'lam={lam}, {np.count_nonzero(x_sol)} non zero, norm(x_sol-x)={norm(x_sol-x)}')
        results_objs[f'lam={lam}'] = objs

    save_plot(results_objs, 'Iteration', 'Objective2', 'Objectives2')


if __name__ == '__main__':
    main()
