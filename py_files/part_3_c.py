from copy import copy
from numpy.linalg import norm
from numpy import matmul


def projected_coordinate_descent(H, g, a, b, x_0, alpha, max_iter, epsilon):
    x = copy(x_0)
    for _ in range(max_iter):
        prev_x = copy(x)
        grad = matmul(H, x) - g
        z = x - alpha * grad
        for i in range(x.shape[0]):
            if z[i] < a[i]:
                x[i] = a[i]
            elif z[i] > b[i]:
                x[i] = b[i]
            else:
                x[i] = z[i]
        if norm(x - prev_x) < epsilon:
            break
    return x


def objective(H, x, g):
    return 0.5 * matmul(matmul(x.T, H), x) - matmul(x.T, g)
