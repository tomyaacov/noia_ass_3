from numpy import array
from py_files.part_3_c import projected_coordinate_descent, objective
from numpy.random import uniform

# parameters setting
H = array([[5, -1, -1, -1, -1],
           [-1, 5, -1, -1, -1],
           [-1, -1, 5, -1, -1],
           [-1, -1, -1, 5, -1],
           [-1, -1, -1, -1, 5]])
g = array([18, 6, -12, -6, 18])
a = array([0, 0, 0, 0, 0])
b = array([5, 5, 5, 5, 5])

# algorithm parameters setting
epsilon = 1e-4
alpha = 0.1
max_iter = 1000
x_0 = uniform(a, b, a.shape[0])

# running the algorithm
x = projected_coordinate_descent(H, g, a, b, x_0, alpha, max_iter, epsilon)
print(x, "\\")
print(objective(H, x, g), "\\")


