import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat

# input_layer_size = 400
# num_labels = 10
# data = loadmat(os.path.join('Data', 'ex3data1.mat'))
# X, y = data['X'], data['y'].ravel()
# y[y == 10] = 0
# m = y.size


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X @ theta)
    j = -y * np.log(h) - (1 - y) * np.log(1 - h)
    J = np.sum(j) / m

    J += np.sum(theta[1:] ** 2) * lambda_ / (2 * m)

    grad = np.transpose(X) @ (h - y) / m
    temp = theta
    temp[0] = 0
    grad += temp * lambda_ / m
    return J, grad

# parameters to test
theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
# use test parameters to check function for cost and gradient
print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')
