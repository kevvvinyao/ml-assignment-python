import numpy as np
from scipy import optimize
from scipy.io import loadmat
import os

input_layer_size = 400
num_labels = 10
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size
lambda_ = 0.1

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


def oneVsAll(X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape  # 5000 * 400

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))  # 10 * 401

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    '''
    These code below is my attempt for gradient-descent algorithm
    But you can use optimize.minimize function to find an optimal value efficiently
    '''
    # learning_rate = 0.1
    # Y = np.zeros((num_labels, m))
    # # all_theta = np.random.rand(num_labels, n + 1)
    # j = 0
    # for i in y:
    #     Y[i, j] = 1
    #     j += 1
    # for j in range(10):
    #     for i in range(50000):
    #         h = utils.sigmoid(X @ all_theta[j, :])
    #         grad = np.transpose(X) @ (h - Y[j, :]) / m
    #         temp = all_theta[j, :]
    #         temp[0] = 0
    #         grad += temp * lambda_ / m
    #         all_theta[j, :] -= learning_rate * grad

    for c in np.arange(num_labels):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, (y == c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)

        all_theta[c] = res.x

    # ============================================================
    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    # P = np.zeros((num_labels, m))
    # for i in range(10):
    #     P[i, :] = utils.sigmoid(X @ all_theta[i, :])
    # for i in range(m):
    #     max_index = np.argmax(P[:, i])
    #     p[i] = max_index
    p = np.argmax(sigmoid(X.dot(all_theta.T)), axis=1)

    # ============================================================
    return p


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


all_theta = oneVsAll(X, y, num_labels, lambda_)
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))
print('Expected Accuracy: 95.1%')
