import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def logistic_func(x):

    ###################################################################
    # YOUR CODE HERE!
    # Output: logistic(x)
    L = 1 / (1 + np.exp(-x))
    ####################################################################

    return L

def train(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    # Initialize weights
    N = X_train.shape[0]
    dim = X_train.shape[1]
    y_train = y_train.reshape((N, 1))
    weights = np.random.uniform(-0.01, 0.01, [dim+1, 1])
    w = weights[1:, :]
    w0 = weights[0, :]

    # Iterate until weights converge
    while True:
        d_weights = np.zeros(dim + 1).reshape(dim+1, 1)
        for t in range(N):
            y = logistic_func(X_train[t].dot(w) + w0)
            for j in range(dim + 1):
                d_weights[j] += (y_train[t] - y) * (1 if j == 0 else X_train[t][j-1])

        ini_weights = weights
        for j in range(dim + 1):
            np.add(weights[j], LearningRate * d_weights[j], out=weights[j])

        if np.allclose(ini_weights, weights, atol=tol, rtol=0):
            break
    ####################################################################

    return weights

def train_matrix(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    N = X_train.shape[0]
    X_train = np.insert(X_train, 0, 1, axis=1)
    dim = X_train.shape[1]
    y_train = y_train.reshape(N, 1)
    weights = np.random.uniform(-0.01, 0.01, [dim, 1])

    while True:
        y = logistic_func(X_train.dot(weights))
        d_weights = np.subtract(y_train, y).T.dot(X_train).reshape(dim, 1)
        ini_weights = weights
        weights += LearningRate * d_weights

        if np.allclose(ini_weights, weights, atol=tol, rtol=0):
            break
    ####################################################################

    return weights

def predict(X_test, weights):

    ###################################################################
    # YOUR CODE HERE!
    # The predict labels of all points in test dataset.
    N = X_test.shape[0]
    X_test = np.insert(X_test, 0, 1, axis=1)
    p_x = logistic_func(X_test.dot(weights))

    predictions = np.array([1 if p_x[t] >= 0.5 else 0 for t in range(N)])
    ####################################################################

    return predictions

def plot_prediction(X_test, X_test_prediction):
    X_test1 = X_test[X_test_prediction == 0, :]
    X_test2 = X_test[X_test_prediction == 1, :]
    plt.scatter(X_test1[:, 0], X_test1[:, 1], color='red')
    plt.scatter(X_test2[:, 0], X_test2[:, 1], color='blue')
    plt.show()


#Data Generation
n_samples = 1000

centers = [(-1, -1), (5, 10)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

# Experiments
w = train(X_train, y_train)

X_test_prediction = predict(X_test, w)
plot_prediction(X_test, X_test_prediction)
plot_prediction(X_test, y_test)

wrong = np.count_nonzero(y_test - X_test_prediction)
print ('Number of wrong predictions is: ' + str(wrong))
