import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = np.multiply(x2, x2)
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    X = np.hstack((x1, x2, x3, x4, x5, x6))
    return X

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    norm_X = X - np.mean(X, axis=0)
    cova = np.cov(norm_X.T)
    D, V = np.linalg.eig(cova)
    index = np.argsort(D)[::-1]
    V = V.T[index]
    D = D[index]
    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    return [V, D]


def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!
    V, D = pca(X)
    print([V, D])
    # [array([[1.70773519e-04, -2.16410974e-02, 4.21111727e-04,
    #          -9.99705685e-01, 1.70773519e-03, 1.08205487e-02],
    #         [-6.13880620e-04, -7.33365948e-04, -9.99980548e-01,
    #          -4.11974521e-04, -6.13880620e-03, 3.66682974e-04],
    #         [-9.95012593e-02, 2.56403373e-03, 6.16777563e-03,
    #          -1.78349842e-03, -9.95012593e-01, -1.28201686e-03],
    #         [2.88952443e-04, 8.94161367e-01, -8.27650102e-04,
    #          -2.41907759e-02, 2.88952443e-03, -4.47080684e-01],
    #         [9.84870275e-01, -6.37665218e-02, 1.39783075e-17,
    #          -1.62840542e-16, -9.84870275e-02, -1.27533044e-01],
    #         [1.00428801e-01, 4.44929926e-01, 4.00672210e-17,
    #          4.23807620e-16, -1.00428801e-02, 8.89859853e-01]]),
    #  array([1.80469530e+05, 3.27681920e+03, 9.24028915e+01, 2.17950420e+01,
    #         7.11550639e-13, 7.13997068e-14])]

    order = [i for i in range(1, len(V)+1)]
    plt.plot(order, D, "o-")
    plt.xlabel("order of eigenvalue")
    plt.ylabel("eigenvalue")
    plt.title("eigenvalues v.s. the order of eigenvalues")
    plt.show()
    summ = sum(D)
    cumsum = []
    pre = 0
    for i, e in enumerate(D):
        cumsum.append((pre+e)/summ)
        pre += e
    plt.plot(order,cumsum,"o-")
    plt.ylabel("POV")
    plt.xlabel("order of eigenvalue")
    plt.title("POV v.s the order of eigenvalues")
    plt.show()
    ####################################################################


if __name__ == '__main__':
    main()

