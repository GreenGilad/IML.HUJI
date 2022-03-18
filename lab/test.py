import numpy as np
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mu = [0, 0, 4, 0]
    var = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    ndarr = np.random.multivariate_normal(mu, var, 1000)
    print(np.ndarr[0])
