import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    mu = 10
    var = 1
    samples = 1000
    samplesArray = np.random.normal(mu, var, samples)
    uvg = UnivariateGaussian()
    uvg.fit(samplesArray)
    print(uvg.mu_, uvg.var_)
    # Question 2 - Empirically showing sample mean is consistent

    absDist = np.zeros(100, )
    numbOfSummedVal = np.arange(10, 1001, 10)
    sum = 0
    for i in range(0, 1000, 1):
        sum += samplesArray[i]
        if i % 10 == 0 and i != 0:
            arrayLoc = int(i / 10)
            absDist[arrayLoc] = abs((sum / i) - mu)

    plt.scatter(numbOfSummedVal, absDist)
    plt.xlabel("samples size")
    plt.ylabel("distance between the estimated- and true value")
    plt.title("distance between the estimated- and true value as function "
              "of sample size")
    plt.show()
    # Question 3 - Plotting Empirical PDF of fitted model

    pdf = uvg.pdf(samplesArray)
    plt.scatter(samplesArray, pdf)
    plt.xlabel("samples value")
    plt.ylabel("samples pdf")
    plt.title("distribution of pds as function of value")
    plt.show()


def test_multivariate_gaussian():
    #Question 4 - Draw samples and print fitted model

    mu = np.array([0, 0, 4, 0])
    var = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    ndarr = np.random.multivariate_normal(mu, var, 1000)
    mvg = MultivariateGaussian().fit(ndarr)
    print(mvg.mu_)
    print(mvg.cov_)
    #Question 5 - Likelihood evaluation

    numOfSamples = 200
    f1 = np.linspace(-10, 10, numOfSamples)
    f3 = np.linspace(-10, 10, numOfSamples)
    counta = 0
    countb = 0
    heatMat = np.zeros((numOfSamples, numOfSamples))
    curr_mu = np.array([0, 0, 0, 0])

    for i in f1:
        for j in f3:
            curr_mu[0] = i
            curr_mu[2] = j
            heat = mvg.log_likelihood(curr_mu, var, ndarr)
            heatMat[counta][countb] = heat
            countb += 1
        countb = 0
        counta += 1

    fig, ax = plt.subplots()
    min, max = np.min(heatMat), np.max(heatMat)
    c = ax.pcolormesh(f1, f3, heatMat, cmap='RdBu', vmin=min, vmax=max)
    ax.set_title('log likelihood heapMap')
    ax.axis([f1.min(), f1.max(), f3.min(), f3.max()])
    fig.colorbar(c, ax=ax)
#     #Question 6 - Maximum likelihood

    val = np.unravel_index(heatMat.argmax(), heatMat.shape)
    print(np.linspace(-10, 10, numOfSamples)[val[1]],
          np.linspace(-10, 10, numOfSamples)[val[0]])
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
