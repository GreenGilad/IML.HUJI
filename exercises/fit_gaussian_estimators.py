from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    ndarr = np.random.normal(mu, var, 1000)
    uvg = UnivariateGaussian()
    uvg.fit(ndarr.transpose())
    print(uvg.mu_, uvg.var_)

    # Question 2 - Empirically showing sample mean is consistent
    y_arr = np.zeros(100,)
    arr_x = np.arange(10, 1001, 10)
    sum = 0
    for i in range(0, 1000, 1):
        sum += ndarr[i]
        if i % 10 == 0 and i != 0:
            tmp = int(i / 10)
            y_arr[tmp] = abs((sum / i) - mu)
    plt.scatter(x=arr_x, y=y_arr)
    plt.title("Fit graph")
    plt.xlabel("sample size")
    plt.ylabel("distance between estimated nad true value")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uvg.pdf(ndarr)
    plt.scatter(x=ndarr, y=pdf)
    plt.title("Empirical PDF of fitted model")
    plt.xlabel("sample value")
    plt.ylabel("sample pdf")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    var = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    ndarr = np.random.multivariate_normal(mu, var, 1000)
    mvg = MultivariateGaussian()
    mvg.fit(ndarr)
    print(mvg.mu_)
    print(mvg.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    xv, yv = np.meshgrid(f, f)
    heatMap = np.ndarray((200, 200))
    maxval = None

    for ind, i in enumerate(f):
        for jnd, j in enumerate(f):
            curr_mu = np.array([i, 0, j, 0])
            heat_val = mvg.log_likelihood(curr_mu, var, ndarr)
            if maxval is None:
                maxval = heat_val
            if heat_val > maxval:
                maxval = heat_val
                maxf1 = ind
                maxf2 = jnd
            heatMap[ind][jnd] = int(heat_val)
    fig, ax = plt.subplots()

    c = ax.pcolormesh(xv, yv, heatMap, cmap='RdBu',
                      vmin=np.min(heatMap),
                      vmax=np.max(heatMap))
    ax.set_title('log likelihood heatmap')
    ax.axis([xv.min(), xv.max(), yv.min(), yv.max()])
    fig.colorbar(c, ax=ax)

    plt.show()

    # Question 6 - Maximum likelihood
    #round to 3 decimal points and then print (maxf1,maxf2)
    print("max values: " + str(maxval) + " " + str(round(f[maxf1], 3)) + " " + str(round(f[maxf2], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
