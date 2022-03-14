from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    """
    tests the class of UnivariateGaussian.
    draws 1000 samples with a given estimation and variance, and prints fitted
    models that represent the distances between the estimated excpectation and
    the real one, and the PDF of each one of the samples
    """
    # Question 1 - Draw samples and print fitted model
    trueMu = 10
    trueVar = 1
    arrOfSamples = np.random.normal(trueMu,trueVar,1000)
    estimatorObj = UnivariateGaussian()

    estimatorObj.fit(arrOfSamples) # estimate 1000 samples
    print("Estimated expectation & variance:")
    print(tuple([estimatorObj.mu_ , estimatorObj.var_]))

    # Question 2 - Empirically showing sample mean is consistent

    # takes the same set of samples as before, and plots the absolute distance
    # between the estimated and true size of expectation:
    arrOfMu = create_arr_of_abs_distances(estimatorObj, arrOfSamples, trueMu)
    fig = create_figure_of_estimated_excpectation(arrOfMu) # create figure
    fig.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    estimatorObj.fit(arrOfSamples)  # estimate 1000 samples
    fig2 = create_fig_of_pdf(arrOfSamples, estimatorObj)
    fig2.show()

    # print(estimatorObj.log_likelihood(10,1,arrOfSamples))

def test_multivariate_gaussian():
    """
    tests the class of MultivariateGaussian.
    draws 1000 samples with a given estimation and variance, and prints fitted
    models that represent the distances between the estimated expectation and
    the real one, and the PDF of each one of the samples
    """
    # Question 4 - Draw samples and print fitted model
    trueMu = np.array([0,0,4,0])
    trueCov = np.array([[1,0.2,0,0.5],
                        [0.2,2,0,0],
                        [0,0,1,0],
                        [0.5,0,0,1]])

    arrOfSamples = np.random.multivariate_normal(trueMu, trueCov, 1000)
    estimatorObg = MultivariateGaussian()
    estimatorObg.fit(arrOfSamples)
    print("\nEstimated expectation:")
    print(estimatorObg.mu_)
    print("\nEstimated cov:")
    print(estimatorObg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    logLikelihood = []
    logLikelihoodOfSingleSample = []

    maxLikelihood = float('-inf') # initializing max as smallest number
    # F1 and F3 run between -10 and 10 so will always be bigger than -20:
    maxF1 = -20
    maxF3 = -20

    for i in range(200):
        for j in range(200):
            currentMu = np.array([f1[i], 0, f3[j], 0])
            currentLikelihood = MultivariateGaussian().\
                log_likelihood(currentMu, trueCov, arrOfSamples)
            logLikelihoodOfSingleSample.append(currentLikelihood)
            if currentLikelihood > maxLikelihood:
                maxLikelihood = currentLikelihood
                maxF1 = f1[i]
                maxF3 = f3[j]
        logLikelihood.append(logLikelihoodOfSingleSample)
        logLikelihoodOfSingleSample = []


    fig = go.Figure(data=go.Heatmap(
        z=logLikelihood,
        x=f3,
        y=f1
    ))

    fig.update_layout(
        title="LogLikelihood HeatMap",
        xaxis_title="f3 values",
        yaxis_title="f1 values",
    )

    fig.show()

    # Question 6 - Maximum likelihood
    print("\nMax f1 & f3: ")
    print(tuple([round(maxF1,3) , round(maxF3,3)]))
    print("\nMax LogLikelihood: ")
    print(round(maxLikelihood, 3))


def create_arr_of_abs_distances(estimatorObj, arrOfSamples, trueMu):
    """
    helper function to test_univariate_gaussian, creates an array in size 100,
    where each index i represents the sample size (i // 10 - 1) - going from
    10 to 1000,and contains the abs value between it's estimated expectation
    and the actual expectation
    @param estimatorObj: the object of UnivariateGaussian class
    @param arrOfSamples: arr of all samples
    @param trueMu: actual expectation of the samples
    @return: the array of abs values
    """
    arrOfMu = np.ndarray((100,))
    i = 10
    while (i <= 1000):
        estimatorObj.fit(arrOfSamples[:i])  # estimate acc. to first i samples
        arrOfMu[i // 10 - 1] = np.abs(estimatorObj.mu_ - trueMu)
        i += 10
    return arrOfMu


def create_figure_of_estimated_excpectation(arrOfMu):
    """
    helper function to test_univariate_gaussian, creates a figure in which the
    x axis represents the size of the sample, and the y axis the abs value
    between between it's estimated expectation and the actual expectation
    @param arrOfMu: array which each index represents a different sample size
    and contains the distance between the excpectations
    @return: the figure object created
    """
    fig = go.Figure()
    arrayOfSampleSizes = np.arange(10, 1001, 10) # arr of the sizes of samples
    fig.add_trace(go.Scatter(
        x=arrayOfSampleSizes,
        y=arrOfMu,
        name="Distance between actual and estimated Expectation"
    ))

    fig.update_layout(
        title="Distance between actual and estimated Expectation",
        xaxis_title="Sample size",
        yaxis_title="Expectation distances"
    )

    return fig

def create_fig_of_pdf(arrOfSamples, estimatorObj):
    """
    helper function to test_univariate_gaussian, creates a figure in which the
    x axis represents the samples, and the y axis the PDF of each sample
    @param arrOfSamples: arr of all samples
    @param estimatorObj: the object of UnivariateGaussian class
    @return: the figure object created
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=arrOfSamples,
        y=estimatorObj.pdf(arrOfSamples), # initialize pdf array
        mode="markers"
    ))

    fig.update_layout(
        title="PDF of samples",
        xaxis_title="Samples",
        yaxis_title="PDF of sample"
    )
    return fig

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
