import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"
TRUE_EXPECTATION=10
TRUE_VARIANCE=1
NUM_SAMPLES= 1000
MULTY_EXPECTATION = np.array([0,0,4,0])
MULTY_SIGMA = np.array([[1,0.2,0,0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(TRUE_EXPECTATION, TRUE_VARIANCE, NUM_SAMPLES)
    uni_gaussian=UnivariateGaussian()
    uni_gaussian.fit(samples)
    print('Estimated expectation an variance: (',uni_gaussian.mu_,',', uni_gaussian.var_,')')

    # Question 2 - Empirically showing sample mean is consistent

    def find_expectation(n_samples):
        return uni_gaussian.fit(samples[0:n_samples]).mu_

    samples_count = np.arange(10, NUM_SAMPLES + 10, 10)

    vec=np.vectorize(find_expectation)
    expectations_diff=abs(vec(samples_count)-TRUE_EXPECTATION)


    fig = px.scatter(x=samples_count, y=expectations_diff)
    fig.update_layout(title="The effect of the samples amount in the expectation accuracy",
                      xaxis_title="Samples size", yaxis_title="Absolute Expectation Distance")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    arr_pdfs = uni_gaussian.pdf(samples)
    fig2 = px.scatter(x=samples, y=arr_pdfs)
    fig2.update_layout(title="The PDFs values as function of the Sample values", xaxis_title="Sample Value",
                      yaxis_title="PDF Value")
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(MULTY_EXPECTATION, MULTY_SIGMA, NUM_SAMPLES)
    multy_gaussian = MultivariateGaussian()
    multy_gaussian.fit(samples)
    print("Estimated expectation: ",multy_gaussian.mu_)
    print("Estimated covariance: \n",multy_gaussian.cov_)


    # Question 5 - Likelihood evaluation
    array_f = np.linspace(-10, 10, 200)
    results = np.zeros(shape=(200,200))
    count_i = 0
    for f1 in array_f:
        count_j = 0
        for f3 in array_f:
            curr_mu = np.array([f1, 0, f3, 0])
            results[count_i][count_j] = MultivariateGaussian.log_likelihood(curr_mu,MULTY_SIGMA,samples)
            count_j+=1
        count_i+=1

    fig = go.Figure(go.Heatmap(x=array_f, y=array_f, z=results, colorbar=dict(title = "Log Likelihood")))
    fig.update_layout(title="The Log Likelihood Heatmap of the expectation [f1, 0, f3, 0]:",
                                                                xaxis_title ="f3 values",
                                                                yaxis_title ="f1 values")
    fig.show()


    # Question 6 - Maximum Likelihood
    max_ind = np.unravel_index(np.argmax(results), results.shape)
    print("The model achieved the maximum log likelihood is: [",round(array_f[max_ind[0]],3), ",0,",round(array_f[max_ind[1]],3),",0]")



if __name__ == '__main__':
     np.random.seed(0)
     test_univariate_gaussian()
     test_multivariate_gaussian()

