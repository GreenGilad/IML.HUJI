
from IMLearn.learners.classifiers.perceptron import Perceptron
import numpy as np

if __name__ == '__main__':
    a = np.full((5, 1), 0)
    b = np.full((5, 1), 1)
    c = Perceptron()
    c.fit(a,b)
