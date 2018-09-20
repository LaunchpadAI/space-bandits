"""Define the abstract class for Bayesian Neural Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BayesianNN(object):
    """A Bayesian neural network keeps a distribution over neural nets."""

    def __init__(self, optimizer):
        pass

    def build_model(self):
        pass

    def train(self, data):
        pass

    def sample(self, steps):
        pass