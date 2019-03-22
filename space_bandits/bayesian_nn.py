"""Define the abstract class for Bayesian Neural Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class BayesianNN(nn.Module):
    """A Bayesian neural network keeps a distribution over neural nets."""

    def __init__(self):
        pass

    def build_model(self):
        pass

    def train(self, data):
        pass

    def sample(self, steps):
        pass
