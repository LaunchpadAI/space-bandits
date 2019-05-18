from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma
import torch

from .bandit_algorithm import BanditAlgorithm
from .contextual_dataset import ContextualDataset
from .neural_bandit_model import NeuralBanditModel
from .linear import LinearBandits
from .neural_linear import NeuralBandits

import os
import pickle
import shutil

__version__ = '0.0.93'

def load_model(path):
    """loads model from path argument"""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
