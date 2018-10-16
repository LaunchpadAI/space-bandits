from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from .bandit_algorithm import BanditAlgorithm
from .contextual_dataset import ContextualDataset
from .neural_bandit_model import NeuralBanditModel
from .linear import LinearBandits
from .neural_linear import NeuralBandits
import tensorflow as tf

import os
import pickle
import shutil

__version__ = '0.0.93'

def load_linear_model(path):
    """loads linear model from path argument"""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model    

def load_neural_model(path):
    """loads linear model from path argument"""
    shutil.unpack_archive(path, extract_dir='tmp')
    pickle_path = os.path.join('tmp', 'master.pkl')
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    model.hparams = model.get_hparams()
    model.bnn = NeuralBanditModel(model.arguments['optimizer'], model.hparams, '{}-bnn'.format(model.name))
    weights_path = os.path.join('tmp', 'weights')
    with model.bnn.graph.as_default():
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, weights_path)
    os.remove(pickle_path)
    os.remove(weights_path + '.data-00000-of-00001')
    os.remove(os.path.join('tmp', 'checkpoint'))
    os.remove(os.path.join('tmp', 'weights.meta'))
    os.remove(os.path.join('tmp', 'weights.index'))
    os.rmdir('tmp')
    return model
