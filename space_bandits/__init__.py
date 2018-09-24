from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from .bandit_algorithm import BanditAlgorithm
from .contextual_dataset import ContextualDataset
from .neural_bandit_model import NeuralBanditModel
import tensorflow as tf

import os
import pickle
import shutil

def init_linear_model(
        num_actions,
        context_dim,
        name='linear_model',
        a0=6,
        b0=6,
        lambda_prior=0.25,
        initial_pulls=2
    ):
    """
    Initializes a bayesian-linear contextual bandits model.
    
        num_actions (int): the number of actions in problem
        
        context_dim (int): the length of context vector
        
        a0 (int): initial alpha value (default 6)
        
        b0 (int): initial beta_0 value (default 6)
        
        lambda_prior (float): lambda prior parameter(default 0.25)
    """
    from .linear import LinearFullPosteriorSampling
    hparams_linear = tf.contrib.training.HParams(
                        num_actions=num_actions,
                        context_dim=context_dim,
                        a0=a0,
                        b0=b0,
                        lambda_prior=lambda_prior,
                        initial_pulls=initial_pulls
    )
    model = LinearFullPosteriorSampling(name, hparams_linear)
    return model

def load_linear_model(path):
    """loads linear model from path argument"""
    from .linear import LinearFullPosteriorSampling
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model    

def init_neural_model(
        num_actions,
        context_dim,
        name='neural_model',
        do_scaling=False,
        init_scale=0.3,
        activation='relu',
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=100,
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=1,
        training_freq_network=50,
        training_epochs=100,
        memory_size=-1,
        a0=6,
        b0=6,
        lambda_prior=0.25,
    ):
    """
    Initializes a neural linear contextual bandits model.
    
        num_actions (int): the number of actions in problem
        
        context_dim (int): the length of context vector
        
        init_scale (float): variance for neural network weights initialization (default 0.3)
        
        activation (tensorflow activation function): activation function for neural network layers (default tf.nn.relu)
        
        layer_sizes (list of integers): defines neural network architecture: n_layers = len(layer_sizes), value is per-layer width. (default [50])
        
        batch_size (integer): batch size for neural network training (default 512)
        
        activate_decay (bool): whether to use learning rate decay (default True),
        
        initial_lr (float): initial learning rate for neural network training (default 0.1)
        
        max_grad_norm (float): maximum gradient value for gradient clipping (default 5.0)
        
        show_training (bool): whether to show details of neural network training
        
        freq_summary (int): summary output frequency in number of steps (default 1000)
        
        buffer_s (int): buffer size for retained examples (default -1)
        
        initial_pulls (int): number of random pulls before greedy behavior (default 100),
        
        reset_lr (bool) = whether to reset learning rate on each nn training (default True),
        
        lr_decay_rate (float): learning rate decay for nn updates (default 0.5)
        
        training_freq (int): frequency for updates to bayesian linear regressor (default 1)
        
        training_freq_network (int): frequency of neural network re-trainings (default 50)
        
        training_epochs (int): number of epochs in each neural network re-training (default 100)
        
        a0 (int): initial alpha value (default 6)
        
        b0 (int): initial beta_0 value (default 6)
        
        lambda_prior (float): lambda prior parameter(default 0.25)
    """
    arguments = {
        'num_actions':num_actions,
        'context_dim':context_dim,
        'name':name,
        'init_scale':init_scale,
        'activation':activation,
        'layer_sizes':layer_sizes,
        'batch_size':batch_size,
        'activate_decay':activate_decay,
        'initial_lr':initial_lr,
        'max_grad_norm':max_grad_norm,
        'show_training':show_training,
        'freq_summary':freq_summary,
        'buffer_s':buffer_s,
        'initial_pulls':initial_pulls,
        'reset_lr':reset_lr,
        'lr_decay_rate':lr_decay_rate,
        'training_freq':training_freq,
        'training_freq_network':training_freq_network,
        'training_epochs':training_epochs,
        'memory_size':memory_size,
        'a0':a0,
        'b0':b0,
        'lambda_prior':lambda_prior,
    }
    
    from .neural_linear import NeuralLinearPosteriorSampling
    model = NeuralLinearPosteriorSampling(name, arguments, do_scaling=do_scaling)
    
    return model

def load_neural_model(path):
    """loads linear model from path argument"""
    from .linear import LinearFullPosteriorSampling
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