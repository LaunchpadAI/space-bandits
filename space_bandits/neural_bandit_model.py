"""Define a family of neural network architectures for bandits.
The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import tensorflow as tf

from .bayesian_nn import BayesianNN


class NeuralBanditModel(BayesianNN):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name, logdir='.'):
        """Saves hyper-params and builds the Tensorflow graph."""
        self.logdir = logdir
        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = self.hparams["verbose"]
        self.times_trained = 0
        self.lr = self.hparams["initial_lr"]
        self.layers = []
        self.build_model()

    def build_layer(self, x, num_units):
        """Builds a layer with input x; dropout and layer norm if specified."""

        init_s = self.hparams['init_scale']

        layer_n = self.hparams.get("layer_norm", False)
        dropout = self.hparams.get("use_dropout", False)

        nn = tf.contrib.layers.fully_connected(
            x,
            num_units,
            activation_fn=tf.nn.relu,
            normalizer_fn=None if not layer_n else tf.contrib.layers.layer_norm,
            normalizer_params={},
            weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
        )

        return nn

    def forward(self, x):
        """forward pass of the neural network"""
        for layer in self.layers:
            x = layer(x)
        return x

    def build_model(self):
        """
        Defines the actual NN model with fully connected layers.
        """
        for layer in self.hparams['layer_sizes']:
            new_layer = self.build_layer(layer)
            self.layers.append(new_layer)
        output_layer = nn.linear()
        self.layers.append()

    def initialize_graph(self):
        """Initializes all variables."""

        with self.graph.as_default():
            if self.verbose:
                print("Initializing model {}.".format(self.name))
            self.sess.run(self.init)

    def assign_lr(self):
        """Resets the learning rate in dynamic schedules for subsequent trainings.
        In bandits settings, we do expand our dataset over time. Then, we need to
        re-train the network with the new data. The algorithms that do not keep
        the step constant, can reset it at the start of each *training* process.
        """

        decay_steps = 1
        if self.hparams['activate_decay']:
            current_gs = self.sess.run(self.global_step)
            with self.graph.as_default():
                self.lr = tf.train.inverse_time_decay(self.hparams['initial_lr'],
                                                      self.global_step - current_gs,
                                                      decay_steps,
                                                      self.hparams['lr_decay_rate'])

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        return tf.train.RMSPropOptimizer(self.lr)

    def create_summaries(self):
        """Defines summaries including mean loss, learning rate, and global step."""

        with self.graph.as_default():
            with tf.name_scope(self.name + "_summaries"):
                tf.summary.scalar("cost", self.cost)
                tf.summary.scalar("lr", self.lr)
                tf.summary.scalar("global_step", self.global_step)
                self.summary_op = tf.summary.merge_all()

    def train(self, data, num_steps):
        """Trains the network for num_steps, using the provided data.
        Args:
          data: ContextualDataset object that provides the data.
          num_steps: Number of minibatches to train the network for.
        """

        if self.verbose:
            print("Training {} for {} steps...".format(self.name, num_steps))

        with self.graph.as_default():

            for step in range(num_steps):
                x, y, w = data.get_batch_with_weights(self.hparams['batch_size'])
                _, cost, summary, lr = self.sess.run(
                    [self.train_op, self.cost, self.summary_op, self.lr],
                    feed_dict={self.x: x, self.y: y, self.weights: w})

                if step % self.hparams['freq_summary'] == 0:
                    if self.hparams['show_training']:
                        print("{} | step: {}, lr: {}, loss: {}".format(
                            self.name, step, lr, cost))
                    self.summary_writer.add_summary(summary, step)

            self.times_trained += 1
