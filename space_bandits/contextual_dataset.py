"""Define a data buffer for contextual bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ContextualDataset(object):
    """The buffer is able to append new data, and sample random minibatches."""

    def __init__(self, context_dim, num_actions, memory_size=-1, intercept=False):
        """Creates a ContextualDataset object.
        The data is stored in attributes: contexts and rewards.
        The sequence of taken actions are stored in attribute actions.
        Args:
          context_dim: Dimension of the contexts.
          num_actions: Number of arms for the multi-armed bandit.
          memory_size: Specify the number of examples to store in memory.
            if memory_size = -1, all data will be stored.
          intercept: If True, it adds a constant (1.0) dimension to each context X,
            at the end.
        """

        self._context_dim = context_dim
        self._num_actions = num_actions
        self.contexts = None
        self.scaled_contexts = None
        self.rewards = None
        self.actions = []
        self.memory_size = memory_size
        self.intercept = intercept
        self.scaling_data = []

    def add(self, context, action, reward):
        """Adds a new triplet (context, action, reward) to the dataset.
        The reward for the actions that weren't played is assumed to be zero.
        Args:
          context: A d-dimensional vector with the context.
          action: Integer between 0 and k-1 representing the chosen arm.
          reward: Real number representing the reward for the (context, action).
        """
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context)
        if len(context.shape) > 1:
            context = context.reshape(-1)
        if self.intercept:
            c = context[:]
            c = torch.cat((c, torch.tensor([1.0]).double()))
            c = c.reshape((1, self.context_dim + 1))
        else:
            if type(context) == type(torch.tensor(0)):
                c = context[:].reshape((1, self.context_dim))
            else:
                c = torch.tensor(context[:]).reshape((1, self.context_dim))

        if self.contexts is None:
            self.contexts = c
        else:
            self.contexts = torch.cat((self.contexts, c))

        r = torch.zeros((1, self.num_actions))
        r[0, action] = float(reward)
        if self.rewards is None:
            self.rewards = r
        else:
            self.rewards = torch.cat((self.rewards, r))

        self.actions.append(action)

        #Drop oldest example if memory constraint
        if self.memory_size != -1:
            if self.contexts.shape[0] > self.memory_size:
                self.contexts = self.contexts[1:, :]
                self.rewards = self.rewards[1:, :]
                self.actions = self.actions[1:]
            #Assert lengths match
            assert len(self.actions) == len(self.rewards)
            assert len(self.actions) == len(self.contexts)

    def _replace_data(self, contexts=None, actions=None, rewards=None):
        if contexts is not None:
            self.contexts = contexts
        if actions is not None:
            self.actions = actions
        if rewards is not None:
            self.rewards = rewards

    def _ingest_data(self, contexts, actions, rewards):
        """Ingests bulk data."""
        if isinstance(rewards, pd.DataFrame) or isinstance(rewards, pd.Series):
            rewards = rewards.values
        if isinstance(actions, pd.DataFrame) or isinstance(actions, pd.Series):
            actions = actions.values
        if isinstance(contexts, pd.DataFrame) or isinstance(contexts, pd.Series):
            contexts = contexts.values
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        if not isinstance(contexts, torch.Tensor):
            contexts = torch.tensor(contexts)
        data_length = len(rewards)
        if self.memory_size != -1:
            if data_length + len(self.rewards) > self.memory_size:
                print('Cannot add more examples: ')
                raise Exception('Too many examples for specified memory_size.')
        try:
            contexts = contexts.reshape(-1, self.context_dim)
        except:
            print('Got bad data contexts shape: ', contexts.shape)
            raise Exception('Expected: ({}, {})'.format(data_length, self.context_dim))
        if self.intercept:
            #add intercepts
            contexts = torch.cat([contexts, torch.ones((data_length, 1)).double()], dim=1)
        try:
            assert len(contexts) == data_length
            assert len(actions) == data_length
        except AssertionError:
            raise AssertionError('Data lengths inconsistent.')
        if self.contexts is None:
            self.contexts = contexts
        else:
            self.contexts = torch.cat((self.contexts, contexts))

        rewards_array = coo_matrix((np.array(rewards), (np.arange(data_length), np.array(actions)))).toarray()
        rewards_array = torch.tensor(rewards_array)
        if self.rewards is None:
            self.rewards = rewards_array
        else:
            self.rewards = torch.cat((self.rewards, rewards_array.float()))

        self.actions = self.actions + list(actions)

    def get_batch(self, batch_size=512):
        """Returns a random minibatch of (contexts, rewards) with batch_size."""
        n, _ = self.contexts.shape
        ind = np.random.choice(range(n), batch_size)
        return self.contexts[ind, :], self.rewards[ind, :], torch.tensor(self.actions)[ind]

    def get_data(self, action):
        """Returns all (context, reward) where the action was played."""
        n, _ = self.contexts.shape
        ind = np.argwhere(np.array(self.actions) == action).reshape(-1)
        return self.contexts[ind, :], self.rewards[ind, action]

    def get_data_with_weights(self):
        """Returns all observations with one-hot weights for actions."""
        weights = torch.zeros((self.contexts.shape[0], self.num_actions))
        a_ind = np.array([(i, val) for i, val in enumerate(self.actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        return self.contexts, self.rewards, weights

    def get_batch_with_weights(self, batch_size, scaled=False):
        """Returns a random mini-batch with one-hot weights for actions."""

        n, _ = self.contexts.shape
        if self.memory_size == -1:
            # use all the data
            ind = np.random.choice(range(n), batch_size)
        else:
            # use only buffer (last buffer_s obs)
            ind = np.random.choice(range(max(0, n - self.memory_size), n), batch_size)

        weights = torch.zeros((batch_size, self.num_actions))
        sampled_actions = torch.tensor(self.actions)[ind]
        a_ind = torch.tensor([(i, val) for i, val in enumerate(sampled_actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        if scaled:
            ctx = self.scaled_contexts[ind, :]
        else:
            ctx = self.contexts[ind, :]
        return ctx, self.rewards[ind, :], weights

    def num_points(self, f=None):
        """Returns number of points in the buffer (after applying function f)."""
        if f is not None:
            return f(self.contexts.shape[0])
        return self.contexts.shape[0]

    def scale_contexts(self, contexts=None):
        """
        Performs mean/std scaling operation on contexts.
        if contexts is provided as argument, returns scaled version
            (scaled by statistics of data in dataset.)
        """
        means = self.contexts.mean(dim=0)
        stds = self.contexts.std(dim=0)
        stds[stds==0] = 1
        self.scaled_contexts = self.contexts.clone()
        for col in range(self._context_dim):
            self.scaled_contexts[:, col] -= means[col]
            self.scaled_contexts[:, col] /= stds[col]
        if contexts is not None:
            if not isinstance(contexts, torch.Tensor):
                contexts = torch.tensor(contexts)
            result = contexts
            for col in range(self._context_dim):
                result[:, col] -= means[col]
                result[:, col] /= stds[col]
            return result

    def get_contexts(self, scaled=False):
        if scaled:
            return self.scaled_contexts
        else:
            return self.contexts

    def __len__(self):
        return len(self.actions)

    @property
    def context_dim(self):
        return self._context_dim

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def contexts(self):
        return self._contexts

    @contexts.setter
    def contexts(self, value):
        self._contexts = value

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        self._rewards = value
