"""Contextual algorithm that keeps a full linear posterior for each arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from .bandit_algorithm import BanditAlgorithm
from .contextual_dataset import ContextualDataset

import pickle


class LinearFullPosteriorSampling(BanditAlgorithm):
    """Thompson Sampling with independent linear models and unknown noise var."""

    def __init__(self, name, hparams):
        """Initialize posterior distributions and hyperparameters.
        Assume a linear model for each action i: reward = context^T beta_i + noise
        Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise
        level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,
        and precision matrices are initialized, and the ContextualDataset created.
        Args:
          name: Name of the algorithm.
          hparams: Hyper-parameters of the algorithm.
        """

        self.name = name
        self.hparams = hparams

        # Gaussian prior for each beta_i
        self._lambda_prior = self.hparams.lambda_prior

        self.mu = [
            np.zeros(self.hparams.context_dim + 1)
            for _ in range(self.hparams.num_actions)
        ]

        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.hparams.context_dim + 1)
                    for _ in range(self.hparams.num_actions)]

        self.precision = [
            self.lambda_prior * np.eye(self.hparams.context_dim + 1)
            for _ in range(self.hparams.num_actions)
        ]

        # Inverse Gamma prior for each sigma2_i
        self._a0 = self.hparams.a0
        self._b0 = self.hparams.b0

        self.a = [self._a0 for _ in range(self.hparams.num_actions)]
        self.b = [self._b0 for _ in range(self.hparams.num_actions)]

        self.t = 0
        self.data_h = ContextualDataset(hparams.context_dim,
                                        hparams.num_actions,
                                        intercept=True)
        
    def expected_values(self, context):
        """
        Computes expected values from context. Does not consider uncertainty.
        Args:
          context: Context for which the action need to be chosen.
        Returns:
          expected reward vector.
        """
        # Compute sampled expected values, intercept is last component of beta
        vals = [
            np.dot(self.mu[i][:-1], context.T) + self.mu[i][-1]
            for i in range(self.hparams.num_actions)
        ]
        return np.array(vals)
        
    def _sample(self, context):
        """
        Samples beta's from posterior, and samples from expected values.
        Args:
          context: Context for which the action need to be chosen.
        Returns:
          action: sampled reward vector."""
        # Sample sigma2, and beta conditional on sigma2
        context = context.reshape(-1, self.hparams.context_dim)
        n_rows = len(context)
        a_projected = np.repeat(np.array(self.a)[np.newaxis, :], n_rows, axis=0)
        sigma2_s = self.b * invgamma.rvs(a_projected)
        if n_rows == 1:
            sigma2_s = sigma2_s.reshape(1, -1)
        beta_s = []
        try:
            for i in range(self.hparams.num_actions):
                mus = np.repeat(self.mu[i][np.newaxis, :], n_rows, axis=0)
                s2s = sigma2_s[:, i]
                rep = np.repeat(s2s[:, np.newaxis], self.hparams.context_dim+1, axis=1)
                rep = np.repeat(rep[:, :, np.newaxis], self.hparams.context_dim+1, axis=2)
                covs = np.repeat(self.cov[i][np.newaxis, :, :], n_rows, axis=0)
                covs = rep * covs
                multivariates = [np.random.multivariate_normal(mus[j], covs[j]) for j in range(n_rows)]
                beta_s.append(multivariates)
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            # Todo: Fix This
            print('Exception when sampling from {}.'.format(self.name))
            print('Details: {} | {}.'.format(e.message, e.args))
            d = self.hparams.context_dim + 1
            for i in range(self.hparams.num_actions):
                multivariates = [np.random.multivariate_normal(np.zeros((d)), np.eye(d)) for j in range(n_rows)]
                beta_s.append(multivariates)
        beta_s = np.array(beta_s)


        # Compute sampled expected values, intercept is last component of beta
        vals = [
            (beta_s[i, :, :-1] * context).sum(axis=-1) + beta_s[i, :, -1]
            for i in range(self.hparams.num_actions)
        ]
        return np.array(vals)

    def action(self, context):
        """Samples beta's from posterior, and chooses best action accordingly.
        Args:
          context: Context for which the action need to be chosen.
        Returns:
          action: Selected action for the context.
        """

        # Round robin until each action has been selected "initial_pulls" times
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            return self.t % self.hparams.num_actions
        else:
            vals = self._sample(context)
            return np.argmax(vals)

    def update(self, context, action, reward):
        """Updates action posterior using the linear Bayesian regression formula.
        Args:
          context: Last observed context.
          action: Last observed action.
          reward: Last observed reward.
        """
        self.t += 1
        self.data_h.add(context, action, reward)
        
        self._update_action(action)

    def _update_action(self, action):
        """Updates posterior for given action"""
        # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
        x, y = self.data_h.get_data(action)

        # The algorithm could be improved with sequential update formulas (cheaper)
        s = np.dot(x.T, x)

        # Some terms are removed as we assume prior mu_0 = 0.
        precision_a = s + self.lambda_prior * np.eye(self.hparams.context_dim + 1)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(x.T, y))

        # Inverse Gamma posterior update
        a_post = self.a0 + x.shape[0] / 2.0
        b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
        b_post = self.b0 + b_upd

        # Store new posterior distributions
        self.mu[action] = mu_a
        self.cov[action] = cov_a
        self.precision[action] = precision_a
        self.a[action] = a_post
        self.b[action] = b_post
        
    def fit(self, contexts, actions, rewards, num_updates=100):
        """Inputs bulk data for training.
        Args:
          contexts: Set of observed contexts.
          actions: Corresponding list of actions.
          rewards: Corresponding list of rewards.
        """
        data_length = len(rewards)
        self.data_h._ingest_data(contexts, actions, rewards)
        self.t += data_length
        #update posterior on ingested data
        for n in range(num_updates):
            for action in range(self.data_h.num_actions):
                self._update_action(action)
                
    def predict(self, contexts, thompson=True):
        """Takes a list or array-like of contexts and batch predicts on them"""
        if thompson:
            reward_matrix = self._sample(contexts)
        else:
            reward_matrix = self.expected_values(contexts)
        return np.argmax(reward_matrix, axis=0)

    def classic_predict(self, contexts, thompson=True):
        """Takes a list or array-like of contexts and batch predicts on them"""
        if thompson:
            rewards = np.array([np.argmax(self._sample(c.reshape(1, -1))) for c in contexts])
        else:
            rewards = np.array([np.argmax(self.expected_values(c.reshape(1, -1))) for c in contexts])
        return rewards
            
    def save(self, path):
        """saves model to path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def a0(self):
        return self._a0

    @property
    def b0(self):
        return self._b0

    @property
    def lambda_prior(self):
        return self._lambda_prior