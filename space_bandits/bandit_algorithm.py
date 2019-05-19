"""Define the abstract class for contextual bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import mean_squared_error

def from_pandas(inp):
    try:
        inp = inp.values
    except:
        pass
    return inp

class BanditAlgorithm(object):
    """A bandit algorithm must be able to do two basic operations.

    1. Choose an action given a context.
    2. Update its internal model given a triple (context, played action, reward).
    """

    def action(self, context):
        pass

    def update(self, context, action, reward):
        pass

    def predict_proba(self, context):
        """

        Assumes rewards are associted with a binary outcome
        where r <= 0 is a zero (negative) outcome,
        r > 0 is a one (positive) outcome.

        With this assumption, this function returns the probability
        for a positive outcome for each available action.

        """
        #deal with pandas objects
        context = from_pandas(context)

        expected_values = self.expected_values(context)
        reward_history = self.data_h.rewards
        action_history = np.array(self.data_h.actions)
        for a in range(self.hparams.num_actions):
            action_args = np.argwhere(action_history==a)
            slc = reward_history[action_args,a][:,0]
            convert = np.where(slc > 0, 1, 0)
            pos = convert.sum()
            pos_weight = pos/len(slc)
            center = np.quantile(expected_values[a], (1-pos_weight))
            expected_values[a] = (expected_values[a] - center)/expected_values[a].std()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        probas = sigmoid(expected_values)
        return probas

    def get_sscore(self, context, action, reward):
        """

        Computes the contextual bandits score S
        on some provided validation data
        (context, action, reward triplet).

        Returns a singular score value S.
        0 < S < 1 indicates some convergence (higher is better)
        S < 0 indicates you're better off with a multi-arm bandit model.

        """
        #deal with pandas objects
        context = from_pandas(context)
        action = from_pandas(action)
        reward = from_pandas(reward)

        #recall training data
        past_actions = np.array(self.data_h.actions)
        past_rewards = np.array(self.data_h.rewards)

        #compute naive expected rewards
        E_b = []
        for a in range(self.hparams.num_actions):
            inds = np.argwhere(past_actions==a)
            slc = past_rewards[inds, a][:, 0]
            E = slc.mean()
            E_b.append(E)
        Err_b = []

        #compute benchmark error
        for a in range(self.hparams.num_actions):
            args = np.argwhere(action==a)[:, 0]
            slc = reward[args]
            y_pred = [E_b[a] for x in range(len(slc))]
            y_true = slc
            error = mean_squared_error(y_pred, y_true)
            Err_b.append(error)
        Err_b = np.array(Err_b)

        #compute validation error
        bal = []
        expected_values = self.expected_values(context)
        Err_m = []
        for a in range(self.hparams.num_actions):
            args = np.argwhere(action==a)[:, 0]
            #record representation of action a
            bal.append(len(args))
            y_pred = expected_values[a, args]
            y_true = reward[args]
            E = mean_squared_error(y_pred, y_true)
            Err_m.append(E)
        bal = np.array(bal) / sum(bal)
        rat_vec = 1 - Err_m/Err_b
        avg = np.average(rat_vec, weights=bal)

        return avg
