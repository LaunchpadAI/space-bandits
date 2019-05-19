"""Define a family of neural network architectures for bandits.
The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from .bayesian_nn import BayesianNN

def build_action_mask(actions, num_actions):
    """
    Takes a torch tensor with integer values.
    Returns a one-hot encoded version, where
        each column corresponds to a single action.
    """
    ohe = torch.zeros((len(actions), num_actions))
    actions = actions.reshape(-1, 1)
    return ohe.scatter_(1, actions, 1)

def build_target(rewards, actions, num_actions):
    """
    Takes a torch tensor with floating point values.
    Returns a one-hot encoded version, where
        each column corresponds to a single action.
        The value is the observed reward.
    """
    ohe = torch.zeros((len(actions), num_actions))
    actions = actions.reshape(-1, 1)
    return ohe.scatter_(1, actions, rewards)


class NeuralBanditModel(nn.Module):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name):
        """Saves hyper-params and builds a torch NN."""
        super(NeuralBanditModel, self).__init__()
        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = self.hparams["verbose"]
        self.times_trained = 0
        self.lr = self.hparams["initial_lr"]
        if self.hparams['activation'] == 'relu':
            self.activation = F.relu
        else:
            act = self.hparams['activation']
            msg = f'activation {act} not implimented'
            raise Exception(msg)
        self.layers = []
        self.build_model()
        self.optim = self.select_optimizer()
        self.loss = nn.modules.loss.MSELoss()

    def build_layer(self, inp_dim, out_dim):
        """Builds a layer with input x; dropout and layer norm if specified."""

        init_s = self.hparams.get('init_scale', 0.3)
        #these features not currently implemented
        layer_n = self.hparams.get("layer_norm", False)
        dropout = self.hparams.get("use_dropout", False)

        layer = nn.modules.linear.Linear(
            inp_dim,
            out_dim
        )
        nn.init.uniform_(layer.weight, a=-init_s, b=init_s)
        name = f'layer {len(self.layers)}'
        self.add_module(name, layer)
        return layer

    def forward(self, x):
        """forward pass of the neural network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1:
                x = self.activation(x)
        return x

    def build_model(self):
        """
        Defines the actual NN model with fully connected layers.
        """
        for i, layer in enumerate(self.hparams['layer_sizes']):
            if i==0:
                inp_dim = self.hparams['context_dim']
            else:
                inp_dim = self.hparams['layer_sizes'][i-1]
            out_dim = self.hparams['layer_sizes'][i]
            new_layer = self.build_layer(inp_dim, out_dim)
            self.layers.append(new_layer)
        output_layer = nn.modules.linear.Linear(out_dim, self.hparams['num_actions'])
        self.layers.append(output_layer)

    def assign_lr(self, lr=None):
        """
        Resets the learning rate to input argument value "lr".
        """
        if lr is None:
            lr = self.lr
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        lr = self.hparams['initial_lr']
        return torch.optim.RMSprop(self.parameters(), lr=lr)

    def scale_weights(self):
        init_s = self.hparams.get('init_scale', 0.3)
        for layer in self.layers:
            nn.init.uniform_(layer.weight, a=-init_s, b=init_s)

    def do_step(self, x, y, w, step):

        decay_rate = self.hparams.get('lr_decay_rate', 0.5)
        base_lr = self.hparams.get('initial_lr', 0.1)

        lr = base_lr * (1 / (1 + (decay_rate * step)))
        self.assign_lr(lr)

        y_hat = self.forward(x.float())
        y_hat *= w
        ls = self.loss(y_hat, y.float())
        ls.backward()
        clip = self.hparams['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.optim.step()
        self.optim.zero_grad()

    def train(self, data, num_steps):
        """Trains the network for num_steps, using the provided data.
        Args:
          data: ContextualDataset object that provides the data.
          num_steps: Number of minibatches to train the network for.
        """

        if self.verbose:
            print("Training {} for {} steps...".format(self.name, num_steps))

        batch_size = self.hparams.get('batch_size', 512)

        data.scale_contexts()
        for step in range(num_steps):
            x, y, w = data.get_batch_with_weights(batch_size, scaled=True)
            self.do_step(x, y, w, step)

    def get_representation(self, contexts):
        """
        Given input contexts, returns representation
        "z" vector.
        """
        x = contexts
        with torch.no_grad():
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = self.activation(x)
        return x
