# -*- coding: utf-8 -*-
"""Wide_Deep_Bandits.ipynb"""

## 02/20/2021 - Added GPU compatibility
## 02/27/2021 - Implemented initial_lr and lr_weight_decay
## 02/27/2021 - Implemented separate learning rate and learning rate decay for wide and deep networks
## 02/27/2021 - Added fit and predict functions to handle batch inputs and batch predicts

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy.sparse import coo_matrix

if torch.cuda.is_available():
  device = "cuda:0" 
else:  
  device = "cpu" 

"""# Define Models

Wide Model:
*   Input - User ID
*   Output - Expected reward for each action

Deep Model:
*   Input - Context
*   Output - Expected reward for each action

Wide and Deep Model: 

*   Combines output from the Wide model and the Deep model


"""

class Wide_Model(nn.Module):
    def __init__(self, embed_size=100, n_action=2, embed_dim=64):
        ## Learns expected reward for each action given User ID
        ## Uses embeddings to 'memorize' individual users
        ## embed_size - size of the dictionary of embeddings
        ## embed_dim -  size of each embedding vector
        ## n_action - number of possible actions

        super(Wide_Model, self).__init__()
        self.embed_size = embed_size
        self.n_action = n_action
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(self.embed_size, self.embed_dim)
        self.lr = nn.Linear(self.embed_dim, self.n_action)
    
    def forward(self, user_idx, get_rep = False):
        ## Input: user ID
        x = self.embedding(user_idx)
        rep = x
        x = self.lr(x)

        if get_rep == True:
          return rep.squeeze()
        else:
          return x.squeeze(axis=0)


class Deep_Model(nn.Module):
    def __init__(self, context_size=5, layer_sizes=[50,100], n_action=2):
        ## Learns expected reward for each action given context
        ## layer_sizes (list of integers): defines neural network architecture: n_layers = len(layer_sizes), 
        ## value is per-layer width. (default [50,100])
        super(Deep_Model, self).__init__()
        self.context_size = context_size
        self.layer_sizes = layer_sizes
        self.n_action = n_action

        self.layers = []
        self.build_model()
        self.activation = nn.ReLU()
    
    def build_layer(self, inp_dim, out_dim):
        """Builds a layer in deep model """
        layer = nn.modules.linear.Linear(inp_dim,out_dim)
        nn.init.uniform_(layer.weight)
        name = f'layer {len(self.layers)}'
        self.add_module(name, layer)
        return layer
    
    def build_model(self):
        """
        Defines the actual NN model with fully connected layers.
        """
        for i, layer in enumerate(self.layer_sizes):
            if i==0:
                inp_dim = self.context_size
            else:
                inp_dim = self.layer_sizes[i-1]
            out_dim = self.layer_sizes[i]
            new_layer = self.build_layer(inp_dim, out_dim)
            self.layers.append(new_layer)
        output_layer = self.build_layer(out_dim, self.n_action)
        self.layers.append(output_layer)

    def forward(self, contexts, get_rep = False):
        """forward pass of the neural network"""
        ## Input: context
        x = contexts
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                rep = x
        if get_rep == True:
          return rep.squeeze()
        else:
          return x.squeeze(axis=0)
  

class Wide_and_Deep_Model(nn.Module):
    def __init__(self, context_size=5, deep_layer_sizes=[50,100], n_action=2, embed_size=100, wide_embed_dim=64):
        super(Wide_and_Deep_Model, self).__init__()
        self.n_action = n_action
        self.context_size = context_size
        self.deep_layer_sizes = deep_layer_sizes
        self.embed_size = embed_size
        self.wide_embed_dim = wide_embed_dim

        self.lr_cred = nn.Linear(self.n_action*2, self.n_action)
        self.lr_crep = nn.Linear(self.wide_embed_dim + self.deep_layer_sizes[-1], self.n_action)

        self.wide_model = Wide_Model(embed_size=self.embed_size, n_action=self.n_action, embed_dim=self.wide_embed_dim).to(device)
        self.deep_model = Deep_Model(context_size=self.context_size, layer_sizes=self.deep_layer_sizes, n_action=self.n_action).to(device)
    
    def forward(self, wide_input, deep_input, combine_method='add_rewards'):
        
        x_wide = self.wide_model(wide_input)
        x_deep = self.deep_model(deep_input)

        possible_combine_methods = ['add_rewards','concat_reward','concat_reward_llr','concat_representation_llr']
        if combine_method not in possible_combine_methods:
          raise NameError('combine_method must be "add_rewards","concat_reward","concat_reward_llr", or "concat_representation_llr"')

        if combine_method == 'add_rewards':
          ## Add the outputs from wide and deep model
          x = x_wide + x_deep
        
        elif combine_method == 'concat_reward':
          ## Concatenate outputs from wide and deep model
          if len(x_wide.size()) == 1:
            x = torch.cat((x_wide,x_deep))
          elif len(x_wide.size()) > 1:
            x = torch.cat((x_wide,x_deep), dim=1)
        
        elif combine_method == 'concat_reward_llr':
          ## Concatenate outputs from wide and deep model
          if len(x_wide.size()) == 1:
            x = torch.cat((x_wide,x_deep))
          elif len(x_wide.size()) > 1:
            x = torch.cat((x_wide,x_deep), dim=1)
          ## One final linear layer to predict rewards
          x = self.lr_cred(x)

        elif combine_method == 'concat_representation_llr':
          ## Get last-layer representations from wide and deep model
          wide_rep = self.wide_model(wide_input, get_rep = True)
          deep_rep = self.deep_model(deep_input, get_rep = True)
          ## Concatenate representations from wide and deep model
          if len(wide_rep.size()) == 1:
            combine_rep = torch.cat([wide_rep, deep_rep])
          elif len(wide_rep.size()) > 1:
            combine_rep = torch.cat([wide_rep, deep_rep], dim=1)
          ## One final linear layer to predict rewards
          x = self.lr_crep(combine_rep)

        return x.squeeze(-1)

"""# Wide and deep bandit model
Three modes: wide, deep, or wide_deep (Use the model_type keyword)
"""

class Wide_Deep_Bandits():
    def __init__(
        self,
        num_actions,
        num_features,
        wide_embed_size=100, ## Size of embedding dictionary for the wide model
        wide_embed_dim=64, ## Dimension of embedding for the wide model
        wd_combine_method = 'concat_representation_llr', ## Method for combining the wide and deep models in the wide+deep model
        update_freq_nn = 100, ## Frequency to update the model, default updates model for every data point
        do_scaling = True, ## Whether to scale the contexts
        num_epochs = 1, ## Number of steps to Train for each update
        max_grad_norm=5.0, ## maximum gradient value for gradient clipping (float)
        initial_pulls=100, ## number of random pulls before greedy behavior (int)
        initial_lr_wide=0.01,## initial learning rate for wide network training (float, default same as torch.optim.RMSProp default)
        initial_lr_deep=0.01,## initial learning rate for deep network training (float, default same as torch.optim.RMSProp default)
        lr_decay_rate_wide=0.0,## learning rate decay for wide network updates (float)
        lr_decay_rate_deep=0.0,## learning rate decay for deep network updates (float)
        reset_lr=True, ## whether to reset learning rate when retraining network (bool)
        batch_size = 512, ## size of mini-batch to train at each step (int)
        model_type = 'wide_deep', ## model_type = 'wide', 'deep', or 'wide_deep'
        name='test_bandits'):
      
        ## Raise error if model_type is not one of the available models
        possible_models = ['deep','wide','wide_deep']
        if model_type not in possible_models:
          raise NameError('model_type must be "deep", "wide", or "wide_deep"')
        
        ## Raise error if wd_combine_method is not one of the available methods
        possible_combine_methods = ['add_rewards','concat_reward','concat_reward_llr','concat_representation_llr']
        if wd_combine_method not in possible_combine_methods:
          raise NameError('wd_combine_method must be "add_rewards","concat_reward","concat_reward_llr", or "concat_representation_llr"')

        self.name = name
        self.model_type = model_type
        self.wide_embed_dim = wide_embed_dim
        self.wide_embed_size = wide_embed_size
        self.wd_combine_method = wd_combine_method
        self.do_scaling = do_scaling
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions
        self.context_dim = num_features
        self.initial_lr_wide = initial_lr_wide
        self.initial_lr_deep = initial_lr_deep
        self.lr_wide = initial_lr_wide
        self.lr_deep = initial_lr_deep
        self.lr_decay_rate_wide = lr_decay_rate_wide
        self.lr_decay_rate_deep = lr_decay_rate_deep
        self.reset_lr = reset_lr
        self.batch_size = batch_size

        ## Initialize model and optimizer depeding on model_type
        if self.model_type == 'deep':
          self.deep_model = Deep_Model(context_size=self.context_dim,
                                       n_action=self.num_actions).to(device)
          #self.optim = torch.optim.RMSprop(self.deep_model.parameters())
          param_dict = [{'params': self.deep_model.parameters(), 'lr': self.initial_lr_deep}]
          self.param_dict = param_dict
          self.initial_param_dict = param_dict
          self.optim = self.select_optimizer()

        if self.model_type == 'wide':
          self.wide_model = Wide_Model(embed_size=self.wide_embed_size, 
                                      n_action=self.num_actions, 
                                      embed_dim=self.wide_embed_dim).to(device)
          #self.optim = torch.optim.RMSprop(self.wide_model.parameters())
          param_dict = [{'params': self.wide_model.parameters(), 'lr': self.initial_lr_wide}]
          self.param_dict = param_dict
          self.initial_param_dict = param_dict
          self.optim = self.select_optimizer()
        

        if self.model_type == 'wide_deep':
          self.wide_deep_model = Wide_and_Deep_Model(context_size=self.context_dim,
                                                    embed_size=self.wide_embed_size, 
                                                    n_action=self.num_actions, 
                                                    wide_embed_dim=self.wide_embed_dim) .to(device)
          #self.optim = torch.optim.RMSprop(self.wide_deep_model.parameters())
          param_dict = [{'params': self.wide_deep_model.wide_model.parameters(), 'lr': self.initial_lr_wide},
                        {'params': self.wide_deep_model.deep_model.parameters(), 'lr': self.initial_lr_deep},
                        {'params': self.wide_deep_model.lr_cred.parameters(), 'lr': 0.01},
                        {'params': self.wide_deep_model.lr_crep.parameters(), 'lr': 0.01}]
          self.param_dict = param_dict
          self.initial_param_dict = param_dict          
          self.optim = self.select_optimizer()
        
        self.loss = nn.modules.loss.MSELoss()

        self.t = 0
        self.update_freq_nn = update_freq_nn 
        self.num_epochs = num_epochs  
        self.data_h = ContextualDataset(self.context_dim,
                                        self.num_actions,
                                        intercept=False)
        
        ## Keep a dictionary of users that matches user's riid to indexes between 0 and num_users
        ## Initialize dicitonary with a "dummy user" that will be used for prediction when the user has never been seen
        self.user_dict = {0:0} 
        self.current_user_size = 1

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        return torch.optim.RMSprop(self.param_dict)
    
    def assign_lr(self, reset=False):
        """
        Assign learning rates using current self.param_dict.
        If reset = True, resets learning rates using self.initial_param_dict
        """
      
        if reset:
            for i in range(len(self.initial_param_dict)):
                self.optim.param_groups[i]['lr'] = self.initial_param_dict[i]['lr']
        else:
            for i in range(len(self.param_dict)):
                self.optim.param_groups[i]['lr'] = self.param_dict[i]['lr']

    def expected_values(self, user_id, context, multiple_rows=True):
        ## Get expected values through forward pass
        ## multiple_rows = True if getting expected values of more than 1 row at a time

        context = torch.tensor(context).float().to(device) 

        if multiple_rows==False: 
          user_idx = self.lookup_user_idx(user_id, multiple_users=False).to(device)
        elif multiple_rows==True:
          user_idx = self.lookup_user_idx(user_id, multiple_users=True).to(device)
        
        if self.model_type == 'deep':
          x = self.deep_model(context)
        if self.model_type == 'wide':
          x = self.wide_model(user_idx)
        if self.model_type == 'wide_deep':
          x = self.wide_deep_model(user_idx, context, combine_method=self.wd_combine_method)
          if self.wd_combine_method == 'concat_reward':
              ## Add the output from wide and deep model
              n_act=self.num_actions ## Number of actions for combining wide and deep outputs
              if multiple_rows==False:
                x = x[0:n_act]+x[n_act:n_act*2]
              elif multiple_rows==True:
                x = x[:,0:n_act]+x[:,n_act:n_act*2]
        return x

    def action(self, user_id, context):
        """Select an action based on expected values of reward"""
        if self.do_scaling:
            context = context.reshape(-1, self.context_dim)
            context = self.data_h.scale_contexts(contexts=context)[0]
        vals = self.expected_values(user_id, context, multiple_rows=False)  ## for wide and deep model
        return np.argmax(vals.cpu().detach().numpy())

    def predict(self, user_ids, contexts):
        """Takes a list of users and list or array-like of contexts and batch predicts on them"""
        contexts = contexts.reshape(-1, self.context_dim)
        if self.do_scaling:
            contexts = self.data_h.scale_contexts(contexts=contexts)
        reward_matrix = self.expected_values(user_ids, contexts, multiple_rows=True)
        return np.argmax(reward_matrix.cpu().detach().numpy(), axis=1)
        
    def update(self, user_id, context, action, reward):
        """
        Args:
          user_id: Last observed user
          context: Last observed context.
          action: Last observed action.
          reward: Last observed reward.
        """
        
        self.t += 1
        self.data_h.add(user_id, context, action, reward)
        self.update_user_dict(user_id)

        if self.t % self.update_freq_nn == 0:
          self.train(self.data_h, self.num_epochs)
        
    def fit(self, user_id, contexts, actions, rewards):
        """Inputs bulk data for training.
        Args:
          user_id: List of users
          contexts: Observed contexts.
          actions: Corresponding list of actions.
          rewards: Corresponding list of rewards.
        """
        data_length = len(rewards)
        self.data_h._ingest_data(user_id,contexts, actions, rewards)

        for i in range(data_length):
          self.update_user_dict(user_id[i])
        #update count
        self.t += data_length
        #update posterior on ingested data
        self._retrain_nn()
    
    
    def train(self, data, num_steps):
        """Trains the network for num_steps, using the provided data.
        Args:
          data: ContextualDataset object that provides the data.
          num_steps: Number of minibatches to train the network for.
        Takes longer to get batch data and train model as the data size increase
        """
        #print("Training at time {} for {} steps...".format(self.t, num_steps))

        batch_size = self.batch_size
        
        if self.do_scaling == True:
            data.scale_contexts() ## have to scale the data first if scaled=True in data.get_batch_with_weights()

        for step in range(num_steps):
            #u, x, y, w = data.get_batch_with_weights(batch_size, scaled=self.do_scaling)
            u, x, y, w = data.get_batch_with_weights_recent(batch_size, n_recent=self.update_freq_nn, scaled=self.do_scaling)
            u = self.lookup_user_idx(u)

            u = u.to(device)
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            ## Training at time step 1 will cause problem if scaled=True, 
            ## because standard deviation=0, and scaled_context will equal nan
            if self.t != 1:
              self.do_step(u, x, y, w, step)
              

    def do_step(self, u, x, y, w, step):
        ## Set new learning rates
        decay_rate_wide = self.lr_decay_rate_wide
        base_lr_wide = self.initial_lr_wide
        decay_rate_deep = self.lr_decay_rate_deep
        base_lr_deep = self.initial_lr_deep

        self.lr_wide = base_lr_wide * (1 / (1 + (decay_rate_wide * step)))
        self.lr_deep = base_lr_deep * (1 / (1 + (decay_rate_deep * step)))

        ## Get y_hat from network forward pass and update parameter dictionary with new learning rates
        if self.model_type == 'deep':
          y_hat = self.deep_model.forward(x.float())
          self.param_dict[0]['lr'] = self.lr_deep
        if self.model_type == 'wide':
          y_hat = self.wide_model.forward(u)
          self.param_dict[0]['lr'] = self.lr_wide
        if self.model_type == 'wide_deep':
          y_hat= self.wide_deep_model(u,x.float(),combine_method=self.wd_combine_method)
          self.param_dict[0]['lr'] = self.lr_wide
          self.param_dict[1]['lr'] = self.lr_deep
          if self.wd_combine_method == 'concat_reward':
            ## replicate y and w to compare with concatenated y_hat from wide_deep model if concatenating reward
            y = torch.cat((y,y), dim=1) 
            w = torch.cat((w,w), dim=1)
        
        self.assign_lr()
        
        #print("y = ", y.float()[0])
        #print("y_hat = ", y_hat[0])
        #print()

        y_hat *= w  ## No back propagation on action not taken
        ls = self.loss(y_hat, y.float())
        ls.backward()

        clip = self.max_grad_norm

        if self.model_type == 'deep':
          torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), clip)
        if self.model_type == 'wide':
          torch.nn.utils.clip_grad_norm_(self.wide_model.parameters(), clip)
        if self.model_type == 'wide_deep':
          torch.nn.utils.clip_grad_norm_(self.wide_deep_model.parameters(), clip)

        self.optim.step()
        self.optim.zero_grad()

    
    def _retrain_nn(self):
        """Retrain the network on original data (data_h)"""
        if self.reset_lr:
            self.assign_lr(reset=True)

        ## Uncomment following lines to automatically set number of steps according to data length and batch size.
        #steps = round(self.num_epochs * (len(self.data_h)/self.batch_size))
        #print(f"training for {steps} steps.")
        self.train(self.data_h, self.num_epochs)

    def lookup_user_idx(self, user_id, multiple_users=True):
      """Returns a list of user indexes for input to the wide network"""
      user_index = []

      if torch.is_tensor(user_id):
        user_id = user_id.tolist()
      
      if multiple_users==True:
        for u in user_id:
          if u in self.user_dict.keys():
            user_idx = self.user_dict[u]
          else:
            #print("User not found, returning dummy user")
            user_idx = 0
          user_index.append(user_idx)
      elif multiple_users==False:
        u = user_id
        if u in self.user_dict.keys():
            user_idx = self.user_dict[u]
        else:
          user_idx = 0
        user_index.append(user_idx)
      return torch.tensor(user_index)
    
    def update_user_dict(self, user_id):
      """Create/update a lookup dictionary that matches user IDs to a user indexes between 0 and num_users"""
      if torch.is_tensor(user_id):
        user_id = user_id.tolist()
      if user_id not in self.user_dict:
        self.user_dict.update({user_id:self.current_user_size})
        self.current_user_size += 1
              
    def save(self, path):
        """saves model to path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)


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
        self.user_ids = []
        self.actions = []
        self.memory_size = memory_size
        self.intercept = intercept
        self.scaling_data = []
    
    def add(self, user_id, context, action, reward):
        """Adds a new triplet (context, action, reward) to the dataset.
            The reward for the actions that weren't played is assumed to be zero.
            Args:
            user_id: User ID, usually an integer
            context: A d-dimensional vector with the context.
            action: Integer between 0 and k-1 representing the chosen arm.
            reward: Real number representing the reward for the (context, action).
            """
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context.astype(float))
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
        self.user_ids.append(user_id)
                                
        #Drop oldest example if memory constraint
        if self.memory_size != -1:
            if self.contexts.shape[0] > self.memory_size:
                self.contexts = self.contexts[1:, :]
                self.rewards = self.rewards[1:, :]
                self.actions = self.actions[1:]
                self.user_ids = self.user_ids[1:]
            #Assert lengths match
            assert len(self.actions) == len(self.rewards)
            assert len(self.actions) == len(self.contexts)
            assert len(self.actions) == len(self.user_ids)

    def _replace_data(self, user_ids=None, contexts=None, actions=None, rewards=None):
        if contexts is not None:
            self.contexts = contexts
        if actions is not None:
            self.actions = actions
        if rewards is not None:
            self.rewards = rewards
        if user_ids is not None:
            self.user_ids = user_ids

    def _ingest_data(self, user_ids, contexts, actions, rewards):
        """Ingests bulk data."""
        if isinstance(rewards, pd.DataFrame) or isinstance(rewards, pd.Series):
            rewards = rewards.values
        if isinstance(actions, pd.DataFrame) or isinstance(actions, pd.Series):
            actions = actions.values
        if isinstance(user_ids, pd.DataFrame) or isinstance(user_ids, pd.Series):
            user_ids = user_ids.values
        if isinstance(contexts, pd.DataFrame) or isinstance(contexts, pd.Series):
            contexts = contexts.values
        
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids)
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
        rewards_array = torch.tensor(rewards_array).float()
            
        if self.rewards is None:
            self.rewards = rewards_array
        else:
            self.rewards = torch.cat((self.rewards, rewards_array.float()))

        self.actions = self.actions + list(actions)
        self.user_ids = self.user_ids + list(user_ids)
    
    def get_batch(self, batch_size=512):
        """Returns a random minibatch of (contexts, rewards) with batch_size."""
        n, _ = self.contexts.shape
        ind = np.random.choice(range(n), batch_size)
        return torch.tensor(self.user_ids)[ind], self.contexts[ind, :], self.rewards[ind, :], torch.tensor(self.actions)[ind]
    
    def get_data(self, action):
        """Returns all (user_id, context, reward) where the action was played."""
        n, _ = self.contexts.shape
        ind = np.argwhere(np.array(self.actions) == action).reshape(-1)
        return torch.tensor(self.user_ids)[ind], self.contexts[ind, :], self.rewards[ind, action]
    
    def get_user_freq(self, user_id):
        """Returns the number of times an user_id occurs in data"""
        user_id = torch.tensor(user_id).tolist()
        user_arr = np.array(self.user_ids)
        user_freq = len(np.where(user_arr == user_id)[0])
        return user_freq
    
    def get_data_with_weights(self):
        """Returns all observations with one-hot weights for actions."""
        weights = torch.zeros((self.contexts.shape[0], self.num_actions))
        a_ind = np.array([(i, val) for i, val in enumerate(self.actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        return torch.tensor(self.user_ids)[ind], self.contexts, self.rewards, weights
    
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
        return torch.tensor(self.user_ids)[ind], ctx, self.rewards[ind, :], weights
    
    
    def get_batch_with_weights_recent(self, batch_size, n_recent=0, scaled=False):
        """Returns a random mini-batch with one-hot weights for actions."""
        
        n, _ = self.contexts.shape
        if self.memory_size == -1:
            # use all the data
            ind = np.random.choice(range(n), batch_size)
        else:
            # use only buffer (last buffer_s obs)
            ind = np.random.choice(range(max(0, n - self.memory_size), n), batch_size)
        
        if n_recent > 0:
            ## returns the n_recent most recently added data points
            if n_recent > batch_size:
                n_recent = batch_size
            if n_recent > n:
                n_recent = n
            ind[:n_recent] = range(n-n_recent,n)
        
        weights = torch.zeros((batch_size, self.num_actions))
        sampled_actions = torch.tensor(self.actions)[ind]
        a_ind = torch.tensor([(i, val) for i, val in enumerate(sampled_actions)])
        weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
        if scaled:
            ctx = self.scaled_contexts[ind, :]
        else:
            ctx = self.contexts[ind, :]
        return torch.tensor(self.user_ids)[ind], ctx, self.rewards[ind, :], weights
    
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
                
    def get_user_ids(self):
        return torch.tensor(self.user_ids)

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
    
    @property
    def user_ids(self):
        return self._user_ids
    
    @user_ids.setter
    def user_ids(self, value):
        self._user_ids = value
