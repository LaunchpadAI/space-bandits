import unittest
import time
import torch
import numpy as np
import pandas as pd
import pickle

import subprocess
import os
from space_bandits.contextual_dataset import ContextualDataset
from space_bandits.toy_problem import get_customer, get_rewards, get_cust_reward
from space_bandits.toy_problem import generate_dataframe
from space_bandits.linear import LinearBandits
from space_bandits.neural_linear import NeuralBandits
from space_bandits.neural_bandit_model import NeuralBanditModel
from space_bandits.bayesian_nn import BayesianNN

def create_neural_bandit_model():
    hparams = create_hparams()
    return NeuralBanditModel('RMS', hparams, 'testmodel')

def create_hparams():
    hparams = {
        'num_actions':3,
        'context_dim':2,
        'name':'testmodel',
        'init_scale':0.3,
        'activation':'relu',
        'verbose':True,
        'optimizer':'RMS',
        'layer_sizes':[50],
        'batch_size':512,
        'activate_decay':True,
        'initial_lr':0.1,
        'max_grad_norm':5.0,
        'show_training':False,
        'freq_summary':1000,
        'buffer_s':-1,
        'initial_pulls':100,
        'reset_lr':True,
        'lr_decay_rate':0.5,
        'training_freq':1,
        'training_freq_network':50,
        'training_epochs':100,
        'memory_size':-1,
        'a0':6,
        'b0':6,
        'lambda_prior':0.25
    }
    return hparams

def create_linear_model():
    model = LinearBandits(
        num_actions=3,
        num_features=2,
    )
    return model

def check_expected_values(model):
    customer = get_customer()
    fts = np.array(customer[1])
    values = model.expected_values(fts)
    return values

def check_sample(model):
    customer = get_customer()
    fts = np.array(customer[1])
    values = model._sample(fts)
    return values

def check_action(model):
    customer = get_customer()
    fts = np.array(customer[1])
    action = model.action(fts)
    return action

def update_model(model):
    fts, reward = get_cust_reward()
    action = np.random.randint(3)
    context = np.array(fts).reshape(-1)
    reward = float(reward[action])
    model.update(context, action, reward)
    return model

def fit_model(model):
    data = generate_dataframe(220)
    ctx = data[['age', 'ARPU']]
    actions = data['action']
    rewards = data['reward']
    model.fit(ctx, actions, rewards)
    return model

def predict_model(model):
    data = generate_dataframe(220)
    ctx = data[['age', 'ARPU']]
    predictions = model.predict(ctx)
    return predictions

def check_toy_problem():
    customer = get_customer()
    ctype, (age, ft) = customer
    assert isinstance(ctype, int)
    assert isinstance(age, int)
    assert isinstance(ft, float)
    reward = get_rewards(customer)
    assert reward.shape == (3,)
    fts, reward = get_cust_reward()
    df = generate_dataframe(10)
    assert isinstance(df, pd.DataFrame)
    return fts, reward

def create_contextual_dataset():
    dataset = ContextualDataset(
        context_dim=5,
        num_actions=5,
        memory_size=200,
        intercept=True
    )
    return dataset

def check_add_data(dataset):
    context = np.random.randn(5)
    action = np.random.randint(0,5)
    reward = np.random.randn()
    dataset.add(context, action, reward)
    return dataset

def check_replace_data(dataset):
    context = dataset.contexts
    action = dataset.actions
    reward = dataset.rewards
    dataset._replace_data(context, action, reward)
    return dataset

def check_ingest_data(dataset):
    context = np.random.randn(199, 5)
    action = np.random.randint(0,5, (199))
    reward = np.random.randn(199)
    dataset._ingest_data(context, action, reward)
    return dataset

def check_get_batch(dataset):
    batch_size = 64
    ctx, r = dataset.get_batch(batch_size)
    assert ctx.shape == (batch_size, 6)
    assert isinstance(ctx, torch.Tensor)
    assert r.shape == (batch_size, 5)
    assert isinstance(r, torch.Tensor)
    return ctx, r, dataset

def check_get_data(dataset):
    action = 0
    ctx, r = dataset.get_data(action)
    assert ctx.shape[1] == 6
    assert isinstance(ctx, torch.Tensor)
    assert r.shape[0] == ctx.shape[0]
    assert isinstance(r, torch.Tensor)
    return ctx, r, dataset

def check_get_data_with_weights(dataset):
    ctx, r, weights = dataset.get_data_with_weights()
    assert ctx.shape == (200, 6)
    assert isinstance(ctx, torch.Tensor)
    assert r.shape == (200,5)
    assert isinstance(r, torch.Tensor)
    assert weights.shape == (200, 5)
    assert isinstance(weights, torch.Tensor)
    return ctx, r, weights

def check_get_batch_with_weights(dataset):
    batch_size = 64
    ctx, r, weights = dataset.get_batch_with_weights(batch_size)
    assert ctx.shape == (batch_size, 6)
    assert isinstance(ctx, torch.Tensor)
    assert r.shape == (batch_size,5)
    assert isinstance(r, torch.Tensor)
    assert weights.shape == (batch_size, 5)
    assert isinstance(weights, torch.Tensor)
    return ctx, r, weights

def check_torch_gpu():
    return torch.cuda.is_available()

def check_torch_cpu():
    torch.tensor([10, 0])
    return True


class AppTest(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bayesian_nn(self):
        nn = BayesianNN()

#    def test_neural_bandit_model(self):
#        model = create_neural_bandit_model()

    def test_linear_model(self):
        model = create_linear_model()
        model = update_model(model)
        model = fit_model(model)
        values = check_expected_values(model)
        assert values.shape == (3,)
        predictions = predict_model(model)
        values = check_sample(model)
        assert values.shape == (3,1)
        action = check_action(model)
        assert isinstance(action, np.int64)
        model.save('test_file')
        assert os.path.isfile('test_file')
        os.remove('test_file')

    def test_toy_problem(self):
        fts, reward = check_toy_problem()

    def test_contextual_dataset(self):
        dataset = create_contextual_dataset()
        dataset = check_add_data(dataset)
        dataset = check_replace_data(dataset)
        dataset = check_ingest_data(dataset)
        dataset = check_add_data(dataset)
        assert dataset.num_points() == 200
        assert len(dataset) == 200
        ctx, r, dataset = check_get_batch(dataset)
        ctx, r, dataset = check_get_data(dataset)
        ctx, r, weights = check_get_data_with_weights(dataset)
        batch = check_get_batch_with_weights(dataset)
        assert isinstance(dataset.contexts, torch.Tensor)
        assert isinstance(dataset.rewards, torch.Tensor)
        assert isinstance(dataset.actions, list)

    def test_torch_cpu(self):
        assert check_torch_cpu()
        return

#    def test_torch_gpu(self):
#        assert check_torch_gpu()
#        return

if __name__ == '__main__':
    unittest.main()
