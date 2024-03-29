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
from space_bandits.neural_linear import NeuralBandits, load_model
from space_bandits.neural_bandit_model import NeuralBanditModel, build_action_mask
from space_bandits.neural_bandit_model import build_target

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

def create_toy_contexual_dataset():
    dataset = ContextualDataset(
        context_dim=2,
        num_actions=3,
        memory_size=-1,
        intercept=False
    )
    for i in range(2000):
        fts, reward = get_cust_reward()
        action = i % 3
        r = reward[action]
        dataset.add(fts, action, r)
    return dataset

def check_scale_contexts(dataset):
    dataset.scale_contexts()

def check_add_data(dataset):
    context = np.random.randn(1, 5)
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
    ctx, r, act = dataset.get_batch(batch_size)
    assert ctx.shape == (batch_size, 6)
    assert isinstance(ctx, torch.Tensor)
    assert r.shape == (batch_size, 5)
    assert isinstance(r, torch.Tensor)
    assert isinstance(act, torch.Tensor)
    assert act.shape == (batch_size,)
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

    def test_action_mask(self):
        dataset = self.test_contextual_dataset()
        _, _, actions = dataset.get_batch()
        ohe = build_action_mask(actions, num_actions=5)
        assert ohe.shape == (512, 5)
        for i in range(512):
            assert ohe[i, actions[i]] == 1

    def test_build_target(self):
        dataset = self.test_contextual_dataset()
        _, rewards, actions = dataset.get_batch()
        ohe = build_target(rewards, actions, num_actions=5)
        assert ohe.shape == (512, 5)

    def test_neural_linear_model(self):
        model = NeuralBandits(
            num_actions=3,
            num_features=2,
            training_freq_network=200,
            layer_sizes=[50]
        )
        fts, reward = get_cust_reward()
        for i in range(300):
            action = model.action(fts)
            r = reward[action]
            model.update(fts, action, r)
        df = generate_dataframe(500)
        X = df[['age', 'ARPU']].values
        A = df['action'].values
        R = df['reward'].values
        model.fit(X, A, R)
        model.save('test_file')
        model = load_model('test_file')
        X = df[['age', 'ARPU']].sample(2).values
        model.predict(X, parallelize=False)
        os.remove('test_file')

    def test_bayesian_nn(self):
        nn = BayesianNN()

    def test_neural_bandit_model(self):
        model = create_neural_bandit_model()
        assert len(model.layers) == 2
        dataset = create_toy_contexual_dataset()
        model.train(dataset, 10)
        ctx = dataset.get_contexts(scaled=True).float()
        z = model.get_representation(ctx)
        assert z.shape == (2000, 50)
        assert z.min() == 0.0

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
        check_scale_contexts(dataset)
        return dataset

    def test_update(self):
        model = LinearBandits(
            3,
            4
        )
        context = np.array([76, 120, 654326, 2])
        action = 1
        reward = 14
        model.update(context, action, reward)

        df = generate_dataframe(500)

        contexts = df[['age', 'ARPU']]
        actions = df['action']
        rewards = df['reward']

        new_model = NeuralBandits(3, 2, layer_sizes=[50, 12], verbose=False)
        #call .fit method; num_updates will repeat training n times
        new_model.fit(contexts, actions, rewards)

        new_context = np.array([26.0, 98.456463])
        action = np.random.randint(0, 2)
        reward = np.random.random() * 10
        print(action)
        new_model.update(new_context, action, reward)

    def test_torch_cpu(self):
        assert check_torch_cpu()
        return

    def test_issue_20(self):
        np.random.seed(0)

        model = LinearBandits(initial_pulls=2, num_actions=3, num_features=1)
        model.fit(
            contexts=pd.DataFrame(np.random.normal(loc=10, size=100)),
            actions=#np.array([2] + [0] * 99),
             #np.array([0] * 100),# doesn't work
             np.array([1] + [0] * 99), # doesn't work
            # np.array([2] + [0] * 99) works
            rewards=np.random.normal(loc=10, size=100),
        )

if __name__ == '__main__':
    unittest.main()
