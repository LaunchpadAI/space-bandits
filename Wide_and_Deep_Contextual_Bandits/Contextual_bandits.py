import torch
import numpy as np

class ContextualBandit():
    
    def __init__(self, device, model, optimizer, loss_func, algorithm):
        self.device = device
        self.model = model.to(device)
        self.context_dim = model.context_dim
        self.z_dim = model.z_dim
        self.action_dim = model.action_dim
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.algorithm = algorithm
        self.reset()
        
    def reset(self):
        self.dataset = []
    
    def observe_data(self, context_source):
        x_t = np.array(context_source)
        assert x_t.shape == (self.action_dim, self.context_dim)
        return x_t 
    
    def get_reward(self, reward_source, a_t):
        r_t = float(reward_source[a_t])
        return r_t

    def run(self, context_source, reward_source):
        x_t = self.observe_data(context_source) # x_t np.array size=(action_dim, context_dim)
        x_t_tensor = torch.tensor(x_t).float().to(self.device)
        z_t = self.model.get_z(x_t_tensor).detach().cpu().numpy() # z_t np.array size=(action_dim, context_dim)
        a_t = self.algorithm.select_action(z_t) # a_t int range (0, action_dim)
        r_t = self.get_reward(reward_source, a_t) # r_t float either from an online simulation or from a reward vertor(size=action_dim)  
        data = (x_t[a_t], z_t[a_t], a_t, r_t)
        self.dataset.append(data)
        self.algorithm.update(data) # update selection strategy (parameters of algorithm)
        
        return a_t
    
    def train(self, start_index=0, batch_size=16, num_epoch=100):
        # prepare dataset and dataloader for training
        train_dataset = BanditDataset(self.dataset[start_index:])
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # train num_epoch epoches
        for i in range(num_epoch):
            for data_batch in train_dataloader:
                contexts, _, actions, rewards = data_batch
                contexts = contexts.float().to(self.device)
                actions = actions.long().to(self.device)
                rewards = rewards.float().to(self.device)
                outputs = self.model(contexts)
                pred_rewards = outputs[range(outputs.shape[0]),actions]
                loss = self.loss_func(pred_rewards, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()                
        # update algorithm's parameters after training
        self.algorithm.reset()
        for data in self.dataset:
            x_t_a_t, _, a_t, r_t = data
            x_t_a_t_tensor = torch.tensor(x_t_a_t).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_t_a_t = self.model.get_z(x_t_a_t_tensor).detach().cpu().numpy().reshape(-1)
            self.algorithm.update((x_t_a_t, z_t_a_t, a_t, r_t))
    
class BanditDataset(torch.utils.data.Dataset):
    
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset
        
    def __getitem__(self, index):
        context, z, action, reward = self.dataset[index]
        return np.array(context), np.array(z), action, reward
    
    def __len__(self):
        return len(self.dataset)
    