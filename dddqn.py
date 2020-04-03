import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import pickle
import sys

device = torch.device("cuda")


class DuelingDQNetwork(nn.Module):

    def __init__(self):
        super(DuelingDQNetwork, self).__init__()
        input_channels = 13
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        feature_size = 256
        self.value = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x)).view(state.size(0), 1, -1)
        value = self.value(x)
        advantage = self.advantage(x)
        advantage = advantage - advantage.mean(dim=2, keepdim=True)
        return value + advantage


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.03, eps_decay=0.996,
                            gamma=0.9999, memory_capacity=5000, batch_size=128, alpha=5e-4, tau=1e-3):
        self.local_Q = DuelingDQNetwork().to(device)
        self.target_Q = DuelingDQNetwork().to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.indexes = np.arange(self.batch_size)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 0
    
    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, state):
        if np.random.random() > self.eps_start:
            self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                action = torch.argmax(self.local_Q(obs)).item()
            self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def sample_batch(self):
        batch = self.replay_memory.sample(self.batch_size)

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        action_batch = torch.tensor(batch[1])
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device, dtype=torch.float)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_batch()
        
        max_actions = torch.argmax(self.local_Q(next_state_batch), dim=2).squeeze(1)
        evaluated = self.target_Q(next_state_batch)[self.indexes,0,max_actions]
        evaluated = reward_batch + self.gamma * evaluated * done_batch

        prediction = self.local_Q(state_batch)[self.indexes,0,action_batch]

        self.optimizer.zero_grad()
        loss = self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
        
        self.eps_start = max(self.eps_end, self.eps_decay * self.eps_start)
        

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            reward = self.memory[int(self.position)][2]
            if reward > 2:
                self.position = (self.position + 1) % self.capacity
                self.push(args)
                return

        self.memory[int(self.position)] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        return list(zip(*random.sample(self.memory, size)))


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent
