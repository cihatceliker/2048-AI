import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import pickle
import sys

device = torch.device("cuda")


class Brain(nn.Module):

    def __init__(self):
        super(Brain, self).__init__()
        input_channels = 16
        self.conv1_2 = nn.Conv2d(input_channels, 64, kernel_size=(1,2))
        self.conv2_1 = nn.Conv2d(input_channels, 64, kernel_size=(2,1))
        self.sec_conv = nn.Conv2d(64, 128, 2)
        self.conv2_pool = nn.Conv2d(input_channels, 128, kernel_size=2, padding=1)
        self.block_1 = nn.Conv2d(input_channels, 128, kernel_size=2)
        self.block_2 = nn.Conv2d(128, 64, 1)
        self.block_3 = nn.Conv2d(64, 128, kernel_size=2)
        self.drop_cnn = nn.Dropout(0.1)
        self.drop_fc1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4224, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 4)

    def forward(self, state):
        x1_2 = torch.relu(self.conv1_2(state))
        x2_1 = torch.relu(self.conv2_1(state))
        x1_2_2 = torch.relu(self.sec_conv(x1_2))
        x2_1_2 = torch.relu(self.sec_conv(x2_1))
        x2_pool = torch.relu(self.conv2_pool(state))
        x2_pool = torch.max_pool2d(x2_pool, 2)
        bl = torch.relu(self.block_1(state))
        bl = torch.relu(self.block_2(bl))
        bl = torch.relu(self.block_3(bl))
        bl_pool = torch.max_pool2d(bl, 2)
        x = torch.cat([
            x1_2.view(state.size(0), 1, -1),
            x2_1.view(state.size(0), 1, -1),
            x1_2_2.view(state.size(0), 1, -1),
            x2_1_2.view(state.size(0), 1, -1),
            x2_pool.view(state.size(0), 1, -1),
            bl.view(state.size(0), 1, -1),
            bl_pool.view(state.size(0), 1, -1),
        ], dim=2)
        #x = self.drop_cnn(x)
        x = torch.relu(self.fc1(x))
        #x = self.drop_fc1(x)
        x = torch.relu(self.fc2(x))
        return self.out(x)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.03, eps_decay=0.996,
                            gamma=0.995, memory_capacity=5000, batch_size=256, alpha=5e-3, tau=1e-3):
        self.local_Q = Brain().to(device)
        self.target_Q = Brain().to(device)
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
            #self.local_Q.eval()
            with torch.no_grad():
                obs = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                action = torch.argmax(self.local_Q(obs)).item()
            #self.local_Q.train()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                self.replay_memory.sample(self.batch_size)

        local_out = self.local_Q(state_batch)
        target = local_out.clone()
        target_out = torch.max(self.target_Q(next_state_batch), dim=2)[0].squeeze(1)
        target[self.indexes,0,action_batch] = reward_batch + self.gamma * target_out * done_batch

        loss = self.loss(local_out, target.detach()).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        saved = {
            "local": self.local_Q.state_dict(),
            "target": self.target_Q.state_dict(),
            "optimizer": self.optimizer,
            "loss": self.loss,
            "eps_decay": self.eps_decay,
            "eps_end": self.eps_end,
            "eps_start": self.eps_start,
            "tau": self.tau,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "scores": self.scores,
            "episodes": self.episodes,
            "durations": self.durations,
            "start": self.start,
            "replay_memory": self.replay_memory,
        }
        pickle.dump(saved, pickle_out)
        pickle_out.close()

    def load(self, filename):
        pickle_in = open(filename+".tt", mode="rb")
        info = pickle.load(pickle_in)
        self.local_Q.load_state_dict(info["local"])
        self.target_Q.load_state_dict(info["target"])
        self.optimizer = info["optimizer"]
        self.loss = info["loss"]
        self.eps_decay = info["eps_decay"]
        self.eps_end = info["eps_end"]
        self.eps_start = info["eps_start"]
        self.tau = info["tau"]
        self.gamma = info["gamma"]
        self.batch_size = info["batch_size"]
        self.scores = info["scores"]
        self.episodes = info["episodes"]
        self.durations = info["durations"]
        self.start = info["start"]
        self.replay_memory = info["replay_memory"]
        pickle_in.close()


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            if self.memory[int(self.position)][2] > 7:
                self.position = (self.position + 1) % self.capacity
                self.push(args)
                return
        self.memory[int(self.position)] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        batch = random.sample(self.memory, size)

        batch = list(zip(*batch))
        
        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        action_batch = torch.tensor(batch[1])
        reward_batch = torch.tensor(batch[2], device=device)
        next_state_batch = torch.tensor(batch[3], device=device, dtype=torch.float)
        done_batch = torch.tensor(batch[4], device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
