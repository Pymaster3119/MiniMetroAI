from rungame import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, layers, width):
        super(DQN, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(n_observations, width))
        for i in range(1, layers):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, n_actions))

    def forward(self, x):
        for i in self.layers:
            x = F.relu(i(x))
        return x