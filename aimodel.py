import rungame as rg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import gymnasium as gym
from gymnasium import spaces
import random

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.state = None

    def reset(self):
        rg.stationtypes = rg.np.zeros((30, 30))
        rg.connections = rg.np.zeros((30,30,6))
        rg.routes = rg.np.zeros((7,8,2))
        rg.timer = 0
        rg.stationspawntimer = 0
        rg.passangerspawntimer = 0
        rg.spawnweights = [0.7, 0.15, 0.1, 0.05]
        rg.metros = rg.np.zeros((28,8))
        rg.gameended = False
        rg.score = 0
        rg.metrospeed = 5
        self.state = 0
        return self.state

    def step(self, action):

        if action[0] == 0:
            rg.addMetroToLine(action[1])
        if action[0] == 1:
            rg.addToMetroLine(action[1], action[2])
        if action[0] == 2:
            rg.removeLastPointFromMetroLine(action[1])

        reward = rg.score
        done = rg.gameended
        return self.state, reward, done, False, {}

    def close(self):
        pass

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
    
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

#Action space - 1 descrete action to determine what move type to execute, then three integers determining what args are sent to that move. If ints are not needed, they are disregarded
n_actions = 4

#State space - 30x30 array for the stationtypes, 30x30x6 array for the passanger counts, 7x8x2 array representing the routes, and a 28x8 array representing the metros
n_observations = 6636

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0