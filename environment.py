import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from rungame import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Env(gymnasium.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = spaces.Box(low=np.array([0] * 4), high=np.array([31] * 4), dtype=np.int64)
        self.observation_space = spaces.Box(low=-1, high=10, shape=(6468,), dtype=np.int64)
        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, spawnweights, metros, gameended, score, metrospeed
        
        stationtypes = torch.zeros((30, 30), dtype=torch.int32, device=device)
        connections = torch.zeros((30, 30, 6), dtype=torch.int32, device=device)
        routes = torch.zeros((7, 8, 2), dtype=torch.int32, device=device)
        timer = 0
        stationspawntimer = 0
        passangerspawntimer = 0
        spawnweights = torch.tensor([0.7, 0.15, 0.1, 0.05], device=device)
        metros = torch.zeros((7, 8), dtype=torch.int32, device=device)
        gameended = False
        score = 0
        metrospeed = 5
        metros[:, 0] = -1

        self.state = self.gatherstate()
        return self.state, {}

    def step(self, action):
        
        updateGame(0.1)

        
        if action[0] == 0:
            addMetroToLine(action[1])
        elif action[0] == 1:
            addToMetroLine(action[1], action[2], action[3])
        elif action[0] == 2:
            removeLastPointFromMetroLine(action[1])

        
        self.state = self.gatherstate()

        
        reward = score
        done = gameended

        return self.state, reward, done, False, {}

    def close(self):
        pass

    def gatherstate(self):
        
        state = []

        state.extend(stationtypes.cpu().numpy().flatten())
        state.extend(connections.cpu().numpy().flatten())
        state.extend(routes.cpu().numpy().flatten())
        state.extend(metros.cpu().numpy().flatten())
        print(len(state))
        return np.array(state, dtype=np.int64)