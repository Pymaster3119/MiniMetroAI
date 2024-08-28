import gymnasium
from gymnasium import spaces
import numpy as np
import torch
import rungame as rg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Env(gymnasium.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = spaces.Box(low=np.array([0] * 4), high=np.array([31] * 4), dtype=np.int64)
        self.observation_space = spaces.Box(low=-1, high=10, shape=(6468,), dtype=np.int64)
        self.state = None
        self.reset()
        self.errors = 0

    def reset(self, seed=None, options=None):
        global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, spawnweights, metros, gameended, score, metrospeed
        
        #Stationtypes - 1 is circle, 2 is triangle, 3 is rectangle, and 4 is misc
        rg.stationtypes = torch.zeros((30, 30), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        #List of passengers - 6 passengers max
        rg.connections = torch.zeros((30, 30, 6), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        #List of routes - 7 routes, 8 stations/route, 2 coordinates/stations
        rg.routes = torch.zeros((7, 8, 2), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        rg.timer = 0
        rg.stationspawntimer = 0
        rg.passengerspawntimer = 0
        rg.spawnweights = [0.7, 0.15, 0.1, 0.05]
        #Metros - 7 metros, for each metro, 0 represents the route number (-1 is invalid), 1 represents distance along route, and 2-7 represents the passengers
        rg.metros = torch.zeros((7, 8), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        rg.metros[:, 0] = -1
        rg.gameended = False
        rg.score = 0
        rg.metrospeed = 5
        rg.errors = 0
        self.errors = 0

        self.state = self.gatherstate()
        return self.state, {}

    def step(self, action):
        # Perform game update on GPU
        rg.updateGame(1)

        # Execute action based on the action input
        if action[0] == 0:
            rg.addMetroToLine(action[1])
        elif action[0] == 1:
            rg.addToMetroLine(action[1], action[2], action[3])
        elif action[0] == 2:
            rg.removeLastPointFromMetroLine(action[1])

        # Gather the updated state
        self.state = self.gatherstate()

        # Compute reward and done status
        reward = rg.score
        done = rg.gameended
        self.errors = rg.errors
        return self.state, reward, done, False, {}

    def close(self):
        pass

    def gatherstate(self):
        # Gather state from global tensors and convert them to NumPy arrays
        state = []

        # Ensure all tensors are on the GPU
        state.extend(rg.stationtypes.cpu().numpy().flatten())
        state.extend(rg.connections.cpu().numpy().flatten())
        state.extend(rg.routes.cpu().numpy().flatten())
        state.extend(rg.metros.cpu().numpy().flatten())

        return np.array(state, dtype=np.int64)