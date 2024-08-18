import rungame as rg
import gymnasium as gym
from gymnasium import spaces
class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = spaces.Box(low=rg.np.array([-2] * 6636), high=rg.np.array([10] * 6636), dtype=rg.np.int64)
        self.observation_space = spaces.Box(low=-1, high=10, shape=(6636,), dtype=rg.np.int64)
        self.state = None

    def reset(self, seed = None, options = None):
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
        self.state = self.gatherstate()
        return (self.state, {})

    def step(self, action):
        rg.updateGame(0.1)
        if action[0] == 0:
            rg.addMetroToLine(action[1])
        if action[0] == 1:
            rg.addToMetroLine(action[1], action[2], action[3])
        if action[0] == 2:
            rg.removeLastPointFromMetroLine(action[1])

        reward = rg.score
        done = rg.gameended
        return self.state, reward, done, False, {}

    def close(self):
        pass

    def gatherstate(self):
        state = []
        for i in rg.stationtypes:
            for j in i:
                state.append(j)
        for i in rg.connections:
            for j in i:
                for k in j:
                    state.append(k)
        for i in rg.routes:
            for j in i:
                for k in j:
                    state.append(k)
        for i in rg.metros:
            for j in i:
                state.append(j)
        return rg.np.array(state, dtype=rg.np.int64)