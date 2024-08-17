import rungame as rg
import gymnasium as gym
from gymnasium import spaces
class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=rg.np.float32)

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
        self.state = gatherstate()
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

    def gatherstate(self):
        pass