import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from gymnasium.envs.registration import register
import tqdm
register(
    id='MiniMetro-v0',
    entry_point= 'environment:Env',
)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = DQN(state_size, 3).to(self.device) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, 2), random.randint(0, 6635), random.randint(0, 6635), random.randint(0, 6635)]
        state = torch.FloatTensor(state).to(self.device)
        act_values = self.model(state)
        action_type = torch.argmax(act_values).item()
        if action_type == 0:
            return [0, random.randint(0, 6635)]
        elif action_type == 1:
            return [1, random.randint(0, 6635), random.randint(0, 6635), random.randint(0, 6635)]
        else:
            return [2, random.randint(0, 6635)]

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).to(self.device)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).to(self.device)
            target_f = self.model(state)
            target_f[action[0]] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = gym.make('MiniMetro-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(state_size, action_size)

episodes = 1000

for e in tqdm.tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1

        if done:
            print(f"episode: {e}/{episodes}, score: {reward}, e: {agent.epsilon:.2}")
            break
    
    agent.replay()

agent.save("model.pth")