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
        self.model_0 = DQN(state_size, 1).to(self.device)
        self.model_1 = DQN(state_size, 1).to(self.device)
        self.model_2 = DQN(state_size, 1).to(self.device)
        
        self.optimizer_0 = optim.Adam(self.model_0.parameters(), lr=self.learning_rate)
        self.optimizer_1 = optim.Adam(self.model_1.parameters(), lr=self.learning_rate)
        self.optimizer_2 = optim.Adam(self.model_2.parameters(), lr=self.learning_rate)
        
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_type = random.randint(0, 2)
            if action_type == 0:
                return [0, random.randint(0, 6635)]
            elif action_type == 1:
                return [1, random.randint(0, 6635), random.randint(0, 6635), random.randint(0, 6635)]
            else:
                return [2, random.randint(0, 6635)]
        state = torch.FloatTensor(state).to(self.device)
        q_value_0 = self.model_0(state).item()
        q_value_1 = self.model_1(state).item()
        q_value_2 = self.model_2(state).item()
        action_type = np.argmax([q_value_0, q_value_1, q_value_2])
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
                if action[0] == 0:
                    target = reward + self.gamma * self.model_0(next_state).item()
                elif action[0] == 1:
                    target = reward + self.gamma * self.model_1(next_state).item()
                elif action[0] == 2:
                    target = reward + self.gamma * self.model_2(next_state).item()

            state = torch.FloatTensor(state).to(self.device)
            target = torch.tensor([[target]], dtype=torch.float32, device=self.device)  # Ensure target is float32
            if action[0] == 0:
                target_f = self.model_0(state)
                self.optimizer_0.zero_grad()
                loss = self.criterion(target_f, target)
                loss.backward()
                self.optimizer_0.step()
            elif action[0] == 1:
                target_f = self.model_1(state)
                self.optimizer_1.zero_grad()
                loss = self.criterion(target_f, target)
                loss.backward()
                self.optimizer_1.step()
            elif action[0] == 2:
                target_f = self.model_2(state)
                self.optimizer_2.zero_grad()
                loss = self.criterion(target_f, target)
                loss.backward()
                self.optimizer_2.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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