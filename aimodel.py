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
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2) 
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2) 

        # Calculate the output size after convolutions
        conv_output_size = (state_size - 4) // 2  # Output size after first conv
        conv_output_size = (conv_output_size - 4) // 2  # Output size after second conv
        conv_output_size *= 32  # Number of channels from the last conv layer

        # Define fully connected layers
        layersizes = [conv_output_size, 32, 32, action_size]
        layers = []
        for i in range(1, len(layersizes)):
            layers.append(nn.Linear(layersizes[i-1], layersizes[i]))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape for Conv1D
        x = torch.relu(self.conv1(x))  # Apply first Conv1D layer
        x = torch.relu(self.conv2(x))  # Apply second Conv1D layer
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        for i in self.layers:
            x = torch.relu(i(x))
        return x

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = 100
        self.train_start = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)  # Single model with 3 output actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, 2), random.randint(0, 10), random.randint(0, 31), random.randint(0, 31)]
        state_tensor = torch.FloatTensor(state).to(self.device, non_blocking=True)
        
        # Forward pass through the model
        act_values = self.model(state_tensor)
        
        # Clamp the output and return the rounded actions
        act_values = torch.clamp(act_values, 0, 31)
        return [round(act_values[0][0].item()), round(act_values[0][1].item()), round(act_values[0][2].item()), round(act_values[0][3].item())]

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
            target_f = target_f.clone()
            if action[0] < target_f.size(1):
                target_f[0][action[0]] = target
            target_f = target_f.view(1, -1)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

env = gym.make('MiniMetro-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(state_size, action_size)

episodes = 50_000
maxscores = -(10**10)
episodewithmaxscore = -1
longestepisode = -(10**10)
longestepisodenum = -1

scores = []
epsilons = []
episodelengths = []
def exit():
    agent.save("model.pth")

    print(f"Maximum Score: {maxscores} in episode {episodewithmaxscore}")

for e in tqdm.tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    eplen = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1
        eplen += 1
        if done:
            scores.append(reward)
            epsilons.append(agent.epsilon)
            episodelengths.append(eplen)
            if reward > maxscores:
                maxscores = reward
                episodewithmaxscore = e
                print(f"High score of {reward} points!\n")
            if eplen > longestepisode:
                longestepisode = eplen
                longestepisodenum = e
                print(f"Longest episode so far with length {eplen}!\n")
            break
    
    agent.replay()
exit()