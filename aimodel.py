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
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, action_size)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
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
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, 2), random.randint(0, 10), random.randint(0, 31), random.randint(0, 31)]
        
        state_array = np.array(state)
        state_tensor = torch.FloatTensor(state_array).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        
        act_values = torch.clamp(act_values, 0, 31).round().int()
        return act_values.cpu().numpy().flatten().tolist()

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            state_array = np.array(state).flatten()
            next_state_array = np.array(next_state).flatten()
            state_tensor = torch.FloatTensor(state_array).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state_array).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(state_tensor)
            target_f = target_f.clone().detach()
            if action[0] < target_f.size(0):
                target_f[action[0]] = target
            states.append(state_tensor)
            targets_f.append(target_f)
        states = torch.stack(states)
        targets_f = torch.stack(targets_f)
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, targets_f)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
scores = []
epsilons = []
episodelengths = []

episodes = 200
maxscores = -(10**10)
episodewithmaxscore = -1
longestepisode = -(10**10)
longestepisodenum = -1

episodenum = 0

def main():
    global scores, epsilons, episodelengths, episodes, maxscores, episodewithmaxscore, longestepisode, longestepisodenum, episodenum
    env = gym.make('MiniMetro-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size, action_size)

    torch.set_num_threads(16)

    for e in range(episodes):
        episodenum = e
        state = env.reset()[0]
        done = False
        time = 0
        eplen = 0

        while not done:
            action = agent.act([state])
            next_state, reward, done, _, _ = env.step(action)
            agent.remember([state], action, reward, [next_state], done)
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
                if eplen > longestepisode:
                    longestepisode = eplen
                    longestepisodenum = e
                break

        agent.replay()
        if (e / episodes) * 100 % 1 == 0:
            agent.save("snapshot" + str(round((e/episodes) * 100)) + ".pth")
            with open("log.txt", "a") as txt:
                txt.write(f"Maximum score: {maxscores} in episode: {episodewithmaxscore} \t Maximum length: {longestepisode} in episode: {longestepisodenum} \t Snapshot stored in snapshot{str(round((e/episodes) * 100))}.pth\n")
    env.close()

    agent.save("model.pth")
    print(f"Maximum Score: {maxscores} in episode {episodewithmaxscore}")

if __name__ == "__main__":
    main()