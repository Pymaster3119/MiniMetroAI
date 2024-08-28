from rungame import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
import random
from aimodel import *
import rungame as rg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = DQN(state_size=6468, action_size=4)
model.load_state_dict(torch.load(input("Which model file do you want?")))
model.to(device)
model.eval()

def gatherstate(self):
    # Gather state from global tensors and convert them to NumPy arrays
    state = []

    # Ensure all tensors are on the GPU
    state.extend(rg.stationtypes.cpu().numpy().flatten())
    state.extend(rg.connections.cpu().numpy().flatten())
    state.extend(rg.routes.cpu().numpy().flatten())
    state.extend(rg.metros.cpu().numpy().flatten())

    return np.array(state, dtype=np.int64)

def getaiaction(state):
    statetensor = torch.FloatTensor(state).to(device)
    with torch.nograd():
        actvalues = model(statetensor)
    actvalues = torch.clamp(actvalues, 0, 31).round().int()
    return actvalues.cpu().numpy().flatten().tolist()

fig, axes = plt.subplots(2, 2, figsize=(10, 5)) 
axs = []
for i in range(2):
    for j in range(2):
        axs.append(axes[i][j])

while True:
    updateGame(10)
    print(score)
    for i in axs:
        i.clear()

    axs[0].imshow(stationtypes.cpu(), cmap='gray')
    axs[0].xaxis.setmajorlocator(plticker.MultipleLocator(base=1.0))
    axs[0].yaxis.setmajorlocator(plticker.MultipleLocator(base=1.0))
    axs[0].grid()
    axs[1].imshow(np.countnonzero(connections.cpu(), axis=2), cmap='gray')

    axs[2].imshow(np.zeros((30, 30, 3)))
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
    for idx, i in enumerate(routes):
        x = []
        y = []
        for j in i:
            if j[0] != 0 and j[1] != 0:
                x.append(j[0])
                y.append(j[1])
        axs[2].plot(y, x, marker="o", color=colors[idx])

    for m in metros:
        if m[0] != -1:
            pos = np.array(findPositionOfMetro(m))
            rect = plt.Rectangle(pos - 0.5, 3, 1, color="blue") 
            axs[2].addpatch(rect)

    axs[3].text(0, 0, f'Score: {score}')
    plt.show(block=False)

    
    state = gatherstate()
    action = getaiaction(state)
    
    if action[0] == 0:
        addMetroToLine(action[1])
    elif action[0] == 1:
        addToMetroLine(action[1], action[2], action[3])
    elif action[0] == 2:
        removeLastPointFromMetroLine(action[1])
    else:
        score -= 100

    if gameended:
        break