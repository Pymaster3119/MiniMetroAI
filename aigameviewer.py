import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
import random
from aimodel import *
import rungame as rg
import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = DQN(state_size=6468, action_size=4)
model.load_state_dict(torch.load(input("Which model file do you want?")))
rg.counterrorsasreducedscore = input("Do you want to keep the errors counted as score? y/n").strip().lower() == 'y'
model.to(device)
model.eval()

def gatherstate():
    # Gather state from global tensors and convert them to NumPy arrays
    state = []

    # Ensure all tensors are on the GPU
    state.extend(rg.stationtypes.cpu().numpy().flatten())
    state.extend(rg.connections.cpu().numpy().flatten())
    state.extend(rg.routes.cpu().numpy().flatten())
    state.extend(rg.metros.cpu().numpy().flatten())

    return np.array(state, dtype=np.int64)

def getaiaction(state):
    state_array = np.array(state)
    state_tensor = torch.FloatTensor(state_array).to(device)
    with torch.no_grad():
        act_values = model(state_tensor)
    
    act_values = torch.clamp(act_values, 0, 31).round().int()
    return act_values.cpu().numpy().flatten().tolist()

axes = []
time.sleep(1)
while True:
    rg.updateGame(0.1)
    print(rg.score)
    if len(axes) != 0:
        for i in axs:
            i.clear()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 5)) 
    axs = []
    for i in range(2):
        for j in range(2):
            axs.append(axes[i][j])

    axs[0].imshow(rg.stationtypes.cpu(), cmap='gray')
    axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    axs[0].grid()
    axs[1].imshow(np.count_nonzero(rg.connections.cpu(), axis=2), cmap='gray')

    axs[2].imshow(np.zeros((30, 30, 3)))
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
    for idx, i in enumerate(rg.routes):
        x = []
        y = []
        for j in i:
            if j[0] != 0 and j[1] != 0:
                x.append(j[0])
                y.append(j[1])
        axs[2].plot(y, x, marker="o", color=colors[idx])

    for m in rg.metros:
        if m[0] != -1:
            pos = np.array(rg.findPositionOfMetro(m))
            rect = plt.Rectangle(pos - 0.5, 3, 1, color="blue") 
            axs[2].add_patch(rect)

    axs[3].text(0, 0, f'Score: {rg.score} Errors:{rg.errors}')
    plt.show(block=False)

    
    state = gatherstate()
    action = getaiaction(state)
    print(action)
    if action[0] == 0:
        rg.addMetroToLine(action[1])
    elif action[0] == 1:
        rg.addToMetroLine(action[1], action[2], action[3])
    elif action[0] == 2:
        rg.removeLastPointFromMetroLine(action[1])
    else:
        rg.score -= 100
    if rg.gameended:
        break

    #input('Press enter to continue')
    plt.pause(0.1)