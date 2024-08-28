from rungame import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
axes = []
while True:
    updateGame(10)
    print(score)
    if len(axes) != 0:
        for i in axs:
            i.clear()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 5)) 
    axs = []
    for i in range(2):
        for j in range(2):
            axs.append(axes[i][j])
    axs[0].imshow(stationtypes.cpu(), cmap='gray')
    axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    axs[0].grid()
    axs[1].imshow(np.count_nonzero(connections.cpu(), axis=2), cmap='gray')

    axs[2].imshow(np.zeros((30,30, 3)))
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
    for idx, i in enumerate(routes):
        x = []
        y = []
        for j in i:
            if j[0] != 0 and j[1] != 0:
                x.append(j[0])
                y.append(j[1])
        axs[2].plot(y, x, marker = "o", color = colors[idx])

    
    for m in metros:
        if m[0] != -1:
            pos = np.array(findPositionOfMetro(m))
            rect = plt.Rectangle(pos - 0.5, 3, 1, color="blue") 
            axs[2].add_patch(rect)
    
    axs[3].text(0,0,f'Score: {score}')
    plt.show(block=False)

    line = int(input("What line"))
    x = int(input("what x"))
    y = int(input("what y"))
    addToMetroLine(line, x, y)

    if input("Do you wanna remova line?") == "t":
        removeLastPointFromMetroLine(int(input("What line?")))
    
    addMetroToLine(int(input("What line do you wanna add a train to?")))
