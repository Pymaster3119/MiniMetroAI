import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
#Stationtypes - 1 is circle, 2 is triangle, 3 is rectangle, and 4 is misc
stationtypes = np.zeros((30, 30))
connections = np.zeros((30,30,6))
routes = np.zeros((7,8,2))
timer = 0
stationspawntimer = 0
passangerspawntimer = 0
spawnweights = [0.7, 0.15, 0.1, 0.05]
metros = np.zeros((28,8))
gameended = False
score = 0
metrospeed = 5
for i in range(metros.shape[0]):
    metros[i][0] = -1

def updateGame(timestamp):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score, metrospeed
    if gameended:
        return
    timer += timestamp
    #Spawn a station (if applicable)
    if timer - 0.5 >= stationspawntimer:
        stationspawntimer = timer
        stationtypes[random.randint(1,29)][random.randint(1,29)] = random.choices(range(1,5), spawnweights)[0]

    #Add passangers
    if timer - 0.2 >= passangerspawntimer:
        index = np.nonzero(stationtypes)
        if len(index[0]) != 0:
            passangerspawntimer = timer
            indexchoice = random.randint(0, len(index[0])-1)
            index = (index[0][indexchoice], index[1][indexchoice])
            commuter = random.choices(range(1,5), spawnweights)[0]
            passangerspawntimer = timer
            assigned = False
            for i in range(6):
                if connections[index[0]][index[1]][i] == 0:
                    connections[index[0]][index[1]][i] = commuter
                    assigned = True
                    break
            if not assigned:
                print("GAME OVER")
                gameended = True

    #Move the metros
    indeces = []
    for idx, i in enumerate(metros):
        if i[0] != -1:
            indeces.append(idx)
    if len(indeces) != 0:
        for i in indeces:
            #Find the length of the route, assuming all straight lines
            length = 0
            previousPoint = (0,0)
            for j in routes[int(metros[i][0])]:
                if not previousPoint == (0,0):
                    length += math.sqrt((previousPoint[0] - j[0]) ** 2 + ((previousPoint[1] - j[1]) ** 2))
                previousPoint = (j[0], j[1])
            try:
                metros[int(i)][1] += (timestamp * metrospeed)/length
            except:
                pass

def addToMetroLine(line, stopindex):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
    #Check if they are valid, and if they aren't then take off some score
    if stationtypes[stopindex[0]][stopindex[1]] == 0:
        score -= 10
        return

    index = 0
    for i in range(routes.shape[1]):
        if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
            index = i
    if not (routes[line][0][0] == 0 and routes[line][0][1] == 0):
        score -= 10
        return
    
    #Add point
    routes[line][index + 1][0] = stopindex[0]
    routes[line][index + 1][1] = stopindex[1]

def addMetroToLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
    index = -1
    for i in range(metros.shape[0]):
        if metros[i][0]==0 and metros[i][1]==0:
            index = i
    if index == -1:
        score -= 10
    metros[index][0] = line

def removeLastPointFromMetroLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
    index = -1
    for i in range(routes.shape[1]):
        if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
            index = i
    if index == -1:
        return
    routes[line, index, 0]=0
    routes[line, index, 1]=0
    
fig = None
if __name__ == "__main__":
    while True:
        updateGame(10)
        print(score)
        if fig != None:
            plt.close(fig)
        fig, axs = plt.subplots(1, 3, figsize=(10, 5)) 
        axs[0].imshow(stationtypes, cmap='gray')
        axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        axs[0].grid()
        axs[1].imshow(np.count_nonzero(connections, axis=2), cmap='gray')

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
                try:
                    length = 0
                    previousPoint = (0,0)
                    i=routes[int(m[0])]
                    for j in i:
                        if previousPoint[0] != 0 and previousPoint[1] != 0 and j[0] != 0 and j[1] != 0:
                            length += math.sqrt((j[0]-previousPoint[0])**2 + (j[1]-previousPoint[1])**2)
                        previousPoint = j
                    distancecovered = (m[1]%100)/length
                    print(distancecovered)
                    length = 0
                    previousPoint = (0,0)
                    for j in i:
                        if previousPoint[0] != 0 and previousPoint[1] != 0 and j[0] != 0 and j[1] != 0:
                            prevlen = length
                            length += math.sqrt((j[0]-previousPoint[0])**2 + (j[1]-previousPoint[1])**2)
                            if prevlen <= distancecovered <= length:
                                lengthalongline = distancecovered - prevlen
                                print(lengthalongline)
                                distance = math.sqrt((j[0]-previousPoint[0])**2 + (j[1]-previousPoint[1])**2)
                                dir = (np.array(j) - np.array(previousPoint))/distance
                                point = np.array(previousPoint) + (distance * dir)
                                print(point )
                                rect = plt.Rectangle(point - 0.5, 3, 1, color="blue") 
                                axs[2].add_patch(rect)
                        previousPoint = j
                except:
                    pass
        plt.show(block=False)

        line = int(input("What line"))
        x = int(input("what x"))
        y = int(input("what y"))
        addToMetroLine(line, (x, y))

        if input("Do you wanna remova line?") == "t":
            removeLastPointFromMetroLine(int(input("What line?")))
        
        addMetroToLine(int(input("What line do you wanna add a train to?")))
