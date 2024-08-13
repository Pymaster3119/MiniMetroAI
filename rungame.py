import numpy as np
import time
import random

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
metrospeed = 0

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
    indeces = np.nonzero(metros)
    for i in indeces[0]:
        metros[i][1] += timestamp * metrospeed
    print(metros)

def addToMetroLine(line, startindex, stopindex):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
    #Check if they are valid, and if they aren't then take off some score (and if it is the end of a line)
    if stationtypes[startindex[0]][startindex[1]] == 0 or stationtypes[stopindex[0]][stopindex[1]] == 0:
        score -= 10
        return

    index = 0
    for i in range(routes.shape[1]):
        if routes[0, i, 0] == 0 and routes[0, i, 1] == 0:
            index = i
    if not (routes[line][0][0] == 0 and routes[line][0][1] == 0) or not (routes[line][index - 1][0] == startindex[0] and routes[line][index - 1][1] == startindex[1]):
        score -= 10
        return
    
    #Add point
    routes[line][index][0] = stopindex[0]
    routes[line][index][1] = stopindex[1]

def addMetroToLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
    index = -1
    for i in range(metros.shape[0]):
        if metros[i][0]==0 and metros[i][1]==0:
            index = i
    if index == -1:
        score -= 10
    metros[index][0] = line

if __name__ == "__main__":

    addToMetroLine(0, (15,2),(3,23))
    for i in range(30):
        addMetroToLine(1)
    while True:
        #print(score)
        updateGame(0.1)
        time.sleep(0.1)