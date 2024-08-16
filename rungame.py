import numpy as np
import random
import math
#Stationtypes - 1 is circle, 2 is triangle, 3 is rectangle, and 4 is misc
stationtypes = np.zeros((30, 30))
#List of passangers - 6 passangers max
connections = np.zeros((30,30,6))
#List of routes - 7 routes, 8 stations/route, 2 coordinates/stations
routes = np.zeros((7,8,2))
timer = 0
stationspawntimer = 0
passangerspawntimer = 0
spawnweights = [0.7, 0.15, 0.1, 0.05]
#Metros - 28 metros, for each metro, 0 represents the route number (-1 is invalid), 1 reprersents distance along route, and 2-7 represents the passangers
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

    #Deal with the metros
    for i in range(28):
        metro = metros[i]
        route = metro[0]
        distancealongroute = metro[1]
        totallinelength = lengthAlongLine(route)
        #Update the metro's position
        metros[i][1] != metrospeed/totallinelength
        distancecovered = distancealongroute/totallinelength

def lengthAlongLine(line):
    length = 0
    for i in range(8):
        station = routes[line][i]
        laststation = routes[line][i-1]
        length += math.sqrt((station[0]-laststation[0])**2 + (station[1]-laststation[1])**2)

    return length

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
    
