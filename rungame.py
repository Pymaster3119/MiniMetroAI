import torch
import random
import math

#Stationtypes - 1 is circle, 2 is triangle, 3 is rectangle, and 4 is misc
stationtypes = torch.zeros((30, 30), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#List of passengers - 6 passengers max
connections = torch.zeros((30, 30, 6), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#List of routes - 7 routes, 8 stations/route, 2 coordinates/stations
routes = torch.zeros((7, 8, 2), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
timer = 0
stationspawntimer = 0
passengerspawntimer = 0
spawnweights = [0.7, 0.15, 0.1, 0.05]
#Metros - 7 metros, for each metro, 0 represents the route number (-1 is invalid), 1 represents distance along route, and 2-7 represents the passengers
metros = torch.zeros((7, 8), device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
metros[:, 0] = -1
gameended = False
score = 0
metrospeed = 5
errors = 0
counterrorsasreducedscore = True


def updateGame(timestamp):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    if gameended:
        return
    timer += timestamp

    # Spawn a station (if applicable)
    if timer - 0.5 >= stationspawntimer:
        stationspawntimer = timer
        stationtypes[random.randint(1, 29)][random.randint(1, 29)] = random.choices(range(1, 5), spawnweights)[0]

    # Add passengers
    if timer - 0.2 >= passengerspawntimer:
        index = torch.nonzero(stationtypes)
        if len(index) != 0:
            passengerspawntimer = timer
            indexchoice = random.randint(0, len(index) - 1)
            index = index[indexchoice].tolist()
            commuter = random.choices(range(1, 5), spawnweights)[0]
            passengerspawntimer = timer
            assigned = False
            for i in range(6):
                if connections[index[0]][index[1]][i] == 0:
                    connections[index[0]][index[1]][i] = commuter
                    assigned = True
                    break
            if not assigned:
                gameended = True

    # Deal with the metros
    for i in range(7):
        metro = metros[i]
        route = metro[0]
        if route != -1:
            distancealongroute = metro[1]
            totallinelength = lengthAlongLine(route)
            if totallinelength != 0:
                # Update the metro's position
                metros[i][1] += metrospeed / totallinelength

                # If next to a station, pick people up
                pos = findPositionOfMetro(metro)
                for j in range(8):
                    station = routes[int(route)][int(j)]
                    if pos[0] != 0 and pos[1] != 0 and math.sqrt((station[0] - pos[0])**2 + (station[1] - pos[1])**2) < 1:
                        # Unload passengers
                        stationtype = stationtypes[int(station[0])][int(station[1])]
                        for k in range(2, 8):
                            if metro[k] == stationtype:
                                metro[k] = 0
                                score += 100

                        # Load up passengers
                        for k in range(5, -1, -1):
                            passangertype = connections[int(station[0])][int(station[1])][k]
                            for l in range(2, 8):
                                if metro[l] == 0:
                                    metros[i][l] = passangertype
                                    connections[int(station[0])][int(station[1])][k] = 0
                                    break

def findPositionOfMetro(metro):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    route = metro[0]
    distancealongroute = metro[1]
    routelength = lengthAlongLine(route)
    distancecovered = distancealongroute / routelength
    length = 0
    for i in range(8):
        station = routes[int(route)][int(i)]
        laststation = routes[int(route)][int(i - 1)]
        prevlength = length
        length += math.sqrt((station[0] - laststation[0])**2 + (station[1] - laststation[1])**2)
        if prevlength < distancealongroute < length:
            x1, y1 = station
            x2, y2 = laststation
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx**2 + dy**2)
            dx /= length
            dy /= length
            x = x1 + dx * (distancealongroute - prevlength)
            y = y1 + dy * (distancealongroute - prevlength)
            return (x, y)
    return (0, 0)

def lengthAlongLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    length = 0
    for i in range(8):
        station = routes[int(line)][int(i)]
        laststation = routes[int(line)][int(i - 1)]
        length += math.sqrt((station[0] - laststation[0])**2 + (station[1] - laststation[1])**2)
    return length

def addToMetroLine(line, stopindexx, stopindexy):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    try:
        if line >= 7:
            raise IndexError
        stopindex = (stopindexx, stopindexy)
        if stationtypes[stopindex[0]][stopindex[1]] == 0:
            if counterrorsasreducedscore:
                score -= 100
            
            errors += 1
            return

        index = 0
        for i in range(routes.shape[1]):
            if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
                index = i
        if not (routes[line][0][0] == 0 and routes[line][0][1] == 0):
            if counterrorsasreducedscore:
                score -= 100
            
            errors += 1
            return

        routes[line][index + 1][0] = stopindex[0]
        routes[line][index + 1][1] = stopindex[1]
        score += 10
    except IndexError:
        if counterrorsasreducedscore:
            score -= 100
        
        errors += 5

def addMetroToLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    try:
        if line >= 7:
            raise IndexError
        index = -1
        for i in range(metros.shape[0]):
            if metros[i][0] == 0 and metros[i][1] == 0:
                index = i
        if index == -1:
            if counterrorsasreducedscore:
                score -= 100
            
            errors += 1
        metros[index][0] = line
        score += 10
    except:
        if counterrorsasreducedscore:
            score -= 100
        
        errors += 1

def removeLastPointFromMetroLine(line):
    global stationtypes, connections, routes, timer, stationspawntimer, passengerspawntimer, gameended, score, metrospeed, errors
    try:
        if line >= 7:
            raise IndexError
        index = -1
        for i in range(routes.shape[1]):
            if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
                index = i
        if index == -1:
            return
        routes[line, index, 0] = 0
        routes[line, index, 1] = 0
        score += 10
    except:
        if counterrorsasreducedscore:
            score -= 100
        
        errors += 1