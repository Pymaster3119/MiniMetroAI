import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#Stationtypes - 1 is circle, 2 is triangle, 3 is rectangle, and 4 is misc
stationtypes = torch.zeros((30, 30), dtype=torch.int32, device=device)
#List of passangers - 6 passangers max
connections = torch.zeros((30, 30, 6), dtype=torch.int32, device=device)
#List of routes - 7 routes, 8 stations/route, 2 coordinates/stations
routes = torch.zeros((7, 8, 2), dtype=torch.int32, device=device)
timer = 0
stationspawntimer = 0
passangerspawntimer = 0
spawnweights = torch.tensor([0.7, 0.15, 0.1, 0.05], device=device)
#Metros - 7 metros, for each metro, 0 represents the route number (-1 is invalid), 1 reprersents distance along route, and 2-7 represents the passangers
metros = torch.zeros((7, 8), dtype=torch.int32, device=device)
gameended = False
score = 0
metrospeed = 5
metros[:, 0] = -1

def updateGame(timestamp):
    global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score, metrospeed
    if gameended:
        return

    timer += timestamp

    # Spawn a station (if applicable)
    if timer - 0.5 >= stationspawntimer:
        stationspawntimer = timer
        x, y = random.randint(1, 29), random.randint(1, 29)
        stationtypes[x, y] = random.choices(range(1, 5), spawnweights.tolist())[0]

    # Add passengers
    if timer - 0.2 >= passangerspawntimer:
        nonzero_indices = stationtypes.nonzero(as_tuple=True)
        if len(nonzero_indices[0]) != 0:
            passangerspawntimer = timer
            indexchoice = random.randint(0, len(nonzero_indices[0]) - 1)
            x, y = nonzero_indices[0][indexchoice].item(), nonzero_indices[1][indexchoice].item()
            commuter = random.choices(range(1, 5), spawnweights.tolist())[0]
            passangerspawntimer = timer
            assigned = False
            for i in range(6):
                if connections[x, y, i] == 0:
                    connections[x, y, i] = commuter
                    assigned = True
                    break
            if not assigned:
                gameended = True

    # Deal with the metros
    for i in range(7):
        metro = metros[i]
        route = metro[0].item()
        if route != -1:
            distancealongroute = metro[1].item()
            totallinelength = lengthAlongLine(route)
            if totallinelength != 0:
                # Update the metro's position
                metros[i, 1] = metrospeed / totallinelength

                # If next to a station, pick people up
                pos = findPositionOfMetro(metro)
                for j in range(8):
                    station = routes[int(route), int(j)]
                    if pos[0] != 0 and pos[1] != 0 and torch.sqrt((station[0] - pos[0]) ** 2 + (station[1] - pos[1]) ** 2) < 1:
                        # Unload passengers
                        stationtype = stationtypes[station[0], station[1]]
                        for k in range(2, 8):
                            if metro[k] == stationtype:
                                metro[k] = 0
                                score += 1

                        # Load up passengers
                        for k in range(5, -1, -1):
                            passangertype = connections[station[0], station[1], k]
                            for l in range(2, 8):
                                if metro[l] == 0:
                                    metros[i, l] = passangertype
                                    connections[station[0], station[1], k] = 0
                                    break

def findPositionOfMetro(metro):
    route = metro[0].item()
    distancealongroute = metro[1].item()
    routelength = lengthAlongLine(route)
    distancecovered = distancealongroute / routelength
    length = 0
    for i in range(8):
        station = routes[int(route), int(i)]
        laststation = routes[int(route), int(i - 1)]
        prevlength = length
        length += torch.sqrt((station[0] - laststation[0]) ** 2 + (station[1] - laststation[1]) ** 2).item()
        if prevlength < distancealongroute < length:
            x1, y1 = station
            x2, y2 = laststation
            dx, dy = x2 - x1, y2 - y1
            length = torch.sqrt(dx ** 2 + dy ** 2).item()
            dx /= length
            dy /= length
            x = x1 + dx * (distancealongroute - prevlength)
            y = y1 + dy * (distancealongroute - prevlength)
            return (x, y)
    return (0, 0)

def lengthAlongLine(line):
    length = 0
    for i in range(8):
        station = routes[int(line), int(i)]
        laststation = routes[int(line), int(i - 1)]
        length += torch.sqrt((station[0] - laststation[0]) ** 2 + (station[1] - laststation[1]) ** 2).item()
    return length

def addToMetroLine(line, stopindexx, stopindexy):
    try:
        if line >= 7:
            raise IndexError
        global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
        stopindex = (stopindexx, stopindexy)
        # Check if they are valid, and if they aren't then take off some score
        if stationtypes[stopindex[0], stopindex[1]] == 0:
            score -= 10
            return

        index = 0
        for i in range(routes.shape[1]):
            if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
                index = i
        if not (routes[line, 0, 0] == 0 and routes[line, 0, 1] == 0):
            score -= 10
            return

        # Add point
        routes[line, index + 1] = torch.tensor([stopindexx, stopindexy], dtype=torch.int32, device=device)
    except IndexError:
        score -= 10

def addMetroToLine(line):
    try:
        if line >= 7:
            raise IndexError
        global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
        index = -1
        for i in range(metros.shape[0]):
            if metros[i, 0] == 0 and metros[i, 1] == 0:
                index = i
        if index == -1:
            score -= 10
        metros[index, 0] = line
    except:
        score -= 10

def removeLastPointFromMetroLine(line):
    try:
        if line >= 7:
            raise IndexError
        global stationtypes, connections, routes, timer, stationspawntimer, passangerspawntimer, gameended, score
        index = -1
        for i in range(routes.shape[1]):
            if routes[line, i, 0] != 0 and routes[line, i, 1] != 0:
                index = i
        if index == -1:
            return
        routes[line, index] = torch.tensor([0, 0], dtype=torch.int32, device=device)
    except:
        score -= 10