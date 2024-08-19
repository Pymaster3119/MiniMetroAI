import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading

fig, ax = plt.subplots(1,1)
x_data, y_data = [], []
line, = ax.plot([], [], 'r-')


def init():
    return line,
def update(frame):
    x_data.append(frame)
    y_data.append(np.sin(frame))
    
    line.set_data(x_data, y_data)
    ax.set_xlim(0, len(x_data))
    return line,
def start():
    global ai
    import aimodel as ai
    
threading.Thread(target=start).start()

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 10, 500), init_func=init, blit=True)

plt.show()