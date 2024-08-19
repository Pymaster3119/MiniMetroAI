import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
# Set up the figure and multiple subplots
fig, axs = plt.subplots(3, 1)
x_data = []

# Initialize data for the two plots
line1, = axs[0].plot([], [])
line2, = axs[1].plot([], [])
line3, = axs[2].plot([], [])

# Initialize the plots
def init():
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    return line1, line2, line3

# Update function for the animation
def update(frame):
    x_data = [i for i in range(len(ai.episodelengths))]
    line1.set_data(x_data, ai.episodelengths)
    line2.set_data(x_data, ai.scores)
    line3.set_data(x_data, ai.epsilons)
    axs[0].relim()
    axs[0].autoscale_view()
    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()
    print(np.linspace(min(x_data), max(x_data), num=5))
    axs[0].set_xticks(np.linspace(min(x_data), max(x_data), num=5))
    axs[0].set_yticks(np.linspace(min(ai.episodelengths), max(ai.episodelengths), num=5))
    return line1, line2, line3


def start():
    ai.main()
    
    
threading.Thread(target=start).start()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 10, 500), init_func=init, blit=False)

plt.tight_layout()
plt.show()