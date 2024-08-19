import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
# Set up the figure and multiple subplots
fig, axs = plt.subplots(5, 1)
x_data = []
highscores = []
highlengths = []

# Initialize data for the two plots
line1, = axs[0].plot([], [])
line2, = axs[1].plot([], [])
line3, = axs[2].plot([], [])
line4, = axs[3].plot([], [])
line5, = axs[4].plot([], [])

# Initialize the plots
def init():
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    return line1, line2, line3

# Update function for the animation
def update(frame):
    global highscores, highlengths
    x_data = [i for i in range(len(ai.episodelengths))]
    if len(highscores) < len(x_data):
        for i in range(len(x_data) - len(highscores)):
            highscores.append(ai.maxscores)
            highlengths.append(ai.longestepisode)c
    line1.set_data(x_data, ai.episodelengths)
    line2.set_data(x_data, ai.scores)
    line3.set_data(x_data, ai.epsilons)
    line4.set_data(x_data, highscores)
    line5.set_data(x_data, highlengths)
    axs[0].relim()
    axs[0].autoscale_view()
    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()
    axs[3].relim()
    axs[3].autoscale_view()
    axs[4].relim()
    axs[4].autoscale_view()
    
    return line1, line2, line3


def start():
    ai.main()
    
    
threading.Thread(target=start).start()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 10, 500), init_func=init, blit=False)

plt.tight_layout()
plt.show()