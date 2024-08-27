import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
# Set up the figure and multiple subplots
fig, axs = plt.subplots(6, 1)
x_data = []
highscores = []
highlengths = []

# Initialize data for the plots
lines = [ax.plot([], [])[0] for ax in axs]

# Initialize the plots
def init():
    for ax in range(5):
        axs[ax].legend()
    return lines

# Update function for the animation
def update(frame):
    global highscores, highlengths
    x_data = list(range(len(ai.episodelengths)))

    if len(highscores) < len(x_data):
        highscores.extend([ai.maxscores] * (len(x_data) - len(highscores)))
        highlengths.extend([ai.longestepisode] * (len(x_data) - len(highlengths)))

    data = [ai.episodelengths, ai.scores, ai.epsilons, highscores, highlengths]

    for line, y_data in zip(lines, data):
        line.set_data(x_data, y_data)
        
    for ax in axs:
        ax.relim()
        ax.autoscale_view()

    axs[5].barh(0,ai.episodenum, color='skyblue')
    axs[5].set_xlim(0,ai.episodes)
    axs[5].set_xticks(np.arange(0, ai.episodes + 1, ai.episodes/50))

    return lines

def start():
    ai.main()

threading.Thread(target=start).start()

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=False)

plt.tight_layout()
plt.show()