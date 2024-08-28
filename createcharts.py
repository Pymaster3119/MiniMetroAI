import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
import time
# Set up the figure and multiple subplots
fig, axs = plt.subplots(7, 1)
x_data = []
highscores = []
highlengths = []

# Initialize data for the plots
lines = [ax.plot([], [])[0] for ax in axs]

# Initialize the plots
def init():
    for ax in range(6):
        axs[ax].legend()
    return lines

# Update function for the animation
def update(frame):
    global highscores, highlengths, starttime
    x_data = list(range(len(ai.episodelengths)))

    if len(highscores) < len(x_data):
        highscores.extend([ai.maxscores] * (len(x_data) - len(highscores)))
        highlengths.extend([ai.longestepisode] * (len(x_data) - len(highlengths)))

    data = [ai.episodelengths, ai.scores, ai.epsilons, highscores, highlengths, ai.errors]

    for line, y_data in zip(lines, data):
        line.set_data(x_data, y_data)
        
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    axs[6].clear()
    axs[6].barh(0,ai.episodenum, color='skyblue')
    axs[6].set_xlim(0,ai.episodes)
    axs[6].set_xticks(np.arange(0, ai.episodes + 1, ai.episodes/25))

    elapsedtime = time.time() - starttime
    axs[6].text((ai.episodes/100) * 20, 0, f"Episodes/second: {ai.episodenum/elapsedtime}", color = "black")
    axs[6].text((ai.episodes/100) * 40, 0, f"Episode {ai.episodenum} of {ai.episodes}", color = "black")
    axs[6].text((ai.episodes/100) * 70, 0, f"High score {ai.maxscores} at point {ai.episodewithmaxscore}", color = "black")
    axs[6].text((ai.episodes/100) * 90, 0, f"Longest episode {ai.longestepisode} at point {ai.longestepisodenum}", color = "black")
    return lines

def start():
    ai.main()

starttime = time.time()

threading.Thread(target=start).start()

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=False)

plt.tight_layout()
plt.show()