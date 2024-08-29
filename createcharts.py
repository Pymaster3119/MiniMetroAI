import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
import time
from scipy.ndimage import gaussian_filter1d

# Create a figure for each graph
figs = [plt.figure() for _ in range(9)]
axs = [fig.add_subplot(1, 1, 1) for fig in figs]

x_data = []
highscores = []
highlengths = []

# Initialize data for the plots
lines = [ax.plot([], [])[0] for ax in axs]

# Initialize the plots
def init():
    for ax in range(7):
        axs[ax].legend([])
        axs[ax].set_xlabel("Episodes")
    return lines

def convertseconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remainingseconds = seconds % 60
    return f"{hours}:{minutes}:{remainingseconds}"

def exponentialmovingaverage(data, alpha):
    ema = [data[0]]
    for x in data[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema

def movingaverage(data, windowsize):
    return [sum(data[i:i+windowsize])/windowsize for i in range(len(data) - windowsize + 1)]

# Update function for the animation
def update(frame):
    global highscores, highlengths, starttime
    x_data = list(range(len(ai.episodelengths)))

    if len(highscores) < len(x_data):
        highscores.extend([ai.maxscores] * (len(x_data) - len(highscores)))
        highlengths.extend([ai.longestepisode] * (len(x_data) - len(highlengths)))

    windowsize = 10
    data = [ai.episodelengths, ai.scores, gaussian_filter1d(ai.scores, 10), ai.epsilons, highscores, highlengths, ai.errors, [ai.errors[i]/ai.episodelengths[i] for i in range(len(ai.errors))]]

    for line, y_data in zip(lines, data):
        line.set_data(x_data, y_data)

    for ax in axs[:-1]:
        ax.relim()
        ax.autoscale_view()
    axs[8].clear()
    axs[8].barh(0, ai.episodenum, color='skyblue')
    axs[8].set_xlim(0, ai.episodes)
    axs[8].set_xticks(np.arange(0, ai.episodes + 1, ai.episodes / 25))

    elapsedtime = time.time() - starttime
    axs[8].text((ai.episodes / 100) * 20, 0, f"Episodes/second: {ai.episodenum / elapsedtime}", color="black")
    axs[8].text((ai.episodes / 100) * 40, 0, f"Episode {ai.episodenum} of {ai.episodes}", color="black")
    axs[8].text((ai.episodes / 100) * 70, 0, f"High score {ai.maxscores} at point {ai.episodewithmaxscore}", color="black")
    axs[8].text((ai.episodes / 100) * 90, 0, f"Longest episode {ai.longestepisode} at point {ai.longestepisodenum}", color="black")
    axs[8].text(0, 0, f"Projected time: {convertseconds((ai.episodenum/elapsedtime) * ai.episodes)}")

    axs[0].set_ylabel('Episode Length')
    axs[1].set_ylabel('Score')
    axs[2].set_ylabel('Score Trend')
    axs[3].set_ylabel('Epsilon')
    axs[4].set_ylabel('High Score')
    axs[5].set_ylabel('Longest Episode')
    axs[6].set_ylabel('Errors')
    axs[7].set_ylabel('Errors per step')
    return lines

def start():
    ai.main()

starttime = time.time()

threading.Thread(target=start).start()

anis = [animation.FuncAnimation(fig, update, init_func=init, blit=False) for fig in figs]

for fig in figs:
    plt.figure(fig.number)
    plt.tight_layout()

plt.show()