import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import aimodel as ai
import time
from scipy.ndimage import gaussian_filter1d

# Create a figure for each graph
figs = [plt.figure() for _ in range(11)]
axs = [fig.add_subplot(1, 1, 1) for fig in figs]

x_data = []
highscores = []
highlengths = []

# Initialize data for the plots
lines = [ax.plot([], [])[0] for ax in axs]

# Initialize the plots
def init():
    for ax in range(len(axs)):
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

def smooth(y, boxpts):
    if len(y) > boxpts:
        box = np.ones(boxpts)/boxpts
        return np.convolve(y, box, mode='same')
    else:
        return []

# Update function for the animation
prevhighscore = -(99 ** 3)
highscorexdata = []
prevhighlength = -(99 ** 3)
highlengthxdata = []
print(prevhighscore)
def update(frame):
    try:
        global highscores, highlengths, starttime, prevhighscore, prevhighlength, highlengthxdata, highscorexdata
        data = [ai.episodelengths, ai.scores, smooth(ai.scores, 5), smooth(ai.episodelengths, 5), ai.epsilons, [], [], ai.errors, [ai.errors[i] / ai.episodelengths[i] if ai.episodelengths[i] != 0 else 0 for i in range(len(ai.errors))], smooth(ai.errors, 5)]
        
        x_data = list(range(len(ai.episodelengths)))
        runedit = True
        for i in data:
            if len(i) != 0 and len(x_data) != len(i):
                runedit = False
        
        if runedit:
            if ai.maxscores > prevhighscore:
                if prevhighscore != -(99 * 3):
                    highscorexdata.append(ai.episodenum - 1)
                    highscores.append(prevhighscore)
                prevhighscore = ai.maxscores
                highscorexdata.append(ai.episodenum)
                highscores.append(prevhighscore)
            if ai.longestepisode > prevhighlength:
                if prevhighlength != -(99 * 3):
                    highlengthxdata.append(ai.episodenum - 1)
                    highlengths.append(prevhighlength)
                prevhighlength = ai.longestepisode
                highlengthxdata.append(ai.episodenum)
                highlengths.append(prevhighlength)


            

            for line, y_data in zip(lines, data):
                if len(y_data) != 0:
                    line.set_data(x_data, y_data)

            lines[5].set_data(highscorexdata, highscores)
            lines[6].set_data(highlengthxdata, highlengths)

            for ax in axs[:-1]:
                ax.relim()
                ax.autoscale_view()
            axs[10].clear()
            axs[10].barh(0, ai.episodenum, color='skyblue')
            axs[10].set_xlim(0, ai.episodes)
            axs[10].set_xticks(np.arange(0, ai.episodes + 1, ai.episodes / 25))

            elapsedtime = time.time() - starttime
            axs[10].text((ai.episodes / 100) * 20, 0, f"Episodes/second: {ai.episodenum / elapsedtime}", color="black")
            axs[10].text((ai.episodes / 100) * 40, 0, f"Episode {ai.episodenum} of {ai.episodes}", color="black")
            axs[10].text((ai.episodes / 100) * 70, 0, f"High score {ai.maxscores} at point {ai.episodewithmaxscore}", color="black")
            axs[10].text((ai.episodes / 100) * 90, 0, f"Longest episode {ai.longestepisode} at point {ai.longestepisodenum}", color="black")
            axs[10].text(0, 0, f"Projected time: {convertseconds((ai.episodenum/elapsedtime) * ai.episodes)}")

            axs[0].set_ylabel('Episode Length')
            axs[1].set_ylabel('Score')
            axs[2].set_ylabel('Score Trend')
            axs[3].set_ylabel('Length Trend')
            axs[4].set_ylabel('Epsilon')
            axs[5].set_ylabel('High Score')
            axs[6].set_ylabel('Longest Episode')
            axs[7].set_ylabel('Errors')
            axs[8].set_ylabel('Errors per step')
            axs[9].set_ylabel('Error trend')
    except:
        pass
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