import re
import numpy as np
from  matplotlib import pyplot as plt
#import seaborn

p = re.compile(r'score: ([\d.]+), epsilon: ([\d.]+), session took: (\d+)\s+steps,')

def file_to_stats(name):
    with open("logs/"+name) as f:
        lines = f.readlines()

    step_sum = 0
    score = []
    steps_survived = []
    steps_started = []
    epsilons = []
    for line in lines:
        match = p.search(line)
        if match is not None:
            score.append(float(match.group(1)))
            steps_survived.append(int(match.group(3)))
            epsilons.append(float(match.group(2)))
            steps_started.append(step_sum)
            step_sum += int(match.group(3))
    return steps_survived, steps_started, score, epsilons

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_log_with_smooth(ax, steps):
    WINDOW = 50
    x = np.arange(0, len(steps))
    y = steps
    ax.plot(x, y, linewidth=0.5, alpha=0.5)
    
    x = moving_average(x,WINDOW)
    y = moving_average(y,WINDOW)
    ax.plot(x, y, linewidth=1, alpha=0.7, color="green")

def compare_plot(log, name: str, xlim=None, ylim=None):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1})
    ax1_b = ax1.twinx()
    ax2_b = ax2.twinx()
    steps, steps_started, score, epsilons = file_to_stats(log)
    plot_log_with_smooth(ax1,steps)
    plot_log_with_smooth(ax2,score)

    ax1_b.plot(epsilons, color="orange", linewidth=1)
    ax2_b.plot(epsilons, color="orange", linewidth=1)

    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim[0])
        ax2.set_ylim(ylim[1])

    ax2.set_xlabel("training iteration")
    ax1.set_ylabel("steps survived")
    ax2.set_ylabel("score achived")
    ax1_b.set_ylabel("epsilon")
    ax2_b.set_ylabel("epsilon")

    plt.tight_layout()
    plt.savefig("figs/"+name+".png", dpi=300)
    plt.close()

def plot_single(log, name: str):
    fig, ax1= plt.subplots(1, figsize=(7,3))
    ax1_b = ax1.twinx()
    steps, steps_started, score, epsilons = file_to_stats(log)
    plot_log_with_smooth(ax1, score)

    ax1_b.plot(epsilons, color="orange", linewidth=1)

    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("steps needed")
    ax1_b.set_ylabel("epsilon")

    plt.tight_layout()
    plt.savefig("figs/"+name+".png", dpi=300)
    plt.close()

compare_plot("log_breakout_1m.txt", "breakout_1m")
compare_plot("log_breakout_10m.txt", "breakout_10m")

compare_plot("log_breakout_10m.txt", "breakout_10m_zoomed", xlim=[35_400, 36_000])
compare_plot("log_breakout_1m.txt", "breakout_1m_zoomed", xlim=[0, 500], ylim=[[80, 400],[0,10]])