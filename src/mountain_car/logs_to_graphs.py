import re
import numpy as np
from  matplotlib import pyplot as plt
#import seaborn

p = re.compile(r'training session \d+ done in (\d+) steps, espsilon: ([\d.]+)')

def file_to_stats(name):
    with open("logs/"+name) as f:
        lines = f.readlines()

    steps = []
    epsilons = []
    for line in lines:
        match = p.search(line)
        if match is not None:
            steps.append(int(match.group(1)))
            epsilons.append(float(match.group(2)))
    return steps, epsilons

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_log_with_smooth(ax, steps):
    WINDOW = 50
    x = np.arange(0, 1000)
    y = steps
    ax.plot(x, y, linewidth=0.5, alpha=0.5)
    
    x = moving_average(x,WINDOW)
    y = moving_average(y,WINDOW)
    ax.plot(x, y, linewidth=1, alpha=0.7, color="green")


def compare_training(log1, log2, name: str):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0.1})
    ax1_b = ax1.twinx()
    ax2_b = ax2.twinx()
    #["log_20000_2.txt", "log_20000_3.txt"]:
    steps, _ = file_to_stats(log1)
    plot_log_with_smooth(ax1,steps)
    steps, epsilons = file_to_stats(log2)
    plot_log_with_smooth(ax2,steps)

    ax1_b.plot(epsilons, color="orange", linewidth=1)
    ax2_b.plot(epsilons, color="orange", linewidth=1)

    ax1.set_ylim([200,80])
    #ax1.set_ylim(ax1.get_ylim()[::-1])
    #ax1_b.set_ylim(ax1.get_ylim()[::-1])
    #ax2.set_ylim(ax2.get_ylim()[::-1]) sharing y only need to do one

    ax2.set_xlabel("training iteration")
    ax1.set_ylabel("steps needed")
    ax2.set_ylabel("steps needed")
    ax1_b.set_ylabel("epsilon")
    ax2_b.set_ylabel("epsilon")

    plt.tight_layout()
    plt.savefig("figs/"+name+".png", dpi=300)
    plt.close()

def plot_single(log, name: str):
    fig, ax1= plt.subplots(1, figsize=(7,3))
    ax1_b = ax1.twinx()
    #["log_20000_2.txt", "log_20000_3.txt"]:
    steps, epsilons = file_to_stats(log)
    plot_log_with_smooth(ax1,steps)

    ax1_b.plot(epsilons, color="orange", linewidth=1)

    ax1.set_ylim([200,80])
    #ax1.set_ylim(ax1.get_ylim()[::-1])
    #ax1_b.set_ylim(ax1.get_ylim()[::-1])
    #ax2.set_ylim(ax2.get_ylim()[::-1]) sharing y only need to do one

    ax1.set_xlabel("training iteration")
    ax1.set_ylabel("steps needed")
    ax1_b.set_ylabel("epsilon")

    plt.tight_layout()
    plt.savefig("figs/"+name+".png", dpi=300)
    plt.close()

compare_training("log_20000_0.txt", "log_20000_1.txt", "mcar_20000_a")
plot_single("log_20000_3.txt", "mcar_20000_b")

compare_training("log_10000_1.txt", "log_10000_2.txt", "mcar_10000")

compare_training("log_40000_0.txt", "log_40000_1.txt", "mcar_40000_a")
compare_training("log_40000_2.txt", "log_40000_3.txt", "mcar_40000_b")

plot_single("log_40000_3_no_iwu.txt", "mcar_40000_no_iwu")