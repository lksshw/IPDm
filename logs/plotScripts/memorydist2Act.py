#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

def load_file(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    return data

def getScores(labels, counts, memScr):
    #we need the avg scores of memory sizes at the last time step
    scores = []
    for i, lbl in enumerate(labels):
        lbl_scrs = memScr[lbl]
        #extract the last n appends from list
        lbl_scr = lbl_scrs[-counts[i]:] #the scr map accumulates scores of each memory-size for every time step, so we only extract the n appends added to each memory-size during the last time step
        mean_score = np.mean(lbl_scr)
        scores.append(mean_score)
    return scores

#file paths
curr_path = os.getcwd()
os.chdir("../")
logs_dir = os.getcwd()

cdmMem_path = os.path.join(logs_dir, "twoActLogs/checkpoints/memorySizeList-vs-time.pkl")

cdmMemScr_path = os.path.join(logs_dir, "twoActLogs/checkpoints/memory_score_map.pkl")

#load files
cdmMem = load_file(cdmMem_path)
memInfo = cdmMem[-1] #get memory counts at the last timestep
labels, counts = np.unique(memInfo, return_counts=True)

memScr = load_file(cdmMemScr_path)
cdmMemData = getScores(labels, counts, memScr)

#plot
fig, ax1 = plt.subplots(figsize = (8, 4))

color = 'tab:blue'
ax1.bar(list(range(len(labels))), counts, align='center', color = color)
ax1.set_xticks(list(range(len(labels))), labels)
ax1.set_xlabel("Memory Size")
ax1.set_ylabel("Frequency", color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.plot(cdmMemData, color=color)
ax2.set_ylabel('Avg. Score', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig(os.path.join(curr_path, "figs/memorysize2Act.png"), dpi=300)
plt.show()

