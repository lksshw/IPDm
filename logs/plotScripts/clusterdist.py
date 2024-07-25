#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


def load_file(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    return data

def getScores(labels, counts, cdmScr):
    #we need scores of clusters from their counts
    scores = []
    for i, lbl in enumerate(labels):
        lbl_scrs = cdmScr[lbl]
        #extract the last n appends from list
        lbl_scr = lbl_scrs[-counts[i]:] #the scr map accumulates scores of each cluster-size for every time step, so we only extract the n appends added to each cluster-size during the last time step
        mean_score = np.mean(lbl_scr)
        scores.append(mean_score)

    return scores

#file paths
curr_path = os.getcwd()
os.chdir("../")
logs_dir = os.getcwd()

cdmSize_path = os.path.join(logs_dir, "CDMLogs/checkpoints/clustersizeList-vs-time.pkl")

cdmScr_path = os.path.join(logs_dir, "CDMLogs/checkpoints/cluster_score_map.pkl")

#load files
cdmSize = load_file(cdmSize_path)
clusterInfo = cdmSize[-1] #get counts at the last timestep
labels, counts = np.unique(clusterInfo, return_counts=True)

cdmScr = load_file(cdmScr_path)
cdmScrData = getScores(labels, counts, cdmScr)

#plot
fig, ax1 = plt.subplots(figsize = (8, 4))

color = 'tab:blue'
ax1.bar(list(range(len(labels))), counts, align='center', color = color)
ax1.set_xticks(list(range(len(labels))), labels)
ax1.set_xlabel("Cluster Size")
ax1.set_ylabel("Frequency", color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.plot(cdmScrData, color=color)
ax2.set_ylabel('Avg. Score', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig(os.path.join(curr_path, "figs/clustersize.png"), dpi=300)
plt.show()

