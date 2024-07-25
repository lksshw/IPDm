#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def binned_avg(lst, chunk_size):
    #chunk a list into multiple sub-lists of chunk-size and
    #get the avg of each chunk

    lst_split = np.array_split(lst, chunk_size)
    new_lst = [np.mean(chunk) for chunk in lst_split]

    return new_lst


def load_file(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    return data

#file paths
curr_path = os.getcwd()
os.chdir("../")
logs_dir = os.getcwd()

cdm_path = os.path.join(logs_dir, "CDMLogs/checkpoints/cooperability-vs-time_merge.pkl")
tft_path = os.path.join(logs_dir, "TfTLogs/checkpoints/cooperability-vs-time_merge.pkl")
twoAct_path = os.path.join(logs_dir, "twoActLogs/checkpoints/cooperability-vs-time_merge.pkl")

#load cooperability files
cdm_coop = load_file(cdm_path)
tft_coop = load_file(tft_path)
twoAct_coop = load_file(twoAct_path)

#get binned versions
cdm_binned = binned_avg(cdm_coop, 100)
tft_binned = binned_avg(tft_coop, 100)
twoAct_binned = binned_avg(twoAct_coop, 100)

#plot
fig, ax = plt.subplots(1, 1, figsize = (12, 5))
ax.plot(cdm_binned, label = "IPD w/ merge (3 act)")
ax.plot(twoAct_binned, label = "IPD w/o merge (2 act)")
ax.plot(tft_binned, label = "IPD-tft (2 act, 1 len memory)")

ax.set_xlabel("N. games (x5000)")
ax.set_ylabel("Cooperability")

fig.legend()
os.chdir(curr_path)
plt.savefig(os.path.join(curr_path, "figs/cooperability.png"), dpi=300)
plt.show()

