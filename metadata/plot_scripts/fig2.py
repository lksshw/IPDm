#!/usr/bin/env python3

# tissue-size vs memory capacity

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.rcParams['text.usetex'] = True

cwd_path = os.getcwd()
path = os.path.join(cwd_path, 'data/IPDmsLogs/')
save_path = os.path.join(cwd_path, 'figures/')
n_runs = 5
max_iter = int(50e3)
grid_size = 20

# std mult factor
mult_fact = 1.0
alpha = 0.3

# memory sizes
mem_sizes = [1, 2, 3, 4]

n_dataPoints = 500

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

def load(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    return data


def getStats(dat):
    mean = np.mean(dat, axis=0)
    std = np.std(dat, axis=0)
    return mean, std


def chunkProcess(dat):
    q = np.array_split(dat, n_dataPoints)
    t = np.array([np.mean(chunk) for chunk in q])
    return t


def plotFigs():
    shape = (n_runs, max_iter)
    for n, i in enumerate(mem_sizes):
        cluster_dat = np.zeros(shape)
        mem_path = os.path.join(path, f"mem_{i}/checkpoints")

        for r in range(n_runs):
            dat = load(os.path.join(
                mem_path, f"clustersizeList-vs-time-run{r}.pkl"))
            dat_custom = np.array(list(dat.values()))[:max_iter]
            pt,  = np.where(dat_custom == 400.0)
            if (len(pt)):
                trunc_pt = list(pt)[0]
            else:
                trunc_pt = max_iter
            cluster_dat[r][:trunc_pt] = dat_custom[:trunc_pt]
            cluster_dat[r][trunc_pt:max_iter] = 400.0

        row = n//2
        col = n % 2
        u_mean, u_std = getStats(cluster_dat)

        # trim to singularity
        u_mean = chunkProcess(u_mean)
        u_std = chunkProcess(u_std)

        pos,  = np.where(u_mean == 400.0)
        if (len(pos)):
            pos_idx = list(pos)[0]
            # print(pos_idx)
            u_mean[pos_idx+10:] = None
            u_std[pos_idx+10:] = None
            ax[row][col].scatter(pos_idx, 400.0, s=20, color='red')

        color = '#2D1E2F'
        ax[row][col].plot(u_mean, color=color)
        ax[row][col].fill_between(list(range(len(u_mean))), np.clip(
            u_mean-mult_fact*u_std, a_min=0.0, a_max=None), np.clip(u_mean+mult_fact*u_std, a_min=None, a_max=400.0), alpha=alpha, color=color)

        pos,  = np.where(u_mean == 400.0)
        if (len(pos)):
            pos_idx = list(pos)[0]
            # print(pos_idx)
            ax[row][col].annotate('Unified Agent', xy=(pos_idx, 400.0), xytext=(+35, -25), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                boxstyle='round,pad=0.2', fc='orange', alpha=0.3), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='red'))

        ax[row][col].set_ylabel("Avg. Tissue Size")
        ax[row][col].set_title(f"Memory = {i}")
        if (row+col > 0):
            x_vals = np.arange(0, 60, 10)
            x_labels = [f'{int(i)//10}' for i in x_vals]
            ax[row][col].set_xticks(x_vals, x_labels)
        else:
            x_vals = np.arange(0, 600, 100)
            x_labels = [f'{int(i)//10}' for i in x_vals]
            ax[row][col].set_xticks(x_vals, x_labels)

        ax[row][col].set_xlabel(r"Games Played (x$10^3$)")
        ax[row][col].set_xlabel(r"Games Played (x$10^3$)")

    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.45)

    plt.savefig(os.path.join(save_path, "fig2.svg"), format='svg')

if __name__ == "__main__":
    plotFigs()
