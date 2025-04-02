#!/usr/bin/env python3

# unicelluar/multicellular fitness as a functino of memory size

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

n_dataPoints = 500  # bin and plot x data points

# std mult factor
mult_fact = 0.4
alpha = 0.3

# memory sizes
mem_sizes = [1, 2, 3, 4]

fig, ax = plt.subplots(2, 2, sharey=False, figsize=(11, 7))


def load(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    return data

def getStats(dat):
    mean = np.mean(dat, axis=0)
    std = np.std(dat, axis=0)
    return mean, std

def acc_run_data(fpath, u_dat, m_dat, r):
    dat = load(fpath)
    uni_temp_data = np.array([v["avg_U"] for k, v in dat.items()])[:max_iter]
    multi_temp_data = np.array([v["avg_M"]
                               for k, v in dat.items()])[:max_iter]
    u_dat[r] = uni_temp_data
    m_dat[r] = multi_temp_data

    return u_dat, m_dat

def chunkProcess(dat):
    q = np.array_split(dat, n_dataPoints)
    t = np.array([np.mean(chunk) for chunk in q])
    return t

def seq_weighted_dat(unequal_dat):
    # get a map from run number to arr_sizes
    szes = {run_num: len(dat) for run_num, dat in unequal_dat.items()}
    # sort this map by value
    asc_szes = dict(sorted(szes.items(), key=lambda item: item[1]))
    # create a new array of size max_iter by padding it with nonetype
    asc_arr = np.zeros((n_runs, max_iter))
    j = 0  # new array index
    for r, _ in asc_szes.items():
        use_data = unequal_dat[r]  # fetch data
        asc_arr[j, :len(use_data)] = use_data
        asc_arr[j, len(use_data):] = None  # fill the rest with none
        j += 1

    # conver the array to float (none -> nan)
    asc_mod_arr = np.array(asc_arr, dtype=np.float64)

    # calc mean across runs (exclude nan values)
    mean_dat = np.nanmean(asc_mod_arr, axis=0)
    std_dat = np.nanstd(asc_mod_arr, axis=0)

    # truncate to exclude nan values (along the time dimension)
    trunc_mean = mean_dat[~np.isnan(mean_dat)]
    trunc_std = std_dat[~np.isnan(std_dat)]

    return trunc_mean, trunc_std


def plotFigs():
    shape = (n_runs, max_iter)
    for n, i in enumerate(mem_sizes):
        u_dat = {}
        m_dat = {}
        mem_path = os.path.join(path, f"mem_{i}/checkpoints")

        for r in range(n_runs):
            fpth = os.path.join(mem_path, f"fitness-run{r}.pkl")
            dat = load(fpth)
            u_dat[r] = np.array([v["avg_U"]
                                for k, v in dat.items()])[:max_iter]
            m_dat[r] = np.array([v["avg_M"]
                                 for k, v in dat.items()])[:max_iter]

        u_mean, u_std = seq_weighted_dat(u_dat)
        m_mean, m_std = seq_weighted_dat(m_dat)

        # trim to singularity
        u_mean = chunkProcess(u_mean)
        u_std = chunkProcess(u_std)

        m_mean = chunkProcess(m_mean)
        m_std = chunkProcess(m_std)

        pos = np.argwhere(u_mean == 0)
        if (len(pos)):
            if (pos[0][0] >0):
                trunc_idx = pos[0][0] + 2
                u_mean[trunc_idx:] = None
                u_std[trunc_idx:] = None

                m_mean[trunc_idx:] = None
                m_std[trunc_idx:] = None

        row = n//2
        col = n % 2

        ax[row][col].plot(u_mean, label="unicellular tissue", color="#04030F")
        ax[row][col].fill_between(list(
            range(len(u_mean))), u_mean-mult_fact*u_std, u_mean+mult_fact*u_std, alpha=alpha, color="#04030F")

        if (len(pos)):
            pos_idx = list(pos)[0][0]
            # print(pos_idx)
            ax[row][col].annotate('Unified Agent', xy=(pos_idx, 0.0), xytext=(+45, +25), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                boxstyle='round,pad=0.2', fc='blue', alpha=0.3), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', color='blue'))

        ax[row][col].plot(
            m_mean, label="multicellular tissue", color="#F44708")
        ax[row][col].fill_between(list(
            range(len(m_mean))), m_mean-mult_fact*m_std, m_mean+mult_fact*m_std, alpha=alpha, color="#F44708")

        ax[row][col].set_title(f"Memory = {i}")

        if (len(pos)) and (i > 0):
            pos_idx = list(pos)[0][0]
            # print(pos_idx)
            # u_mean[pos_idx+10:] = None
            # u_std[pos_idx+10:] = None
            ax[row][col].scatter(pos_idx, 0.0, s=20, color='blue')

        if (row+col > 0):
            x_vals = np.arange(0, 60, 10)
            x_labels = [f'{int(i)//10}' for i in x_vals]
            ax[row][col].set_xticks(x_vals, x_labels)
        else:
            x_vals = np.arange(0, 600, 100)
            x_labels = [f'{int(i)//10}' for i in x_vals]
            ax[row][col].set_xticks(x_vals, x_labels)

        ax[row][col].set_xlabel(r'Games Played (x$10^{3}$)')
        # ax[][1].set_xlabel(r'Games Played (x$10^{2}$)')

        ax[row][col].set_ylabel(r'Avg. Fitness')

    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.45)

    plt.savefig(os.path.join(save_path, "fig3.svg"), format='svg')


if __name__ == "__main__":
    plotFigs()
