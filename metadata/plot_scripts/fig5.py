#!/usr/bin/env python3

# comparison of ipd-ms vs ipd

# plot1: avg memory vs time (case w/merging and case w/o merging)
# plot2: memory size vs freq (last ts) (case w/ merging and case w/o merging)

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.rcParams['text.usetex'] = True

cwd_path = os.getcwd()
cdm_path = os.path.join(cwd_path, 'data/IPDmsLogs/checkpoints')
twoAct_path = os.path.join(cwd_path, 'data/IPDLogs/checkpoints')
save_path = os.path.join(cwd_path, 'figures/')
n_runs = 5
max_iter = int(50e3)
grid_size = 20
max_memSize = 5

# std mult factor
mult_fact = 0.5
alpha = 0.3

n_dataPoints = 50  # bin and plot x data points

ts_shape = (n_runs, max_iter)

w_merging_memData = {}
wo_merging_memData = {}

mergeBar = np.zeros((n_runs, max_memSize))
twoActBar = np.zeros((n_runs, max_memSize))

mergeBarScore = np.zeros((n_runs, max_memSize))
twoActBarScore = np.zeros((n_runs, max_memSize))


def load(fname):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    # print(len(data))
    return data


def process(data):
    d = np.array(list(data.values()))
    singularity_pos = np.where(d == 0)[0]
    if (len(singularity_pos)):
        d = d[:singularity_pos[0]]
    return d


def processBar(data):
    dist = {lbl: len(scr_list) for lbl, scr_list in data.items()}
    scr = {lbl: np.max(scr_list) if len(scr_list)
           else 0.0 for lbl, scr_list in data.items()}
    return dist, scr


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

    mean_dat = np.nanmean(asc_mod_arr, axis=0)
    std_dat = np.nanstd(asc_mod_arr, axis=0)

    # truncate to exclude nan values
    trunc_mean = mean_dat[~np.isnan(mean_dat)]
    trunc_std = std_dat[~np.isnan(std_dat)]

    return trunc_mean, trunc_std


def gatherData():
    for i in range(n_runs):
        # avg memory data
        mergeMemData = load(os.path.join(
            cdm_path, f"memorySizeList-vs-time-run{i}.pkl"))
        twoActMemData = load(os.path.join(
            twoAct_path, f"memorySizeList-vs-time-run{i}.pkl"))

        mergeDat = process(mergeMemData)
        twoActDat = process(twoActMemData)

        w_merging_memData[i] = mergeDat
        wo_merging_memData[i] = twoActDat

        # last mem data
        mergeBarData = load(os.path.join(cdm_path, f"lastMem-run{i}.pkl"))
        twoActBarData = load(os.path.join(twoAct_path, f"lastMem-run{i}.pkl"))

        # processed last-mem data
        mBdist, mBscr = processBar(mergeBarData)
        twoBdist, twoBscr = processBar(twoActBarData)

        tot_agentsMerge = sum(mBdist.values())
        tot_agentstwoAct = sum(twoBdist.values())
        for lbl, list_len in mBdist.items():
            mergeBar[i][lbl] = list_len/(tot_agentsMerge+1e-8)
            mergeBarScore[i][lbl] = mBscr[lbl]

        for lbl, list_len in twoBdist.items():
            twoActBar[i][lbl] = twoBdist[lbl]/(tot_agentstwoAct+1e-8)
            twoActBarScore[i][lbl] = twoBscr[lbl]


def getStats(dat, norm=0):
    mean = dat.mean(axis=0)
    std = dat.std(axis=0)
    return mean, std


def plotFigs():

    fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(10, 10))

    # get memory-ts data
    w_merging_mean, w_merging_std = seq_weighted_dat(w_merging_memData)
    wo_merging_mean, wo_merging_std = seq_weighted_dat(wo_merging_memData)

    w_merging_mean = chunkProcess(w_merging_mean)
    w_merging_std = chunkProcess(w_merging_std)

    wo_merging_mean = chunkProcess(wo_merging_mean)
    wo_merging_std = chunkProcess(wo_merging_std)

    # plot mem-ts data
    ax1.plot(w_merging_mean, label="IPD-ms", color="#688E26")
    ax1.fill_between(list(range(len(w_merging_mean))), w_merging_mean-mult_fact *
                     w_merging_std, w_merging_mean+mult_fact*w_merging_std, alpha=alpha, color="#688E26")

    ax1.plot(wo_merging_mean, label="IPD", color="#F44708")
    ax1.fill_between(list(range(len(wo_merging_mean))), wo_merging_mean-mult_fact *
                     wo_merging_std, wo_merging_mean+mult_fact*wo_merging_std, alpha=alpha, color="#F44708")
    ax1.set_xlabel(r'Games Played (x$10^3$)')
    ax1.set_ylabel(r"Avg. Memory Length")

    # get bar-dist-data
    bar_merge_mean, bar_merge_std = getStats(mergeBar, norm=0)
    bar_twoAct_mean, bar_twoAct_std = getStats(twoActBar, norm=0)
    bar_merge_mean = bar_merge_mean*100
    bar_merge_std = bar_merge_std*100
    # print(bar_merge_mean, bar_merge_std)
    # print(bar_twoAct_mean, bar_twoAct_std)

    # get scores
    scr_merge_mean, scr_merge_std = getStats(mergeBarScore)
    scr_twoAct_mean, scr_twoAct_std = getStats(twoActBarScore)
    # print(scr_merge_mean, scr_merge_std)
    # print(scr_twoAct_mean, scr_twoAct_std)

    ax2.bar(np.arange(len(bar_merge_mean)), bar_merge_mean,
            label="IPD-ms (Snapshot of Terminal Game)", color="#688E26")
    ax2.errorbar(np.arange(len(bar_merge_mean)), bar_merge_mean,
                 yerr=0.9*mult_fact*bar_merge_std, fmt="o", markersize=4.0, color="blue", capsize=4.0)

    ax2.set_xlabel(r"Memory Size")
    ax2.set_ylabel(r"\%. of Population", color="#688E26")
    ax2.tick_params(axis='y', labelcolor="#688E26")

    ax3 = ax2.twinx()  # instantiate a second Axes that shares the same x-axis
    ax3.grid(None)
    # color = 'tab:red'
    color = "#0A100D"  # 'tab:red'
    # ax3.get_shared_y_axes().join(ax3, ax2)
    # tick_values = np.linspace(0.0, max(scr_merge_mean + scr_merge_std), 10)
    # print(tick_values)

    ax3.set_ylabel(r'Max. Fitness', color=color)
    ax3.plot(scr_merge_mean, color=color)
    ax3.fill_between(list(range(len(scr_merge_mean))), scr_merge_mean-mult_fact *
                     scr_merge_std, scr_merge_mean+mult_fact*scr_merge_std, alpha=alpha, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4.bar(np.arange(len(bar_twoAct_mean)), bar_twoAct_mean*100,
            label="IPD (Snapshot of Terminal Game)", color="#F44708")
    ax4.errorbar(np.arange(len(bar_twoAct_mean)), bar_twoAct_mean*100,
                 yerr=bar_twoAct_std*100, fmt="o", markersize=4.0, color="purple", capsize=4.0)

    tick_values = np.arange(0, 0.3*100, 0.05*100)
    ax4.set_yticks(tick_values)
    ax4.set_xlabel(r"Memory Size")
    ax4.set_ylabel(r"\%. of Population", color="#F44708")
    ax4.tick_params(axis='y', labelcolor="#F44708")

    ax5 = ax4.twinx()  # instantiate a second Axes that shares the same x-axis
    ax5.grid(None)
    tick_values = np.linspace(min(scr_twoAct_mean), max(
        scr_twoAct_mean)+max(scr_twoAct_std), 6)
    ax5.set_yticks(tick_values, labels=[f"{i:.1f}" for i in tick_values])

    color = "#0A100D"  # 'tab:red'
    ax5.set_ylabel(r'Max. Fitness', color=color)
    ax5.plot(scr_twoAct_mean, color=color)
    ax5.fill_between(list(range(len(scr_twoAct_mean))), scr_twoAct_mean-mult_fact *
                     scr_twoAct_std, scr_twoAct_mean+mult_fact*scr_twoAct_std, alpha=alpha, color=color)
    ax5.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.5)

    ax1.legend()
    ax2.legend()
    ax4.legend()

    plt.savefig(os.path.join(save_path, "fig5.svg"), format="svg")

if __name__ == "__main__":
    gatherData()
    plotFigs()
