#!/usr/bin/env python3

#heatmap (size X, Y): probability of finding a tissue of size X near a tissue of size Y

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

cwd_path = os.getcwd()
path = os.path.join(cwd_path, 'data/')
save_path = os.path.join(cwd_path, 'figures/')
grid_size = 20
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

# The section within comments is to calculate probabilities from scratch;
# a cached file is used here instead..

# # # def load(fname):
# # #     with open(fname, "rb") as inp:
# # #         data = pickle.load(inp)
# # #     return data

# # # fname = os.path.join(path, f"cluster_size_map{2}.pkl")
# # # dat_f = load(fname)

# # # def process_iter_data(log_data, sim_rec_temp):
# # #     sim_rec = sim_rec_temp.copy()
# # #     for ag_idx, ag_dat in log_data.items():
# # #         from_sz = ag_dat['sz']
# # #         nbr_info = ag_dat['nbr_sz']
# # #         nbr_size_list = list(nbr_info.values())
# # #         count_data = dict(Counter(nbr_size_list))
# # #         for to_size, counts in count_data.items():
# # #             sim_rec[from_sz][to_size] = np.mean(
# # #                 [sim_rec[from_sz][to_size], counts/np.sum(list(count_data.values()))])
# # #     return sim_rec

# # # def run():
# # #     # from_size: to_size (store prob)
# # #     sim_record = np.zeros((401, 401), dtype=np.float32)
# # #     for n, log_dat in dat_f.items():
# # #         print(f"processing iter: {n}/{len(dat_f)}")
# # #         sim_record = process_iter_data(log_dat, sim_record.copy())
# # #     np.save(os.path.join(path, 'sim_rec_prob.npy'), sim_record)


def plot_map(sim_record):
    fig, ax = plt.subplots(figsize= (10, 6))
    new_rec = []
    row_labels = []
    for row in range(sim_record.shape[0]):
        if np.sum(sim_record[row, :]):
            new_rec.append(sim_record[row, :])
            row_labels.append(row)

    fin_rec = []
    col_labels = []
    new_rec = np.array(new_rec)
    for col in range(new_rec.shape[1]):
        if np.sum(new_rec[:, col]):
            fin_rec.append(new_rec[:, col])
            col_labels.append(col)

    fin_rec = np.array(fin_rec)
    fs = 20
    sns.set(font_scale =2)
    ax = sns.heatmap(
            fin_rec, xticklabels=row_labels, yticklabels=col_labels)
    cbar = ax.collections[0].colorbar
    cbar.set_label(r"$\mathbb{P}(X, Y)$", labelpad=20)
    ax.set_xticklabels([min(col_labels)]+ [""]*(len(col_labels)-2) +[max(col_labels)], fontsize=fs)
    ax.set_yticklabels([min(row_labels)]+ [""]*(len(row_labels)-2) +[max(row_labels)], fontsize =fs)
    ax.set_ylabel("Tissue Size (X)", fontsize=fs)
    ax.set_xlabel("Tissue Size (Y)", fontsize=fs)
    ax.tick_params(left=False, bottom=False, right=False, top = False)
    fig.tight_layout()

    plt.savefig(os.path.join(save_path, 'fig6.svg'), format='svg')

if __name__ == "__main__":
    # run() # use to compute probabilities from scratch
    sim_dat = np.load(os.path.join(path, 'sim_rec_prob.npy'))
    plot_map(sim_dat)
