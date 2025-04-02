#!/usr/bin/env python3

# transfer entropy plot

import os
import pickle
import numpy as np
import seaborn as sns
import multiprocessing
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import animation
from PyIF import te_compute as te
from scipy.interpolate import interp2d
from matplotlib.image import AxesImage
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sns.set()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

cwd_path = os.getcwd()
dat_path = os.path.join(cwd_path, 'data/')
save_path = os.path.join(cwd_path, 'figures/')
n_runs = 5
max_iter = int(15e3)
grid_size = 20

# std mult factor
mult_fact = 1.0
alpha = 0.3

# memory sizes
mem_size = 2
n_lag = 120

def load(fname, dict_load=False):
    with open(fname, "rb") as inp:
        data = pickle.load(inp)
    vals = list(data.values())
    if (dict_load):
        return data
    else:
        for k, v in enumerate(vals):
            # try:
            if k > 1 and np.sum(v) == 0.0:
                fin_val = k
                break
            else:
                fin_val = k
        vals = vals[:fin_val+1]
        return vals

# Remove comment from the code section below to compute transfer entropy from scratch
# we use a pre-cached file here to save time..


###def setup_load_files(r_num):
###    mem_path = os.path.join(dat_path, f"IPDmsLogs/mem_{mem_size}/checkpoints")

###    grid_fname = os.path.join(mem_path, f"bst_idv_grid_strat-run{r_num}.pkl")
###    size_fname = os.path.join(mem_path, f"bst_idv_grid-run{r_num}.pkl")
###    strat_fname = os.path.join(mem_path, f"strat_map_{2}.pkl")

###    grid_data = np.array(load(grid_fname))  # grid data
###    size_data = np.array(load(size_fname))  # size data

###    min_size = np.min([len(grid_data), len(size_data)])
###    #clip
###    grid_data = grid_data[:min_size]
###    size_data = size_data[:min_size]

###    strat_map = load(strat_fname, dict_load=True)  # strat data
###    rev_strat_map = {v: k for k, v in strat_map.items()}  # rev strat map

###    return grid_data, size_data, strat_map, rev_strat_map


###def uq(l):
###    n_dict = {}
###    roll = 0
###    for it in l:
###        uq = []
###        for ch in it:
###            uq.append(ch)
###        temp = list(set(uq))
###        if len(temp):
###            new_t = sorted(temp)
###        else:
###            new_t = temp
###        new_str = ''.join(list(new_t))
###        n_dict[new_str] = roll
###        roll += 1
###    return n_dict


###def reorg(dat_inp, strat_map, remap_dict):
###    dat = dat_inp.copy()
###    for old_str, v in strat_map.items():
###        # str, val
###        tmp = []
###        for ch in old_str:
###            tmp.append(ch)
###        n_temp = list(set(tmp))
###        s_temp = sorted(n_temp)
###        new_k = ''.join(s_temp)
###        new_id = remap_dict[new_k]
###        dat = np.where(dat == v, new_id, dat)
###    return dat


###def compute_single_te(x, y, lag=1):
###    te_val = te.te_compute(x, y, k=1, embedding=lag,
###                           safetyCheck=False, GPU=False)
###    return te_val

#### calc te
###def calc_te(size_data, new_grid_dat, order='a-b', lag=1):
###    htmap = np.zeros((grid_size, grid_size))
###    for row in range(grid_size):
###        for col in range(grid_size):
###            seq_anatomy = size_data[:, row, col]
###            seq_behaviour = new_grid_dat[:, row, col]
###            if (order == 'a-b'):
###                print(len(seq_anatomy), len(seq_behaviour))
###                te_val = compute_single_te(seq_anatomy, seq_behaviour, lag=lag)
###            else:
###                te_val = compute_single_te(seq_behaviour, seq_anatomy, lag=lag)
###            htmap[row][col] = te_val
###            # print(f'Frame: {row}, {col}')

###    m_te = np.mean(htmap)
###    return m_te


###def run(rnum):
###    grid_data, size_data, strat_map, rev_strat_map = setup_load_files(rnum)
###    new_dict = uq(list(strat_map.keys()))
###    roll_ct = 400
###    remap_dict = {}
###    # print(new_dict.keys())
###    for k, v in new_dict.items():
###        remap_dict[k] = roll_ct
###        roll_ct += 1

###    # flatten
###    rev_strat_map = {v: k for k, v in remap_dict.items()}

###    new_grid_dat = np.zeros_like(grid_data)
###    for gh in range(grid_data.shape[0]):
###        mod_dat = reorg(grid_data[gh], strat_map, remap_dict)
###        new_grid_dat[gh] = mod_dat

###    rec_map = np.zeros((n_lag, 2))
###    for l_val in range(1, n_lag):
###        ab_te = calc_te(size_data, new_grid_dat, order='a-b', lag=l_val)
###        rec_map[l_val, 0] = ab_te

###        ba_te = calc_te(size_data, new_grid_dat, order='b-a', lag=l_val)
###        rec_map[l_val, 1] = ba_te

###        if (l_val % 5 == 0):
###            print(f"run: {rnum} | lval: {l_val}/{n_lag}")

###    np.save(os.path.join(dat_path, f"heatmap-run-{rnum}.npy", rec_map))


###def consolidate_saves():
###    save_mat = np.zeros((n_runs, n_lag, 2))
###    for i in range(n_runs):
###        r_map = np.load(os.path.join(dat_path, f'heatmap-run-{i}.npy'))
###        save_mat[i] = r_map.copy()
###    np.save(os.path.join(dat_path, f"heatmap.npy", save_mat))

def plot_fig():
    mult_fact = 1.0
    r_map = np.load(os.path.join(dat_path, f'heatmap.npy'))

    # separate run values for each te direction
    r_ab = r_map[:, :, 0]
    r_ba = r_map[:, :, 1]

    # calc mean and std for both
    r_ab_mean = np.mean(r_ab, axis=0)
    r_ab_std = np.std(r_ab, axis=0)

    r_ba_mean = np.mean(r_ba, axis=0)
    r_ba_std = np.std(r_ba, axis=0)

    r_ab_mean = r_ab_mean[1:]
    r_ab_std = r_ab_std[1:]

    r_ba_mean = r_ba_mean[1:]
    r_ba_std = r_ba_std[1:]

    plt.plot(list(range(1, len(r_ab_mean)+1)), r_ab_mean,
             label=r"Anatomy $\rightarrow$ Behaviour", color='#D64933')
    plt.fill_between(list(range(1, len(r_ab_mean)+1)), r_ab_mean -
                     mult_fact * r_ab_std, r_ab_mean + mult_fact * r_ab_std, color='#D64933', alpha=0.3)
    plt.plot(list(range(1, len(r_ab_mean)+1)), r_ba_mean,
             label=r"Behaviour $\rightarrow$ Anatomy", color='#003844')

    plt.fill_between(list(range(1, len(r_ba_mean)+1)), r_ba_mean -
                     mult_fact * r_ba_std, r_ba_mean + mult_fact * r_ba_std, color='#003844', alpha=0.3)
    plt.xlabel(r"Lag duration ($t_t$)")
    plt.ylabel(r"Transfer Entropy (nats)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fig8.svg'), format='svg')

if __name__ == "__main__":
    # remove comment to compute te from scratch
    # # # pool = multiprocessing.Pool(os.cpu_count())
    # # # pool.map(run, range(n_runs))
    # # # consolidate_saves()
    plot_fig()
