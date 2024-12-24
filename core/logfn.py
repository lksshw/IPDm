#!/usr/bin/env python3

import os
import scipy
import pickle
import itertools
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import core.helperfunctions as hf


class LogWriter:
    def __init__(self, save_path, hyperParams):
        self.hp = hyperParams
        self.save_path = save_path

        if (self.hp.mode == "fixed"):
            self.memDir = os.path.join(
                self.save_path, f"mem_{self.hp.max_agentMemory}")
            os.makedirs(self.memDir, exist_ok=True)
            self.save_path = self.memDir

        self.checkpoint_path = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.figures_path = os.path.join(self.save_path, "figures")
        os.makedirs(self.figures_path, exist_ok=True)

        self.init_data()

    def init_data(self):
        # log data (each ts)
        # store avg cluster size at each ts
        self.clusterSizeData = {k: 0.0 for k in range(self.hp.max_iter)}
        self.cooperability = {k: {"avg_U": 0.0, "avg_M": 0.0}
                              for k in range(self.hp.max_iter)}  # store at each ts

        # store avg memory size at each ts
        self.memorySizeData = {k: 0.0 for k in range(self.hp.max_iter)}
        self.lastMemScores = {k: [] for k in range(self.hp.max_agentMemory+1)}

        # unicellular and multicellular fitness
        self.fitness_scores = {k: {"avg_U": 0.0, "avg_M": 0.0}
                               for k in range(self.hp.max_iter)}  # store at each ts

        # unicellular and multicellular fitness of the best and worst individual
        self.idv_fitness_scores = {sz: [] for sz in range(
            int(self.hp.board_size**2)+1)}  # store at each ts

        # tft data log
        self.tft_agents_score = []
        self.other_agents_score = []

        # leader/follower data
        self.lf_fitness = {k: {"leader": 0.0, "follower": 0.0}
                           for k in range(self.hp.max_iter)}
        self.lf_memSize = {k: {"leader": 0, "follower": 0}
                           for k in range(self.hp.max_iter)}

        self.policy_rec = 0

        # target cell track
        self.target_fitness = []
        self.other_fitness = []

        # track correlation b/w fitness and tissue size
        self.corr_record = {k: 0.0 for k in range(self.hp.max_iter)}

        # best idv matrix
        self.best_idv_matrix = {k: np.zeros(
            (self.hp.board_size, self.hp.board_size)) for k in range(self.hp.max_iter)}

        # best idv matrix (fitness based)
        self.best_idv_matrix_fn = {k: np.zeros(
            (self.hp.board_size, self.hp.board_size)) for k in range(self.hp.max_iter)}

        self.best_idv_matrix_strat = {k: np.zeros(
            (self.hp.board_size, self.hp.board_size)) for k in range(self.hp.max_iter)}

        self.visit_count = {}

        # record agent data (idx, act_played(t), fitness(t))
        self.agent_data = {}

        #strat map
        self.strat_to_index_map = {}
        self.strat_count = 0

    def init_stateVisitCounter(self, state_map):
        # state visitation frequency
        self.visit_count = {state: 0.0 for state, v in state_map.items()}

    def update_visitCount(self, my_state, opp_state):
        if (my_state != ''):
            self.visit_count[my_state] += 0.5
        if (opp_state != ''):
            self.visit_count[opp_state] += 0.5

    def record_LeaderFollowerData(self, agent_list, bs, curr_iter):
        pop_fitnessList = []
        pop_memLen = []
        for ag_idx in agent_list:
            pop_fitnessList.append(bs.tree[ag_idx].agent.get_score())
            pop_memLen.append(bs.tree[ag_idx].agent.memory_length)

        max_idx = np.argmax(pop_fitnessList)
        self.lf_fitness[curr_iter]["leader"] = pop_fitnessList[max_idx]
        self.lf_memSize[curr_iter]["leader"] = pop_memLen[max_idx]

        # remove max fitness agent
        pop_fitnessList.pop(max_idx)
        pop_memLen.pop(max_idx)

        # get avg fitness of the rest
        self.lf_fitness[curr_iter]["follower"] = np.mean(pop_fitnessList)
        self.lf_memSize[curr_iter]["follower"] = np.mean(pop_memLen)

    def record_policy(self, agent_list, bs, curr_iter):
        scores = []
        for ag_idx in agent_list:
            scores.append(bs.tree[ag_idx].agent.get_score())

        bst_idx = np.argmax(scores)
        self.policy_rec = bs.tree[agent_list[bst_idx]].agent.policy.qTable

    def record_fitness_size_correlation(self, agent_list, bs, curr_iter):
        sizes = []
        scores = []
        # for ag_idx in agent_list:
        #     me = bs.tree[ag_idx]
        #     cnt = len(hf.get_root(ag_idx, bs))
        #     payoff = me.agent.get_score()
        #     sizes.append(cnt)
        #     scores.append(payoff)

        # # correlation between the 1st array(0) and the second(1)
        # self.corr_record[curr_iter] = np.corrcoef(sizes, scores)[0][1]
        # val = stats.pearsonr(sizes, scores)
        # if (np.isnan(val[0])):
        #     pass
        # else:
        #     print(val)

    def record_cooperability(self, agent_list, bs, curr_iter):
        uCoop = 0  # unicellular cooperability
        mCoop = 0  # multicellular cooperability

        uCellCount = 0
        mCellCount = 0
        for ag_idx in agent_list:
            me = bs.tree[ag_idx]
            cnt = len(hf.get_root(ag_idx, bs))
            if (cnt > 1):  # multicellular
                mCellCount += 1
                try:
                    if (me.agent.memory[-1] == 'C'):
                        mCoop += 1
                except IndexError:
                    continue

            else:  # unicellular
                uCellCount += 1
                try:
                    if (me.agent.memory[-1] == 'C'):
                        uCoop += 1
                except:
                    continue

        self.cooperability[curr_iter]["avg_U"] = uCoop/(uCellCount+1e-08)
        self.cooperability[curr_iter]["avg_M"] = mCoop/(mCellCount+1e-08)

    def record_idv_fitness(self, agent_list, bs, curr_iter):
        # get scores of different sized clusters

        size_scores = {k: [] for k in range(int(self.hp.board_size**2)+1)}
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs))  # cluster size
            score = bs.tree[ag_idx].agent.get_score()
            size_scores[cnt].append(score)

        size_scores_mod = {k: np.max(v) if len(
            v) else 0.0 for k, v in size_scores.items()}
        for sze, val in size_scores_mod.items():
            self.idv_fitness_scores[sze].append(val)

    def record_fitness(self, agent_list, bs, curr_iter):
        # get scores of unicellular and multicelluar clusters
        uCell_scores = []
        mCell_scores = []
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs))  # cluster size
            score = bs.tree[ag_idx].agent.get_score()
            if (cnt > 1):  # multicellular
                mCell_scores.append(score)
            else:
                uCell_scores.append(score)
        if (len(uCell_scores)):
            self.fitness_scores[curr_iter]["avg_U"] = np.mean(uCell_scores)

        if (len(mCell_scores)):
            self.fitness_scores[curr_iter]["avg_M"] = np.mean(mCell_scores)

    def record_clusterInfo(self, agent_list, bs, curr_iter):
        # get the cluster size of each agent
        # get max score for each cluster
        cluster_size = []
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs))
            curr_score = bs.tree[ag_idx].agent.get_score()
            # self.cluster_score_map[cnt].append(curr_score) #store scores of a particular cluster size
            cluster_size.append(cnt)
        unique_sizes = set(cluster_size)
        self.clusterSizeData[curr_iter] = np.mean(list(unique_sizes))

    def record_memoryInfo(self, agent_list, bs, curr_iter):
        # get a distribution of memory_sizes
        mem_sizes = []
        for ag_idx in agent_list:
            memLen = bs.tree[ag_idx].agent.memory_length
            mem_sizes.append(memLen)
        self.memorySizeData[curr_iter] = np.mean(mem_sizes)

    def record_lastMem(self, agent_list, bs):
        self.lastMemScores = {k: [] for k in range(self.hp.max_agentMemory+1)}
        for ag_idx in agent_list:
            memLen = bs.tree[ag_idx].agent.memory_length
            score = bs.tree[ag_idx].agent.get_score()
            self.lastMemScores[memLen].append(score)

    def create_grid_viz(self, bs, flag):
        stained_grid = np.zeros((self.hp.board_size, self.hp.board_size))

        for i in range(self.hp.board_size**2):
            leaf_id = hf.get_leaf(i, bs)  # get supergroup agent
            leaf_ag_size = len(hf.get_root(leaf_id, bs))  # get its size
            row, col = bs._getBoardPos(i)

            #get fitness
            ftn_idv = bs.tree[leaf_id].agent.get_score()
            #get strat
            idv_strat = bs.tree[leaf_id].agent.memory[-3:]

            # stain row,col with its tissue size
            if flag == 'fn':
                stained_grid[row][col] = ftn_idv
            elif flag == 'sz':
                stained_grid[row][col] = leaf_ag_size
            elif flag == 'strat':
                try:
                    stained_grid[row][col] = self.strat_to_index_map[idv_strat]
                except KeyError:
                    self.strat_to_index_map[idv_strat] = self.strat_count
                    stained_grid[row][col] = self.strat_to_index_map[idv_strat]
                    self.strat_count += 1

        return stained_grid

    def store_grid(self, agent_list, bs, curr_iter):
        ag_scores = []
        for ag_idx in agent_list:
            ag_scores.append(bs.tree[ag_idx].agent.get_score())

        best_idx = np.argmax(ag_scores)
        best_ag_idx = agent_list[best_idx]

        self.best_idv_matrix[curr_iter] = self.create_grid_viz(bs, flag = 'sz')

        self.best_idv_matrix_fn[curr_iter] = self.create_grid_viz(bs, flag='fn')

        self.best_idv_matrix_strat[curr_iter] = self.create_grid_viz(bs, flag='strat')

    #def gather_agent_data(self, agent_list, bs, curr_iter):
    #    for ag_idx in agent_list:
    #        if ag_idx not in self.agent_data.keys():
    #            #then make sure you create a sub-dict entry of size max_iter
    #            self.agent_data[ag_idx] = {ky: {} for ky in range(self.hp.max_iter)}

    #        size_cnt = len(hf.get_root(ag_idx, bs))
    #        if (bs.tree[ag_idx].agent.acts_played):
    #            lst_act = bs.tree[ag_idx].agent.memory[-1]
    #        else:
    #            lst_act = ""
    #        ftn = bs.tree[ag_idx].agent.get_score()

    #        self.agent_data[ag_idx][curr_iter] = {"size": size_cnt, "act": lst_act, "ftn": ftn}

    def record_single_gameplay(self, ag1, ag2, bs, curr_iter):

        for i in [ag1, ag2]:
            if i not in self.agent_data.keys():
               #then make sure you create a sub-dict entry of size max_iter
               self.agent_data[i] = {ky: {} for ky in range(self.hp.max_iter)}

            ag_cnt = len(hf.get_root(i, bs))
            ag_fn = bs.tree[i].agent.get_score()
            ag_act = bs.tree[i].agent.memory[-1]
            self.agent_data[i][curr_iter] = {"sz": ag_cnt, "fn": ag_fn, "act": ag_act}

    def gather_data(self, agent_list, bs, curr_iter):
        self.record_clusterInfo(agent_list, bs, curr_iter)
        self.record_cooperability(agent_list, bs, curr_iter)
        self.record_memoryInfo(agent_list, bs, curr_iter)
        self.record_lastMem(agent_list, bs)

        # fitness data
        self.record_fitness(agent_list, bs, curr_iter)

        # leader/follower cell trace
        self.record_LeaderFollowerData(agent_list, bs, curr_iter)

        # record best agent policies
        self.record_policy(agent_list, bs, curr_iter)

        # record a map of tissue_size -> max_fitness over time
        self.record_idv_fitness(agent_list, bs, curr_iter)

        # store best agent grid
        self.store_grid(agent_list, bs, curr_iter)

        # size-fitness corr
        self.record_fitness_size_correlation(agent_list, bs, curr_iter)

        # agent data
        # self.gather_agent_data(agent_list, bs, curr_iter)

    def gather_deviantData(self, agent_list, bs, curr_iter, ag_idx):
        target_agentRecord = 0.0
        avg_fitness = []
        for ag in agent_list:
            if ag == ag_idx:
                target_agentRecord = bs.tree[ag].agent.get_score()
            else:
                avg_fitness.append(bs.tree[ag].agent.get_score())

        self.target_fitness.append(target_agentRecord)
        self.other_fitness.append(np.mean(avg_fitness))

    def gatherTftData(self, agent_list, bs):
        # check the action each agent would play against its opponent, verify if it is equal to the opponent's last action

        tft_agents_score = []
        other_agents_score = []

        for ag_idx in agent_list:
            me = bs.tree[ag_idx]

            tft_score = 0
            for op_idx in me.neighbors:
                opp = bs.tree[op_idx]
                smallest_memLen = hf.smallest(
                    me.agent.memory_length, opp.agent.memory_length)
                my_state = hf.getState_from(
                    me.agent.memory, opp.agent.memory, smallest_memLen)
                my_action = me.agent.act_given(my_state)
                tft_score += hf.tft_satisfied(my_action, opp.agent.memory)

            # if i'm tft with all my neighbors
            if ((tft_score == len(me.neighbors)) and len(me.neighbors)):
                tft_agents_score.append(me.agent.get_score())

            elif (len(me.neighbors)):
                other_agents_score.append(opp.agent.get_score())

            else:
                # if you arent' tft, then you belong to the category of "other agents"
                other_agents_score.append(0.0)

        self.tft_agents_score.append(tft_agents_score)
        self.other_agents_score.append(other_agents_score)

    def save_file(self, fname, file):
        with open(fname, "wb") as F:
            pickle.dump(file, F, pickle.HIGHEST_PROTOCOL)

    def load_file(self, fname):
        with open(fname, "rb") as inp:
            data = pickle.load(inp)
        return data

    def save_data(self, bs, run_num, iter):
        # cluster
        self.save_file(os.path.join(self.checkpoint_path,
                       f"clustersizeList-vs-time-run{run_num}.pkl"), self.clusterSizeData)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"cooperability-vs-time_merge-run{run_num}.pkl"), self.cooperability)

        # memory
        self.save_file(os.path.join(self.checkpoint_path,
                       f"memorySizeList-vs-time-run{run_num}.pkl"), self.memorySizeData)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"lastMem-run{run_num}.pkl"), self.lastMemScores)

        # save boardstate
        # self.save_file(os.path.join(self.checkpoint_path, f"boardstate-run{run_num}.pkl"), bs)

        # save tft data
        # self.save_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"), self.tft_agents)
        # self.save_file(os.path.join(self.checkpoint_path, "other_agents.pkl"), self.other_agents)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"fitness-run{run_num}.pkl"), self.fitness_scores)

        self.save_file(os.path.join(self.checkpoint_path,
                       f"tft_agents_score-run{run_num}.pkl"), self.tft_agents_score)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"other_agents_score-run{run_num}.pkl"), self.other_agents_score)

        # save metadata
        meta_data = {"run_num": run_num, "iter_num": iter}
        self.save_file(os.path.join(self.checkpoint_path,
                       f"meta_data-run{run_num}.pkl"), meta_data)

        # leader-follower data
        self.save_file(os.path.join(self.checkpoint_path,
                       f"lwfitness_data-run{run_num}.pkl"), self.lf_fitness)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"lwmem_data-run{run_num}.pkl"), self.lf_memSize)

        # policy data
        self.save_file(os.path.join(self.checkpoint_path,
                       f"policy_data-run{run_num}.pkl"), self.policy_rec)

        # target cell data
        self.save_file(os.path.join(self.checkpoint_path,
                       f"targetCellData-run{run_num}.pkl"), self.target_fitness)
        self.save_file(os.path.join(self.checkpoint_path,
                       f"otherCellData-run{run_num}.pkl"), self.other_fitness)

        # state visitation data
        self.save_file(os.path.join(self.checkpoint_path,
                       f"stateVisit-run{run_num}.pkl"), self.visit_count)

        # idv fitness

        self.save_file(os.path.join(self.checkpoint_path,
                       f"idv_fitness_scores-run{run_num}.pkl"), self.idv_fitness_scores)

        # best grid state
        self.save_file(os.path.join(self.checkpoint_path,
                       f"bst_idv_grid-run{run_num}.pkl"), self.best_idv_matrix)

        self.save_file(os.path.join(self.checkpoint_path,
                       f"bst_idv_grid_fn-run{run_num}.pkl"), self.best_idv_matrix_fn)

        self.save_file(os.path.join(self.checkpoint_path,
                       f"bst_idv_grid_strat-run{run_num}.pkl"), self.best_idv_matrix_strat)

        # size-fitness correlation
        self.save_file(os.path.join(self.checkpoint_path,
                       f"size_score_corr-run{run_num}.pkl"), self.corr_record)

        # save agent data
        self.save_file(os.path.join(self.checkpoint_path,
                       f"agent_data-run{run_num}.pkl"), self.agent_data)

        #save strat map
        self.save_file(os.path.join(self.checkpoint_path,
                       f"strat_map_{run_num}.pkl"), self.strat_to_index_map)

    def load_checkpoint(self, run_num):
        # meta_data
        try:
            meta_data = self.load_file(os.path.join(
                self.checkpoint_path, f"meta_data-run{run_num}.pkl"))
        except:
            return -1, -1

        iter_num = meta_data["iter_num"]

        # boardstate
        # bs = self.load_file(os.path.join(self.checkpoint_path, f"boardstate-run{run_num}.pkl"))

        # cluster data
        self.clusterSizeData = self.load_file(os.path.join(
            self.checkpoint_path, f"clustersizeList-vs-time-run{run_num}.pkl"))
        self.cooperability = self.load_file(os.path.join(
            self.checkpoint_path, f"cooperability-vs-time_merge-run{run_num}.pkl"))

        # memory data
        self.memorySizeData = self.load_file(os.path.join(
            self.checkpoint_path, f"memorySizeList-vs-time-run{run_num}.pkl"))
        self.lastMemScores = self.load_file(os.path.join(
            self.checkpoint_path, f"lastMem-run{run_num}.pkl"))

        # fitness
        self.fitness_scores = self.load_file(os.path.join(
            self.checkpoint_path, f"fitness-run{run_num}.pkl"))

        # tft data
        # self.tft_agents = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        # self.other_agents = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))
        self.tft_agents_score = self.load_file(os.path.join(
            self.checkpoint_path, f"tft_agents_score-run{run_num}.pkl"))
        self.other_agents_score = self.load_file(os.path.join(
            self.checkpoint_path, f"other_agents_score-run{run_num}.pkl"))

        # leader-follower data
        self.lf_fitness = self.load_file(os.path.join(
            self.checkpoint_path, f"lwfitness_data-run{run_num}.pkl"))
        self.lf_memSize = self.load_file(os.path.join(
            self.checkpoint_path, f"lwmem_data-run{run_num}.pkl"))

        # policy data
        self.policy_rec = self.load_file(os.path.join(
            self.checkpoint_path, f"policy_data-run{run_num}.pkl"))

        # target data
        self.target_fitness = self.load_file(os.path.join(
            self.checkpoint_path, f"targetCellData-run{run_num}.pkl"))
        self.other_fitness = self.load_file(os.path.join(
            self.checkpoint_path, f"otherCellData-run{run_num}.pkl"))

        # state visit count
        self.visit_count = self.load_file(os.path.join(
            self.checkpoint_path, f"stateVisit-run{run_num}.pkl"))

        # idv fitness

        self.idv_fitness_scores = self.load_file(os.path.join(
            self.checkpoint_path, f"idv_fitness_scores-run{run_num}.pkl"))

        self.best_idv_matrix = self.load_file(os.path.join(
            self.checkpoint_path, f"bst_idv_grid-run{run_num}.pkl"))

        self.best_idv_matrix_fn = self.load_file(os.path.join(
            self.checkpoint_path, f"bst_idv_grid_fn-run{run_num}.pkl"))

        self.best_idv_matrix_strat = self.load_file(os.path.join(
            self.checkpoint_path, f"bst_idv_grid_strat-run{run_num}.pkl"))

        self.corr_record = self.load_file(os.path.join(
            self.checkpoint_path, f"size_score_corr-run{run_num}.pkl"))

        self.agent_data = self.load_file(os.path.join(self.checkpoint_path, f"agent_data-run{run_num}.pkl"))

        return iter_num, bs

    def binned_avg(self, lst, chunk_size):
        # chunk a list into multiple sub-lists of chunk-size and
        # get the avg of each chunk

        lst_split = np.array_split(lst, chunk_size)
        new_lst = [np.mean(chunk) for chunk in lst_split]

        return new_lst

    def plot_clusterData(self, chunk_size):
        # plot setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        def addlabels(x, y):
            for i in range(len(x)):
                avg_score = np.mean(cluster_score_map[x[i]])
                rounded_score = np.round(avg_score, 2)
                # show average score of each cluster on top of its respective bar
                ax1.text(
                    i, y[i]+10, f"avg-score: {rounded_score}", color='red', ha='center')

        # load data
        bar_info = self.load_file(os.path.join(
            self.checkpoint_path, "clustersizeList-vs-time.pkl"))
        cooperability = self.load_file(os.path.join(
            self.checkpoint_path, "cooperability-vs-time_merge.pkl"))
        cluster_score_map = self.load_file(os.path.join(
            self.checkpoint_path, "cluster_score_map.pkl"))

        # plot
        bar_info = bar_info[-1]
        labels, counts = np.unique(bar_info, return_counts=True)
        ax1.bar(list(range(len(labels))), counts, align='center')
        addlabels(labels, counts)
        ax1.set_xticks(list(range(len(labels))), labels)
        ax1.set_xlabel("cluster size")
        ax1.set_ylabel("frequency")

        coop_binned = self.binned_avg(cooperability, chunk_size)
        ax2.plot(coop_binned, color='green', label='cooperability')
        ax2.set_xlabel("time x1e2")
        ax2.set_ylabel("No. of agents cooperating")

        ax1.legend()
        ax2.legend()
        plt.savefig(os.path.join(self.figures_path,
                    "cluster_plot.png"), dpi=300)
        plt.show()

    def plot_memoryData(self):
        # plot setup
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

        def addlabels(x, y):
            for i in range(len(x)):
                avg_score = np.mean(memory_score_map[x[i]])
                rounded_score = np.round(avg_score, 2)
                # show average score of each memory_kind on top of its respective bar
                ax1.text(
                    i, y[i]+10, f"avg-score: {rounded_score}", color='red', ha='center')

        # load data
        bar_info = self.load_file(os.path.join(
            self.checkpoint_path, "memorySizeList-vs-time.pkl"))
        memory_score_map = self.load_file(os.path.join(
            self.checkpoint_path, "memory_score_map.pkl"))

        # plot
        bar_info = bar_info[-1]  # data from the last time step
        labels, counts = np.unique(bar_info, return_counts=True)
        ax1.bar(list(range(len(labels))), counts, align='center')
        addlabels(labels, counts)
        ax1.set_xticks(list(range(len(labels))), labels)
        ax1.set_xlabel("Memory length")
        ax1.set_ylabel("Frequency")

        ax1.legend()
        plt.savefig(os.path.join(self.figures_path,
                    "memory_size_score.png"), dpi=300)
        plt.show()

    def plot_tftData(self):
        # plot setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # load data
        # tft_dat = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        # oth_dat = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))

        tft_scr_dat = self.load_file(os.path.join(
            self.checkpoint_path, "tft_agents_score.pkl"))
        oth_scr_dat = self.load_file(os.path.join(
            self.checkpoint_path, "other_agents_score.pkl"))

        # plot
        tft_info = [len(l)/(len(l)+len(oth_scr_dat[idx]))
                    for idx, l in enumerate(tft_scr_dat)]  # vs timestep
        oth_info = [len(j)/(len(j)+len(tft_scr_dat[idx]))
                    for idx, j in enumerate(oth_scr_dat)]  # vs timestep

        tft_score_info = [np.sum(l) if len(l) else 0.0 for l in tft_scr_dat]
        # tft_scoreMax_info = [np.max(l) if len(l) else 0.0 for l in tft_scr_dat]
        # tft_scoreMin_info = [np.min(l) if len(l) else 0.0 for l in tft_scr_dat]

        oth_score_info = [np.sum(k) if len(k) else 0.0 for k in oth_scr_dat]
        # oth_scoreMax_info = [np.max(k) if len(k) else 0.0 for k in oth_scr_dat]
        # oth_scoreMin_info = [np.min(k) if len(k) else 0.0 for k in oth_scr_dat]

        ax1.plot(tft_info, color='red', label='tft agents')
        ax1.plot(oth_info, color='blue', label='other agents')
        ax1.set_xlabel("N. Games")
        ax1.set_ylabel("Proportion of Pop.")

        ax2.plot(tft_score_info, color='red', label='tft score')
        ax2.plot(oth_score_info, color='blue', label='oth score')
        ax2.set_xlabel("N. Games")
        ax2.set_ylabel("Population Score")

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "tft.png"), dpi=300)
        plt.show()


if __name__ == "__main__":
    path = "/Users/niwhskal/IPD/logs"
    lw = LogWriter(path)
