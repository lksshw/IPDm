#!/usr/bin/env python3

import os
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import core.helperfunctions as hf

class LogWriter:
    def __init__(self, save_path, hyperParams):
        self.save_path = save_path
        self.checkpoint_path = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.figures_path = os.path.join(self.save_path, "figures")
        os.makedirs(self.figures_path, exist_ok=True)

        self.hp = hyperParams
        self.init_data()


    def init_data(self):
        #log data (each ts)
        self.clusterSizeData = {k: 0.0 for k in range(self.hp.max_iter)} #store avg cluster size at each ts
        self.cooperability = {k: {"avg_U": 0.0, "avg_M": 0.0} for k in range(self.hp.max_iter)} #store at each ts

        self.memorySizeData = {k: 0.0 for k in range(self.hp.max_iter)} #store avg memory size at each ts
        self.lastMemScores = {k: [] for k in range(self.hp.max_agentMemory+1)}

        #unicellular and multicellular fitness
        self.fitness_scores = {k: {"avg_U": 0.0, "avg_M": 0.0} for k in range(self.hp.max_iter)} #store at each ts

        #tft data log
        self.tft_agents_score = []
        self.other_agents_score = []

    def record_cooperability(self, agent_list, bs, curr_iter):
        uCoop = 0 #unicellular cooperability
        mCoop = 0 #multicellular cooperability
        for ag_idx in agent_list:
            me = bs.tree[ag_idx]
            cnt = len(hf.get_root(ag_idx, bs))
            try:
                if(me.agent.memory[-1] == "C"):
                    if (cnt >1): #multicellular
                        mCoop += 1
                    else:
                        uCoop += 1
            except IndexError:
                pass
        self.cooperability[curr_iter]["avg_U"] = uCoop/len(agent_list)
        self.cooperability[curr_iter]["avg_M"] = mCoop/len(agent_list)

    def record_fitness(self, agent_list, bs, curr_iter):
        #get scores of unicellular and multicelluar clusters
        uCell_scores = []
        mCell_scores = []
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs)) #cluster size
            score = bs.tree[ag_idx].agent.get_score()
            if (cnt >1): #multicellular
                mCell_scores.append(score)
            else:
                uCell_scores.append(score)
        self.fitness_scores[curr_iter]["avg_U"] = np.mean(uCell_scores)
        self.fitness_scores[curr_iter]["avg_M"] = np.mean(mCell_scores)

    def record_clusterInfo(self, agent_list, bs, curr_iter):
        #get the cluster size of each agent
        #get max score for each cluster
        cluster_size = []
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs))
            curr_score = bs.tree[ag_idx].agent.get_score()
            # self.cluster_score_map[cnt].append(curr_score) #store scores of a particular cluster size
            cluster_size.append(cnt)
        unique_sizes = set(cluster_size)
        self.clusterSizeData[curr_iter] = np.mean(list(unique_sizes))

    def record_memoryInfo(self, agent_list, bs, curr_iter):
        #get a distribution of memory_sizes
        mem_sizes = []
        for ag_idx in agent_list:
            memLen = bs.tree[ag_idx].agent.memory_length
            mem_sizes.append(memLen)
        unique_sizes = set(mem_sizes)
        self.memorySizeData[curr_iter] = np.mean(list(unique_sizes)) #avg_mem size at each ts

    def record_lastMem(self, agent_list, bs):
        for ag_idx in agent_list:
            memLen = bs.tree[ag_idx].agent.memory_length
            score = bs.tree[ag_idx].agent.get_score()
            self.lastMemScores[memLen].append(score)

    def gather_data(self, agent_list, bs, curr_iter):
        self.record_clusterInfo(agent_list, bs, curr_iter)
        self.record_cooperability(agent_list, bs, curr_iter)
        self.record_memoryInfo(agent_list, bs, curr_iter)
        if (curr_iter == self.hp.max_iter-1):
            self.record_lastMem(agent_list, bs)

        #fitness data
        self.record_fitness(agent_list, bs, curr_iter)

    def gatherTftData(self, agent_list, bs):
        #check the action each agent would play against its opponent, verify if it is equal to the opponent's last action

        tft_agents_score = []
        other_agents_score = []

        for ag_idx in agent_list:
            me = bs.tree[ag_idx]

            tft_score = 0
            for op_idx in me.neighbors:
                opp = bs.tree[op_idx]
                smallest_memLen = hf.smallest(me.agent.memory_length, opp.agent.memory_length)
                my_state = hf.getState_from(me.agent.memory, opp.agent.memory, smallest_memLen)
                my_action = me.agent.act_given(my_state)
                tft_score += hf.tft_satisfied(my_action, opp.agent.memory)

            if (tft_score == len(me.neighbors)): #if i'm tft with all my neighbors
                tft_agents_score.append(me.agent.get_score())

            else:
                other_agents_score.append(opp.agent.get_score())

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
        #cluster
        self.save_file(os.path.join(self.checkpoint_path, f"clustersizeList-vs-time-run{run_num}.pkl"), self.clusterSizeData)
        self.save_file(os.path.join(self.checkpoint_path, f"cooperability-vs-time_merge-run{run_num}.pkl"), self.cooperability)

        #memory
        self.save_file(os.path.join(self.checkpoint_path, f"memorySizeList-vs-time-run{run_num}.pkl"), self.memorySizeData)
        self.save_file(os.path.join(self.checkpoint_path, f"lastMem-run{run_num}.pkl"), self.lastMemScores)

        #save boardstate
        self.save_file(os.path.join(self.checkpoint_path, f"boardstate-run{run_num}.pkl"), bs)

        #save tft data
        # self.save_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"), self.tft_agents)
        # self.save_file(os.path.join(self.checkpoint_path, "other_agents.pkl"), self.other_agents)
        self.save_file(os.path.join(self.checkpoint_path, f"fitness-run{run_num}.pkl"), self.fitness_scores)

        self.save_file(os.path.join(self.checkpoint_path, f"tft_agents_score-run{run_num}.pkl"), self.tft_agents_score)
        self.save_file(os.path.join(self.checkpoint_path, f"other_agents_score-run{run_num}.pkl"), self.other_agents_score)

        #save metadata
        meta_data = {"run_num": run_num, "iter_num": iter}
        self.save_file(os.path.join(self.checkpoint_path, f"meta_data-run{run_num}.pkl"), meta_data)

    def load_checkpoint(self, run_num):
        #meta_data
        try:
            meta_data = self.load_file(os.path.join(self.checkpoint_path, f"meta_data-run{run_num}.pkl"))
        except:
            return -1, -1

        iter_num = meta_data["iter_num"]

        #boardstate
        bs = self.load_file(os.path.join(self.checkpoint_path, f"boardstate-run{run_num}.pkl"))

        #cluster data
        self.clusterSizeData = self.load_file(os.path.join(self.checkpoint_path, f"clustersizeList-vs-time-run{run_num}.pkl"))
        self.cooperability = self.load_file(os.path.join(self.checkpoint_path, f"cooperability-vs-time_merge-run{run_num}.pkl"))

        #memory data
        self.memorySizeData = self.load_file(os.path.join(self.checkpoint_path, f"memorySizeList-vs-time-run{run_num}.pkl"))
        self.lastMemScores = self.load_file(os.path.join(self.checkpoint_path, f"lastMem-run{run_num}.pkl"))

        #fitness
        self.fitness_scores = self.load_file(os.path.join(self.checkpoint_path, f"fitness-run{run_num}.pkl"))

        #tft data
        # self.tft_agents = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        # self.other_agents = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))

        self.tft_agents_score = self.load_file(os.path.join(self.checkpoint_path, f"tft_agents_score-run{run_num}.pkl"))
        self.other_agents_score = self.load_file(os.path.join(self.checkpoint_path, f"other_agents_score-run{run_num}.pkl"))

        return iter_num, bs

    def binned_avg(self, lst, chunk_size):
        #chunk a list into multiple sub-lists of chunk-size and
        #get the avg of each chunk

        lst_split = np.array_split(lst, chunk_size)
        new_lst = [np.mean(chunk) for chunk in lst_split]

        return new_lst


    def plot_clusterData(self, chunk_size):
        #plot setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        def addlabels(x,y):
            for i in range(len(x)):
                avg_score = np.mean(cluster_score_map[x[i]])
                rounded_score = np.round(avg_score, 2)
                ax1.text(i, y[i]+10, f"avg-score: {rounded_score}", color='red', ha = 'center') #show average score of each cluster on top of its respective bar

        #load data
        bar_info = self.load_file(os.path.join(self.checkpoint_path, "clustersizeList-vs-time.pkl"))
        cooperability = self.load_file(os.path.join(self.checkpoint_path, "cooperability-vs-time_merge.pkl"))
        cluster_score_map = self.load_file(os.path.join(self.checkpoint_path, "cluster_score_map.pkl"))

        #plot
        bar_info = bar_info[-1]
        labels, counts = np.unique(bar_info, return_counts=True)
        ax1.bar(list(range(len(labels))), counts, align='center')
        addlabels(labels, counts)
        ax1.set_xticks(list(range(len(labels))), labels)
        ax1.set_xlabel("cluster size")
        ax1.set_ylabel("frequency")

        coop_binned = self.binned_avg(cooperability, chunk_size)
        ax2.plot(coop_binned, color = 'green', label = 'cooperability')
        ax2.set_xlabel("time x1e2")
        ax2.set_ylabel("No. of agents cooperating")

        ax1.legend()
        ax2.legend()
        plt.savefig(os.path.join(self.figures_path, "cluster_plot.png"), dpi = 300)
        plt.show()

    def plot_memoryData(self):
        #plot setup
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
        def addlabels(x,y):
            for i in range(len(x)):
                avg_score = np.mean(memory_score_map[x[i]])
                rounded_score = np.round(avg_score, 2)
                ax1.text(i, y[i]+10, f"avg-score: {rounded_score}", color='red', ha = 'center') #show average score of each memory_kind on top of its respective bar

        #load data
        bar_info = self.load_file(os.path.join(self.checkpoint_path, "memorySizeList-vs-time.pkl"))
        memory_score_map = self.load_file(os.path.join(self.checkpoint_path, "memory_score_map.pkl"))

        #plot
        bar_info = bar_info[-1] #data from the last time step
        labels, counts = np.unique(bar_info, return_counts=True)
        ax1.bar(list(range(len(labels))), counts, align='center')
        addlabels(labels, counts)
        ax1.set_xticks(list(range(len(labels))), labels)
        ax1.set_xlabel("Memory length")
        ax1.set_ylabel("Frequency")

        ax1.legend()
        plt.savefig(os.path.join(self.figures_path, "memory_size_score.png"), dpi = 300)
        plt.show()


    def plot_tftData(self):
        #plot setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        #load data
        # tft_dat = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        # oth_dat = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))

        tft_scr_dat = self.load_file(os.path.join(self.checkpoint_path, "tft_agents_score.pkl"))
        oth_scr_dat = self.load_file(os.path.join(self.checkpoint_path, "other_agents_score.pkl"))

        #plot
        tft_info = [len(l)/(len(l)+len(oth_scr_dat[idx])) for idx, l in enumerate(tft_scr_dat)] #vs timestep
        oth_info = [len(j)/(len(j)+len(tft_scr_dat[idx])) for idx, j in enumerate(oth_scr_dat)] #vs timestep

        tft_score_info = [np.sum(l) if len(l) else 0.0 for l in tft_scr_dat]
        # tft_scoreMax_info = [np.max(l) if len(l) else 0.0 for l in tft_scr_dat]
        # tft_scoreMin_info = [np.min(l) if len(l) else 0.0 for l in tft_scr_dat]

        oth_score_info = [np.sum(k) if len(k) else 0.0 for k in oth_scr_dat]
        # oth_scoreMax_info = [np.max(k) if len(k) else 0.0 for k in oth_scr_dat]
        # oth_scoreMin_info = [np.min(k) if len(k) else 0.0 for k in oth_scr_dat]

        ax1.plot(tft_info, color = 'red', label = 'tft agents')
        ax1.plot(oth_info, color = 'blue', label = 'other agents')
        ax1.set_xlabel("N. Games")
        ax1.set_ylabel("Proportion of Pop.")

        ax2.plot(tft_score_info, color = 'red', label = 'tft score')
        ax2.plot(oth_score_info, color = 'blue', label = 'oth score')
        ax2.set_xlabel("N. Games")
        ax2.set_ylabel("Population Score")

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "tft.png"), dpi = 300)
        plt.show()

if __name__ == "__main__":
    path = "/Users/niwhskal/IPD/logs"
    lw = LogWriter(path)
