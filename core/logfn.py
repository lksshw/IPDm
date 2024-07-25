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

        #log data
        self.hp = hyperParams
        self.clusterSizeData = []
        self.cooperability = []
        self.cluster_score_map = {k: [] for k in range(1, self.hp.board_size**2+1)} #avg score for each cluster_size
        self.memory_sizes = []
        self.memory_score_map = {k: [] for k in range(0, self.hp.max_agentMemory+1)} #max score for each memory_size

        #tft data log
        self.tft_agents = []
        self.other_agents = []

        self.tft_agents_score = []
        self.other_agents_score = []

    def get_cooperability(self, agent_list, bs):
        coop = 0
        for ag_idx in agent_list:
            me = bs.tree[ag_idx]
            try:
                if(me.agent.memory[-1] == "C"):
                    coop += 1
            except IndexError:
                pass
        return coop/len(agent_list) #normalize by the number of agents

    def get_clusterInfo(self, agent_list, bs):
        #get the cluster size of each agent
        #get max score for each cluster
        cluster_size = []
        for ag_idx in agent_list:
            cnt = len(hf.get_root(ag_idx, bs))
            curr_score = bs.tree[ag_idx].agent.get_score()
            self.cluster_score_map[cnt].append(curr_score) #store scores of a particular cluster size
            cluster_size.append(cnt)
        return cluster_size

    def get_memoryInfo(self, agent_list, bs):
        #get a distribution of memory_sizes
        mem_list = []
        for ag_idx in agent_list:
            memLen = bs.tree[ag_idx].agent.memory_length
            mem_list.append(memLen)
            curr_score = bs.tree[ag_idx].agent.get_score()
            self.memory_score_map[memLen].append(curr_score)
        return mem_list

    def gather_data(self, agent_list, bs):
        cluster_sizes = self.get_clusterInfo(agent_list, bs)
        coop_count = self.get_cooperability(agent_list, bs)

        self.clusterSizeData.append(cluster_sizes)
        self.cooperability.append(coop_count)
        #cluster score map exists as well

        memData = self.get_memoryInfo(agent_list, bs)
        self.memory_sizes.append(memData)
        #memory score map exists as well

    def gatherTftData(self, agent_list, bs):
        #check the action each agent would play against its opponent, verify if it is equal to the opponent's last action
        tft_agents = []
        other_agents = []

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
                tft_agents.append(ag_idx)
                tft_agents_score.append(me.agent.get_score())

            else:
                other_agents.append(ag_idx)
                other_agents_score.append(opp.agent.get_score())

        self.tft_agents.append(tft_agents)
        self.other_agents.append(other_agents)

        self.tft_agents_score.append(tft_agents_score)
        self.other_agents_score.append(other_agents_score)

    def save_file(self, fname, file):
        with open(fname, "wb") as F:
            pickle.dump(file, F, pickle.HIGHEST_PROTOCOL)

    def load_file(self, fname):
        with open(fname, "rb") as inp:
            data = pickle.load(inp)
        return data

    def save_data(self, bs, iter):
        #cluster
        self.save_file(os.path.join(self.checkpoint_path, "clustersizeList-vs-time.pkl"), self.clusterSizeData)
        self.save_file(os.path.join(self.checkpoint_path, "cooperability-vs-time_merge.pkl"), self.cooperability)
        self.save_file(os.path.join(self.checkpoint_path, "cluster_score_map.pkl"), self.cluster_score_map)

        #memory
        self.save_file(os.path.join(self.checkpoint_path, "memorySizeList-vs-time.pkl"), self.memory_sizes)
        self.save_file(os.path.join(self.checkpoint_path, "memory_score_map.pkl"), self.memory_score_map)

        #save boardstate
        self.save_file(os.path.join(self.checkpoint_path, "boardstate.pkl"), bs)

        #save tft data
        self.save_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"), self.tft_agents)
        self.save_file(os.path.join(self.checkpoint_path, "other_agents.pkl"), self.other_agents)

        self.save_file(os.path.join(self.checkpoint_path, "tft_agents_score.pkl"), self.tft_agents_score)
        self.save_file(os.path.join(self.checkpoint_path, "other_agents_score.pkl"), self.other_agents_score)

        #save metadata
        meta_data = {"iter_num": iter}
        self.save_file(os.path.join(self.checkpoint_path, "meta_data.pkl"), meta_data)

    def load_checkpoint(self):
        #meta_data
        try:
            meta_data = self.load_file(os.path.join(self.checkpoint_path, "meta_data.pkl"))
        except:
            return -1, -1

        iter_num = meta_data["iter_num"]

        #boardstate
        bs = self.load_file(os.path.join(self.checkpoint_path, "boardstate.pkl"))

        #cluster data
        self.clusterSizeData = self.load_file(os.path.join(self.checkpoint_path, "clustersizeList-vs-time.pkl"))
        self.cooperability = self.load_file(os.path.join(self.checkpoint_path, "cooperability-vs-time_merge.pkl"))
        self.cluster_score_map = self.load_file(os.path.join(self.checkpoint_path, "cluster_score_map.pkl"))

        #memory data
        self.memory_sizes = self.load_file(os.path.join(self.checkpoint_path, "memorySizeList-vs-time.pkl"))
        self.memory_score_map = self.load_file(os.path.join(self.checkpoint_path, "memory_score_map.pkl"))

        #tft data
        self.tft_agents = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        self.other_agents = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))

        self.tft_agents_score = self.load_file(os.path.join(self.checkpoint_path, "tft_agents_score.pkl"))
        self.other_agents_score = self.load_file(os.path.join(self.checkpoint_path, "other_agents_score.pkl"))

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
        tft_dat = self.load_file(os.path.join(self.checkpoint_path, "tft_agents.pkl"))
        oth_dat = self.load_file(os.path.join(self.checkpoint_path, "other_agents.pkl"))

        tft_scr_dat = self.load_file(os.path.join(self.checkpoint_path, "tft_agents_score.pkl"))
        oth_scr_dat = self.load_file(os.path.join(self.checkpoint_path, "other_agents_score.pkl"))

        #plot
        tft_info = [len(l)/(len(l)+len(oth_dat[idx])) for idx, l in enumerate(tft_dat)] #vs timestep
        oth_info = [len(j)/(len(j)+len(tft_dat[idx])) for idx, j in enumerate(oth_dat)] #vs timestep

        tft_score_info = [np.max(l) if len(l) else 0.0 for l in tft_scr_dat]
        oth_score_info = [np.max(k) if len(k) else 0.0 for k in oth_scr_dat]

        ax1.plot(tft_info, color = 'red', label = 'tft agents')
        ax1.plot(oth_info, color = 'blue', label = 'other agents')
        ax1.set_xlabel("N. Games")
        ax1.set_ylabel("Proportion of Pop.")

        ax2.plot(tft_score_info, color = 'red', label = 'tft score')
        ax2.plot(oth_score_info, color = 'blue', label = 'oth score')
        ax2.set_xlabel("N. Games")
        ax2.set_ylabel("Max. score")

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "tft.png"), dpi = 300)
        plt.show()

if __name__ == "__main__":
    path = "/Users/niwhskal/IPD/logs"
    lw = LogWriter(path)
