#!/usr/bin/env python3

import os
import gc
import numpy as np
import core.env as env
import multiprocessing
import matplotlib.pyplot as plt
from core.logfn import LogWriter
import core.helperfunctions as hf
import core.hyperParams as hyperParams

global_rng = np.random.default_rng(1234)
seeds = [global_rng.integers(1000000) for i in range(hyperParams.HPTfT().n_runs)] #pre define seeds

def run(run_count):
    rng_initSeed = seeds[run_count]
    print(rng_initSeed)
    rng = np.random.default_rng(rng_initSeed)
    hp = hyperParams.HPTfT()
    board_size = hp.board_size

    #iteration number
    iter_limit = hp.max_iter

    save_path = os.getcwd()
    save_path = os.path.join(save_path, "logs/TfTLogs")

    #logger
    lw = LogWriter(save_path, hp)

    #check if a checkpoint exists
    iter_num, bs_new = lw.load_checkpoint(run_count)
    if iter_num == -1:
        print("checkpoint not found..")
        it = 0
        bs = env.BoardState(rng, hp, board_size)

    else:
        print(f"checkpoint found!")
        print(f"resuming! Run: {run_count} | iter: {iter_num}")
        it = iter_num
        bs = bs_new

    #get initial turn order
    agents = bs.getTurnOrder()

    while (it<iter_limit):
        # pick an agent at random
        my_idx = rng.choice(agents)
        me = bs.tree[my_idx]
        my_agent = me.agent

        # opponent is a neighbor (chosen at random)
        try:
            #error handling when neighbor list is empty (the entire grid is a single agent)
            opp_idx = rng.choice(list(me.neighbors))

        except ValueError:
            raise Exception("The grid is a single agent; terminating...")

        opp = bs.tree[opp_idx]
        opp_agent = opp.agent

       # in case of split, store subagent indexes; initialization here
        deletedAgent_info = {}

        hf.fight(my_agent, opp_agent, hp)

        #check to mutate my policy
        if (bs.rng.random() <=hp.policy_mutation_rate):
            if (hf.worse_than_neighbor(me, bs)):
                my_agent.policy = hf.get_best_neighborPolicy(me, bs)

        #update turnOrder
        agents = bs.getTurnOrder()

        #make sure all neighbors are aware of superagents/splits (redundant here)
        # hf.update_neighbors(agents, bs)

        #log: scores, and size
        lw.gather_data(agents, bs, it)
        #log tft data
        lw.gatherTftData(agents, bs)

        print(f"{it} | N_agents: {len(agents)} | max_score: {lw.tft_agents_score[-1]}")
        it += 1

        #save data once every n games
        if(it % hp.save_every == 0):
            print(f"saving! Run - {run_count} | iter: {it}")
            lw.save_data(bs, run_count, it)

if __name__ == "__main__":
    # pool = multiprocessing.Pool(os.cpu_count() - 1)
    # hp = hyperParams.HPTfT()
    # pool.map(run, 1)
    run(0)

