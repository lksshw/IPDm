#!/usr/bin/env python3

import os
import gc
import numpy as np
import multiprocessing
from core.env import BoardState
from core.logfn import LogWriter
import core.helperfunctions as hf
import core.hyperParams as hyperParams

global_rng = np.random.default_rng(3242)
seeds = [global_rng.integers(1000000) for i in range(hyperParams.HPTfT().n_runs)] #pre define seeds

def run(run_count):
    rng_initSeed = seeds[run_count]
    print(rng_initSeed)
    rng = np.random.default_rng(rng_initSeed)

    hp = hyperParams.HP2Act()
    board_size = hp.board_size

    #logger
    save_path = os.getcwd()
    save_path = os.path.join(save_path, "logs/IPDLogs")
    lw = LogWriter(save_path, hp)

    #max iteration
    max_iter = hp.max_iter

    #check if a checkpoint exists
    iter_num, bs_new = lw.load_checkpoint(run_count)
    if iter_num == -1:
        print("checkpoint not found..")
        it = 0
        bs = BoardState(rng, hp, board_size)

    else:
        print(f"checkpoint found at iter: {iter_num}")
        print("resuming...")
        it = iter_num
        bs = bs_new

    agents = bs.getTurnOrder()


    #log state visitation frequency
    state_map = bs.tree[agents[0]].agent.policy.state_map
    lw.init_stateVisitCounter(state_map)

    while (it<max_iter):

        #logs
        lw.gather_data(agents, bs, it)

        # pick an agent at random
        my_idx = rng.choice(agents)
        me = bs.tree[my_idx]
        my_agent = me.agent

        # opponent is a neighbor (chosen at random)
        try:
            #error handling when neighbor list is empty (the entire grid is a single agent)
            opp_idx = rng.choice(list(me.neighbors))

        except ValueError:
            it += 1

            #save everything since this is the last time step
            print(f"saving! Run - {run_count} | iter: {it}")
            lw.save_data(bs, run_count, it)

            #then terminate
            print(f" singularity at game: {it} | N_agents = {len(agents)} | terminating... ")
            break

        opp = bs.tree[opp_idx]
        opp_agent =  opp.agent

       # in case of split, store subagent indexes
        deletedAgent_info = {}

        #gets actions from states, updates memories, scores, and policies
        #these agents are updated in-place in the bs
        hf.fight(my_agent, opp_agent, hp)

        #check to mutate my policy
        if (bs.rng.random() <=hp.policy_mutation_rate):
            if (hf.worse_than_neighbor(me, bs)):
                my_agent.policy = hf.get_best_neighborPolicy(me, bs)

        #update turnOrder
        agents = bs.getTurnOrder()

        #warning: make sure all neighbors are aware of superagents
        #redundant for 2Act
        #hf.update_neighbors(agents, bs)

        # print(f"{it} | Game: agent-{my_idx} vs {opp_idx} | Scores: agent-> {my_agent.get_score():.2f}, opp-> {opp_agent.get_score():.2f} | N_agents: {len(agents)}")
        it += 1

        #save data once every n games
        if(it % hp.save_every == 0):
            print(f"saving! Run - {run_count} | iter: {it}")
            lw.save_data(bs, run_count, it)

if __name__ == "__main__":
    pool = multiprocessing.Pool(os.cpu_count() - 1)
    hp = hyperParams.HP2Act()
    pool.map(run, range(hp.n_runs))

