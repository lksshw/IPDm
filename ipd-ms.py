#!/usr/bin/env python3

import os
import gc
import argparse
import numpy as np
import multiprocessing
import core.hyperParams as hyperParams
from core.env import BoardState
import core.helperfunctions as hf
from core.logfn import LogWriter
import time
# from memory_profiler import profile

# @profile
def run(run_count, hp, rng_initSeed):

    rng = np.random.default_rng(rng_initSeed)

    board_size = hp.board_size

    #logger
    save_path = os.getcwd()
    save_path = os.path.join(save_path, "logs/IPDmsLogs")
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
        print(f"checkpoint found!")
        print(f"resuming! Run: {run_count} | iter: {iter_num}")
        it = iter_num
        bs = bs_new

    #get initial turn order
    agents = bs.getTurnOrder()

    #log state visitation frequency
    state_map = bs.tree[agents[0]].agent.policy.qTable.keys()
    # lw.init_stateVisitCounter(list(state_map))

    while (it<max_iter):
        start = time.time()

        #logs
        # lw.gather_data(agents, bs, it)
        # # lw.gatherTftData(agents, bs)
        # lw.record_cluster_size_map(agents, bs, it)
        # lw.record_merge_data(agents, bs, it)

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
        opp_agent = opp.agent

        # in case of split, store subagent indexes
        deletedAgent_info = {}

        #gets actions from states, updates memories, scores, and policies
        #these agents are updated in-place in the bs
        my_state, opp_state = hf.fight(my_agent, opp_agent, hp)

        # lw.update_visitCount(my_state, opp_state)
        lw.record_single_gameplay(my_idx, opp_idx, bs, it) #record the act and memory of agents playing a single round

        #check to mutate my policy
        if (bs.rng.random() <=hp.policy_mutation_rate):
            if (hf.worse_than_neighbor(me, bs)):
                my_agent.policy = hf.get_best_neighborPolicy(me, bs)

        #otherwise, merge/split

        #check for merge or split by randomizing their selection
        op_choose = rng.choice(['op_m', 'op_s'])

        if op_choose == 'op_m':
            #if op code is merge, try merge first, then split
            if (my_agent.memory[-1] == "M" or opp_agent.memory[-1] == "M"):
                # print(f"Merge between: agent-{my_idx}, opp-{opp_idx}")
                hf.merge(me, opp, bs)

            # if you don't merge, then try to split
            else:
                # 1. check if I'm a superagent and I've chosen split;
                if (hf.is_superAgent(me) and my_agent.memory[-1] == "S"):
                    # print(f"Splitting... agent-{my_idx}")
                    deletedAgent_info = hf.split(me, bs)
                    # print(deletedAgent_info)

                #check if my opponent is a superagent and he wants to split
                if (hf.is_superAgent(opp) and opp_agent.memory[-1] == "S"):
                    deletedAgent_info = hf.split(opp, bs)

        elif op_choose == "op_s":
            #if op code is split; then try splitting first
            if (my_agent.memory[-1] == 'S' or opp_agent.memory[-1] == 'S'):

                # 1. check if I'm a superagent and I've chosen split;
                if (hf.is_superAgent(me) and my_agent.memory[-1] == "S"):
                    # print(f"Splitting... agent-{my_idx}")
                    deletedAgent_info = hf.split(me, bs)
                    # print(deletedAgent_info)

                #check if my opponent is a superagent and he wants to split
                if (hf.is_superAgent(opp) and opp_agent.memory[-1] == "S"):
                        deletedAgent_info = hf.split(opp, bs)

            #when neither wants to split, check to merge
            elif (my_agent.memory[-1] == "M" or opp_agent.memory[-1] == "M"):
                # print(f"Merge between: agent-{my_idx}, opp-{opp_idx}")
                hf.merge(me, opp, bs)

        else:
            raise Exception(f"unknown op code: {op_choose}")

        #update turnOrder
        agents = bs.getTurnOrder()

        #warning: make sure all neighbors are aware of superagents
        hf.update_neighbors(agents, bs)

        end = time.time()
        print(f"{it} | Game: agent-{my_idx} vs {opp_idx} | Scores: agent-> {my_agent.get_score():.2f}, opp-> {opp_agent.get_score():.2f} | N_agents: {len(agents)} | te: {end-start:.2f}s")
        it += 1

        #save data once every n games
        if(it % hp.save_every == 0):
            print(f"saving! Run - {run_count} | iter: {it}")
            lw.save_data(bs, run_count, it)

        gc.collect()

if __name__ == "__main__":

    hp = hyperParams.HP4Act()
    global_rng = np.random.default_rng(9583)

    seeds = [global_rng.integers(1000000) for i in range(hp.n_runs)] #pre define seeds

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-l', '--mem_len', type=int)
    parser.add_argument('-bs', '--board_size', type=int, default=20)
    args = parser.parse_args()

    hp.mode = args.mode
    hp.max_agentMemory = args.mem_len
    hp.max_stateLen = hp.max_agentMemory*2
    hp.board_size = args.board_size

    print(f"[EXP]: IPD-ms.py \nmode: {hp.mode}\nmax_mem: {hp.max_agentMemory}\nn_iter: {hp.max_iter}\nn_runs: {hp.n_runs}\nBoard Size: {hp.board_size}")

    pool = multiprocessing.Pool(os.cpu_count() - 2)
    pool.starmap(run, [(i, hp, seeds[i]) for i in range(hp.n_runs)])
