#!/usr/bin/env python3

import os
import gc
import numpy as np
import core.env as env
import matplotlib.pyplot as plt
from core.logfn import LogWriter
import core.helperfunctions as hf
import core.hyperParams as hyperParams

rng = np.random.default_rng(1234)
hp = hyperParams.HPTfT()
board_size = hp.board_size

#iteration number
it = 0
iter_limit = hp.max_iter

save_path = os.getcwd()
save_path = os.path.join(save_path, "logs/TfTLogs")

#logger
lw = LogWriter(save_path, hp)

#check if a checkpoint exists
iter_num, bs_new = lw.load_checkpoint()
if iter_num == -1:
    print("checkpoint not found..")
    it = 0
    bs = env.BoardState(rng, hp, board_size)

else:
    print(f"checkpoint found at iter: {iter_num}")
    print("resuming...")
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
    lw.gather_data(agents, bs)
    #log tft data
    lw.gatherTftData(agents, bs)

    print(f"{it} | {my_idx} vs {opp_idx} | agent_scr-> {my_agent.get_score():.2f}, opp_scr-> {opp_agent.get_score():.2f} | tft_count: {len(lw.tft_agents[-1])} | oth_count: {len(lw.other_agents[-1])}")
    it += 1

    #save data once every n games
    if(it % hp.save_every == 0):
        print("saving ...")
        lw.save_data(bs, it)

#lw.plot_clusterData(1000)
lw.plot_tftData()
