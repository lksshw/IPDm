#!/usr/bin/env python3

import os
import gc
import numpy as np
import core.hyperParams as hyperParams
from core.env import BoardState
import core.helperfunctions as hf
from core.logfn import LogWriter

rng = np.random.default_rng(1234)
hp = hyperParams.HP3Act()
board_size = hp.board_size

#logger
save_path = os.getcwd()
save_path = os.path.join(save_path, "logs/CDMLogs")
lw = LogWriter(save_path, hp)

#max iteration
max_iter = hp.max_iter

##check if a checkpoint exists
#iter_num, bs_new = lw.load_checkpoint()
#if iter_num == -1:
#    print("checkpoint not found..")
#    it = 0
#    bs = BoardState(rng, hp, board_size)

#else:
#    print(f"checkpoint found at iter: {iter_num}")
#    print("resuming...")
#    it = iter_num
#    bs = bs_new

##get initial turn order
#agents = bs.getTurnOrder()

#while (it<max_iter):

#    #logs
#    lw.gather_data(agents, bs)

#    # pick an agent at random
#    my_idx = rng.choice(agents)
#    me = bs.tree[my_idx]
#    my_agent = me.agent

#    # opponent is a neighbor (chosen at random)
#    try:
#        #error handling when neighbor list is empty (the entire grid is a single agent)
#        opp_idx = rng.choice(list(me.neighbors))

#    except ValueError:
#        print("The grid is a single agent; terminating...")
#        break

#    opp = bs.tree[opp_idx]
#    opp_agent = opp.agent

#   # in case of split, store subagent indexes
#    deletedAgent_info = {}

#    #gets actions from states, updates memories, scores, and policies
#    #these agents are updated in-place in the bs
#    hf.fight(my_agent, opp_agent, hp)

#    #check to mutate my policy
#    if (bs.rng.random() <=hp.policy_mutation_rate):
#        if (hf.worse_than_neighbor(me, bs)):
#            my_agent.policy = hf.get_best_neighborPolicy(me, bs)

#    #otherwise, merge/split
#    if (my_agent.memory[-1] == "M" and opp_agent.memory[-1] == "M"):
#        print(f"Merge between: agent-{my_idx}, opp-{opp_idx}")
#        hf.merge(me, opp, bs)

#    elif (hf.is_superAgent(me) and me.agent.get_score() <hp.threshold):
#        print(f"Splitting... agent-{my_idx}")
#        deletedAgent_info = hf.split(me, bs)
#        print(deletedAgent_info)

#    #update turnOrder
#    agents = bs.getTurnOrder()

#    #warning: make sure all neighbors are aware of superagents
#    hf.update_neighbors(agents, bs)

#    print(f"{it} | Game: agent-{my_idx} vs {opp_idx} | Scores: agent-> {my_agent.get_score():.2f}, opp-> {opp_agent.get_score():.2f} | N_agents: {len(agents)}")
#    it += 1

#    #save data once every n games
#    if(it % hp.save_every == 0):
#        print("saving ...")
#        lw.save_data(bs, it)

lw.plot_clusterData(1000)
# lw.plot_memoryData()
