#!/usr/bin/env python3

import uuid
import numpy as np
import core.env as env
from core.agent import Agent

def get_superAgentNeighbors(n1: set, n2:set, id1:int, id2: int) -> set:
    temp = n1.union(n2)
    merge_agents = set([id1, id2])
    unique_neighbors = temp - merge_agents
    return unique_neighbors

def trim_toSimilarSize(mem1, mem2):
    if(len(mem1)!=0 and len(mem2)!=0):
        if (len(mem1) <len (mem2)):
            return mem1, mem2[len(mem2)-len(mem1):]
        else:
            return mem1[len(mem1)-len(mem2):], mem2

    else:
        return "",""

def trim_toSimilarMemLen(mem1, mem2, size):
    if(len(mem1)!=0 and len(mem2)!=0):
        if (len(mem1)>=size):
            return mem1[len(mem1)-size:], mem2[len(mem2)-size:]
        else:
            return mem1, mem2

    else:
        return "", ""

def trim(mem1, mem2, smallest_memLen):
    m1, m2 = trim_toSimilarSize(mem1, mem2)
    mem1, mem2 = trim_toSimilarMemLen(m1, m2, smallest_memLen)
    return mem1, mem2


def getState_from(mem1, mem2, smallest_memLen):
    #trim memories to the smallest memory length
    trimmed_m1, trimmed_m2 = trim(mem1, mem2, smallest_memLen)

    state= ""
    for m1,m2 in zip(trimmed_m1, trimmed_m2):
        state += m1+m2

    return state

def smallest(m1, m2):
    if (m1<m2):
        return m1
    else:
        return m2

def fight(me, opp, hp):
    smallest_memLen = smallest(me.memory_length, opp.memory_length)

    #me vs opp
    my_state = getState_from(me.memory, opp.memory, smallest_memLen)
    my_action = me.act_given(my_state)

    #opp vs me
    opp_state = getState_from(opp.memory, me.memory, smallest_memLen)
    opp_action = opp.act_given(opp_state)

    #get scores
    my_score, opp_score = get_payoffScore(my_action, opp_action, hp)

    #update memories
    me.add_memory(my_action)
    opp.add_memory(opp_action)

    #update scores
    me.set_score(my_score)
    opp.set_score(opp_score)

    #get next state (q-learning) and update my policy
    my_newState = getState_from(me.memory, opp.memory, smallest_memLen)
    me.policy.update_qTable(my_state, my_action, my_score, my_newState)

    #similarly to the opponent
    opp_newState = getState_from(opp.memory, me.memory, smallest_memLen)
    opp.policy.update_qTable(opp_state, opp_action, opp_score, opp_newState)

def get_payoffScore(my_act, opp_act, hyperParams):
    hp = hyperParams
    act_myPerspective = my_act+opp_act
    my_score = hp.payTable[act_myPerspective]

    act_oppPerspective = opp_act + my_act
    opp_score = hp.payTable[act_oppPerspective]

    return my_score, opp_score

def merge(me, someone, bs):

    #create agent
    new_memLen = np.max([me.agent.memory_length, someone.agent.memory_length])

    #all new agents carry unique id's
    newAgent_id = bs.uuid_pointer
    new_node = env.Tree(agent = Agent(bs.rng, bs.hp, newAgent_id, new_memLen), parents = set([me.agent.agent_id, someone.agent.agent_id]), children = set([]), neighbors = get_superAgentNeighbors(me.neighbors, someone.neighbors, me.agent.agent_id, someone.agent.agent_id))

    #reset memory
    new_node.agent.memory = ""

    #set the policy to be that of the agent with the best score
    new_node.agent.policy = [me.agent.policy.copy(), someone.agent.policy.copy()][np.argmax([me.agent.get_score(), someone.agent.get_score()])]

    #add it to the board
    bs.tree[newAgent_id] = new_node

    #update parents' children
    me.children = set([newAgent_id])
    someone.children = set([newAgent_id])

    #update uuid pointer
    bs.uuid_pointer += 1

    #update tree size
    bs.tree_size += 1

def is_superAgent(me):
    if (len(list(me.parents))):
        return 1
    else:
        return 0

def split(me, bs):
    #get superAgents' parents, set their policy and memory to be of the superAgent,
    #set their scores to be of the superagent, delete child, delete superagent
    #update bs size
    deleted_agent_info = {me.agent.agent_id: list(me.parents)} #format: superagentkey: subagent
    for parent_id in list(me.parents):
        parent = bs.tree[parent_id]

        parent.agent.policy = me.agent.policy.copy()
        parent.agent.memory = me.agent.memory
        parent.agent.avg_score = me.agent.get_score()
        parent.children = set([])

    #delete superagent
    del bs.tree[me.agent.agent_id]

    #update board size
    bs.tree_size -= 1

    return deleted_agent_info

def worse_than_neighbor(me, bs):
    scores = []
    for neighbor_idx in me.neighbors:
        neighbor = bs.tree[neighbor_idx]
        scores.append(neighbor.agent.get_score())

    scores = np.array(scores)
    summed_val = sum(me.agent.get_score() < scores) #if my score is worse than everyone elses'
    if summed_val == len(scores):
        return 1
    else:
        return 0

def get_best_neighborPolicy(me, bs):
    options = [bs.tree[idx].agent.get_score() for idx in me.neighbors]
    agent_idxs = [idx for idx in me.neighbors]

    best_idx = agent_idxs[np.argmax(options)]
    best_policy = bs.tree[best_idx].agent.policy.copy()
    return best_policy

def get_leaf(node_idx, bs):
    parent_idx = node_idx
    parent = bs.tree[parent_idx]
    while(len(parent.children)):
        if len(parent.children) >1:
            raise Exception("parents can have only one child")

        parent_idx = list(parent.children)[0]
        parent = bs.tree[parent_idx]
    return parent_idx

def get_root(agent_id, bs):
    children = []

    child_idx = agent_id
    child = bs.tree[child_idx]
    if(len(child.parents) == 0):
        return [child_idx]
    first, sec = list(child.parents)
    child_idx = get_root(first, bs)
    children += child_idx

    child_idx = get_root(sec, bs)
    children += child_idx
    return children

def accumulate_neighbors(root_list, bs):
    neighbor_superSet = []

    #combine individual agents' neighbors to get the superagent's neighbors
    for r in root_list:
        neighbor_superSet.append(bs.getNeighbors(r)) #get base neighbors (individual indexes)

    temp = set(list(set().union(*neighbor_superSet)))
    merge_agents = set(root_list)
    unique_neighbors = temp - merge_agents
    new_neighbors = []

    #for each such neighbor, check if it belong to a sueper agent and update
    for nbr_idx in unique_neighbors:
        leaf_idx = get_leaf(nbr_idx, bs)
        new_neighbors.append(leaf_idx)

    return set(new_neighbors)


def update_neighbors(agent_list, bs):
    for agent_id in agent_list:
        #the general idea is to find the root constituents of each agent,
        #get each of their base neighbors(individual agent neighbors); follow their children
        #to eventually arrive at the correct neighbor
        nbrs = accumulate_neighbors(get_root(agent_id, bs), bs)
        bs.tree[agent_id].neighbors = nbrs

def tft_satisfied(my_act, opp_mem):
    if len(opp_mem)==0:
        return 0

    if my_act == opp_mem[-1]:
        return 1

    else:
        return 0


if __name__ == "__main__":
    a = ""
    b = ""
    sm = smallest(5, 3)
    m1, m2 = trim_toSimilarSize(a, b)
    print(m1, m2)
    print(trim_toSimilarMemLen(m1, m2, 10))
    # print(trim(a,b, 2))
    t = [11,2, 3][2]
    print(t)



