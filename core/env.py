#!/usr/bin/env python3

import gc
from core.agent import Agent
import core.helperfunctions as hf

class Tree:
    def __init__(self, agent:Agent, parents:set, children:set, neighbors:set):
        self.agent = agent
        self.parents = parents
        self.children = children
        self.neighbors = neighbors

    def __del__(self):
        del self.agent
        del self.parents
        del self.children
        del self.neighbors
        gc.collect()

class BoardState:
    def __init__(self, rng, hyperParams, board_size:int):

        self.rng = rng
        self.hp = hyperParams
        self.uuid_pointer = 0

        self.board_size = board_size
        self.tree = self._initialize_tree()
        self.tree_size = self.board_size**2

    def _initialize_tree(self):

        tree = {}
        for i in range(self.board_size**2):
            #get its neighbors
            neighbors = self.getNeighbors(i)
            #set a random memory length
            if (self.hp.mode == "range_memory"):
                mem_len = self.rng.choice(self.hp.max_agentMemory+1) #a memory of 0 involves picking random actions
            elif (self.hp.mode == "fixed"):
                mem_len = self.hp.max_agentMemory
            else:
                raise Exception("Memory init fault")
            #push to tree
            tree[i] = Tree(agent=Agent(self.rng, self.hp, i, mem_len), parents=set([]), children=set([]), neighbors= neighbors)

        #update last uuid
        self.uuid_pointer = len(tree)

        return tree

    def _getBoardPos(self, idx) ->None:
        row = idx //self.board_size
        col = idx % self.board_size
        return row, col

    def _board2index(self, row, col) ->int:
        idx = int(row*self.board_size + col)
        return idx

    def getNeighbors(self, idx)->set:
        #get agent i's 2D board position
        row, col = self._getBoardPos(idx)

        neighbors = []

        if (row-1 >= 0): #check top
            neighbors.append(self._board2index(row-1, col))

        if (row+1 < self.board_size):#check bot
            neighbors.append(self._board2index(row+1, col))

        if (col-1 >= 0): #check left
            neighbors.append(self._board2index(row, col-1))

        if (col+1 < self.board_size): #check right
            neighbors.append(self._board2index(row, col+1))

        if (row-1>=0 and col-1>=0): #check nw
            neighbors.append(self._board2index(row-1, col-1))

        if (row-1>=0 and col+1<self.board_size): #check ne
            neighbors.append(self._board2index(row-1, col+1))

        if (row+1<self.board_size and col-1>=0): #check sw
            neighbors.append(self._board2index(row+1, col-1))

        if (row+1<self.board_size and col+1<self.board_size): #check se
            neighbors.append(self._board2index(row+1, col+1))

        return set(neighbors)

    def getTurnOrder(self):
        #todo: parallelize
        agents = []
        for i in range(self.board_size**2):
            leaf_agentId = hf.get_leaf(i, self)
            agents.append(leaf_agentId)

        return list(set(agents))


if __name__ == "__main__":

    rng = np.random.default_rng(1234)
    hp = HP()
    bs = BoardState(rng, hp, 3)

    me = bs.tree[0].agent
    opp = bs.tree[1].agent
