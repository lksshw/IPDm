#!/usr/env/bin python3

import gc
import numpy as np
from itertools import product

class Policy:

    # constructor (initialize empty qTable)
    def __init__(self, rng, hp) -> None:
        assert hp.max_agentMemory <= 7, "maximum memory length must be <=7"
        assert hp.n_actions == 3 or hp.n_actions ==2,  "policy is designed for 2 or 3 actions"

        self.rng = rng
        #hyperparams
        self.hp = hp

        #format: rows = states(str), columns = actions(float32) (order: C, D, M)
        self.actions_list = ['C', 'D', 'M']
        self.actions_list = self.actions_list[:hp.n_actions]
        self._generate_qTableMap()
        self._generate_qTable()

    # maintain a map from state:str to row_index:int so you can query the qTable
    def _generate_qTableMap(self) ->None:

        self.state_map = {}
        row_idx = 0
        for i in np.arange(2, self.hp.max_stateLen+2, 2): #note: states  exist in pairs, odd lengthed states aren't required.
            #get combinations of length i of the actions_list
            combs = [''.join(comb) for comb in product(self.actions_list, repeat=i)]
            #create an entry for each combination
            for item in combs:
                self.state_map[item] = row_idx
                row_idx +=1

    # populate a q table
    def _generate_qTable(self) -> None:
        n_states = len(self.state_map)
        # self.qTable = self.rng.random((n_states, self.hp.n_actions))

        if self.hp.policy_type == "static":
            self.qTable = np.zeros((n_states, self.hp.n_actions))
            for i in range(n_states):
                act_prob = self.rng.random(self.hp.n_actions)
                norm_prob = act_prob/sum(act_prob)
                self.qTable[i, :] = norm_prob

        elif self.hp.policy_type == "qlearning":
            self.qTable = self.rng.random((n_states, self.hp.n_actions))

        else:
            raise Exception("unknown policy initialization method")


    def get_action(self, state:str) ->str:
        if (len(state) ==0 ):
            action = self.rng.choice(self.actions_list)
            return action

        if state not in self.state_map:
            raise Exception(f"{state} is an invalid memory state")

        else:
            #with prob epsilon; explore
            if self.rng.random() < self.hp.epsilon:
                action = self.rng.choice(self.actions_list)

            else:
                action_int = np.argmax(self.qTable[self.state_map[state]]) #this will be an int
                action = self.actions_list[action_int] #query the corresponding action

        return action

    def update_qTable(self, curr_state:str, act:str, payoff_score:float, next_state:str) -> None:
        #if you have 0 memory, you have no policy
        if (len(curr_state) == 0):
            return

        #once the agent takes an action, receive its payoff score and update
        if next_state not in self.state_map:
            raise Exception(f"{next_state} is an invalid memory state")

        action_int = self.actions_list.index(act)
        q_t = self.qTable[self.state_map[curr_state], action_int]
        max_qt1= np.max(self.qTable[self.state_map[next_state]])
        q_tNew = (1-self.hp.alpha)*q_t + self.hp.alpha*(payoff_score + self.hp.gamma*max_qt1)
        self.qTable[self.state_map[curr_state], action_int] = q_tNew

    def copy(self):
        p1 = Policy(self.rng, self.hp)
        p1.qTable = self.qTable.copy()
        return p1

    def __del__(self):
        # del self.qTable
        del self.state_map
        del self.actions_list
        gc.collect()

#tests
if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    hyperParams = HP.HP3Act()

    p1 = Policy(rng, hyperParams)
    state_1 = "CCDMCM"

    # p2 = Policy(rng, hyperParams)

    # state_2 = "DCMD"

    # res = (p1.qTable + p2.qTable)/2
    # print(res)

    # print(p1.qTable[p1.state_map[state_1]])
    # p1.pick_action(state_1)

    # print(p1.action)
    # p1.update_qTable(1.0, state_1)


