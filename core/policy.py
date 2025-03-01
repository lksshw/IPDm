#!/usr/env/bin python3

import gc
import numpy as np
from itertools import product
# import hyperParams as HP

class Policy:

    # constructor (initialize empty qTable)
    def __init__(self, rng, hp) -> None:
        assert hp.max_agentMemory <= 7, "maximum memory length must be <=7"
        assert hp.n_actions == 4 or hp.n_actions ==2,  "policy is designed for 2 or 3 actions"

        self.rng = rng
        #hyperparams
        self.hp = hp

        #format: rows = states(str), columns = actions(float32) (order: C, D, M)
        self.actions_list = ['C', 'D', 'M', 'S']
        self.actions_list = self.actions_list[:hp.n_actions]
        self._generate_qTable()

    #create qtable (json format)
    def _generate_qTable(self) ->None:

        self.qTable = {}
        for i in np.arange(2, self.hp.max_stateLen+2, 2): #note: states  exist in pairs, odd lengthed states aren't required.
            #get combinations of length i of the actions_list
            combs = [''.join(comb) for comb in product(self.actions_list, repeat=i)]
            #create an entry for each combination
            for item in combs:
                if self.hp.policy_type == "static":
                    act_prob = np.round(self.rng.random(self.hp.n_actions), decimals = 4)
                    norm_prob = act_prob/sum(act_prob)
                    self.qTable[item] = {ac: norm_prob[idx_ac] for idx_ac, ac in enumerate(self.actions_list)}

                elif self.hp.policy_type == "qlearning":
                    self.qTable[item] = {ac: np.round(self.rng.random(), decimals = 4) for idx_ac, ac in enumerate(self.actions_list)}

                else:
                    raise Exception("unknown policy initialization method")

    def get_action(self, state:str) ->str:
        if (len(state) ==0 ):
            action = self.rng.choice(self.actions_list)
            return action

        if state not in list(self.qTable.keys()):
            raise Exception(f"{state} is an invalid memory state")

        else:
            #with prob epsilon; explore
            if self.rng.random() < self.hp.epsilon:
                action = self.rng.choice(self.actions_list)

            else:
                action_int = np.argmax(list(self.qTable[state].values())) #this will be an int
                action = self.actions_list[action_int] #query the corresponding action

        return action

    def update_qTable(self, curr_state:str, act:str, payoff_score:float, next_state:str) -> None:
        #if you have 0 memory, you have no policy
        if (len(curr_state) == 0):
            return

        #once the agent takes an action, receive its payoff score and update
        if next_state not in list(self.qTable.keys()):
            raise Exception(f"{next_state} is an invalid memory state")

        q_t = self.qTable[curr_state][act]
        max_qt1 = np.max(list(self.qTable[next_state].values()))
        q_tNew = (1-self.hp.alpha)*q_t + self.hp.alpha*(payoff_score + self.hp.gamma*max_qt1)
        self.qTable[curr_state][act] = np.round(q_tNew, decimals = 4)

    def copy(self):
        p1 = Policy(self.rng, self.hp)
        p1.qTable = self.qTable.copy()
        return p1

    def __del__(self):
        del self.qTable
        del self.actions_list
        gc.collect()

#tests
if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    hp = HP.HP4Act()

    p1 = Policy(rng, hp)
    print(p1.qTable.keys())
    state_1 = "DM"

    print(p1.get_action(""))

    p1.get_action(state_1)
    print(p1.qTable[state_1])

    p1.update_qTable(state_1, 'C', 0.5, 'CDMM')
    print(p1.qTable[state_1])

