#!/usr/env/bin python3

import gc
from core.policy import Policy

class Agent:
    #constructor
    def __init__(self, rng, hyperParams, agent_id:int, memory_length:int) -> None:

        self.rng = rng
        self.hp = hyperParams

        self.agent_id = agent_id
        self.memory_length = memory_length #initialization
        self.avg_score = 0.0

        #current memory
        self.memory = ""

        #n_actions played so far
        self.acts_played = 0

        self.policy = Policy(self.rng, self.hp)

        #if self.hp.mode == "fixed":
        #    #populate with random actions of size mem_len
        #    actions = ['C', 'D', 'M']
        #    actions_toUse = actions[:self.hp.n_actions]
        #    acts = list(self.rng.choice(actions_toUse, size=self.memory_length))
        #    prev_mem = ''.join(acts)
        #    self.add_memory(prev_mem)


    #use during merge/split/mutation
    def set_memory_length(self, m_len:int) -> None:
        self.memory_length = m_len

    #push new memory
    def add_memory(self, mem:str) ->None:
        #most recent -> rightmost
        self.memory += mem
        self.acts_played += 1

    def set_policy(self, new_policy:Policy) -> None:
        self.policy = new_policy

    #return an action given a state
    def act_given(self, state:str) ->str:
        return self.policy.get_action(state)

    def set_score(self, score:float) ->None:
        self.avg_score += score #make sure you divide by acts_played when you use it later

    def get_score(self) ->None:
        if(self.acts_played):
            return self.avg_score/self.acts_played
        else:
            return self.avg_score

    #destructor
    def __del__(self):
        del self.agent_id
        del self.memory_length
        del self.policy

        # free memory
        collected = gc.collect()

#unit tests
if __name__ == "__main__":
    rng = np.random.default_rng(33342)
    hyperParams = HP()

    def get_state(t, q):
        state= ""
        for m1,m2 in zip(t,q):
            state += m1+m2

        #trim memory length to the shortest memory (count from right to left)
        ####

        return state

    a1 = Agent(rng, hyperParams, 0, 0, 5)
    a2 = Agent(rng, hyperParams, 1, 0, 5)

    i = 0
    while(i < 5):

        state_a1 = get_state(a1.memory, a2.memory)
        act1 = a1.act_given(state_a1)

        state_a2 = get_state(a2.memory, a1.memory)
        act2 = a2.act_given(state_a2)

        print(state_a1, state_a2)

        a1.add_memory(act1)
        a2.add_memory(act2)

        new_state_a1 = get_state(a1.memory, a2.memory)
        a1.policy.update_qTable(state_a1, act1, 1.0, new_state_a1)

        new_state_a2 = get_state(a2.memory, a1.memory)
        a2.policy.update_qTable(state_a2, act2, 1.0, new_state_a2)

        print(act1, act2)
        print(a1.memory, a2.memory)
        print("--"*10)
        i+=1

