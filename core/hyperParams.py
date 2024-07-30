class HP:
    def __init__(self):
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 0.05

        self.board_size = 10

        #iter max
        self.max_iter = int(100)

        #threshold score of superAgent below which it splits
        self.threshold = 5.0

        #mutate policy to that of a better neighbor
        self.policy_mutation_rate = 0.5

        #save frequency
        self.save_every = 10

        #n_runs
        self.n_runs = 5

        self.payTable = {
                "CC": 8.0,
                "CD": 0.0,
                "CM": 8.0,
                "DD": 5.0,
                "DC": 10.0,
                "DM": 10.0,
                "MM": 0.0,
                "MC": 8.0,
                "MD": 0.0,
                } #format: me,you

class HP3Act(HP):
    def __init__(self):
        super().__init__()
        self.n_actions = 3
        self.mode = "range_memory"
        self.max_agentMemory = 5
        self.max_stateLen = self.max_agentMemory*2 #state will a combination of my memory+opponent memory

class HP2Act(HP):
    def __init__(self):
        super().__init__()
        self.n_actions = 2
        self.mode = "range_memory"
        self.max_agentMemory = 5
        self.max_stateLen = self.max_agentMemory*2 #state will a combination of my memory+opponent memory

class HPTfT(HP):
    def __init__(self):
        super().__init__()
        self.mode = "fixed" #memory setting (see _init_tree in env for context)
        self.n_actions = 2
        self.max_agentMemory = 1
        self.max_stateLen = self.max_agentMemory*2 #state will a combination of my memory+opponent memory
