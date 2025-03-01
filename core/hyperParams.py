class HP:
    def __init__(self):
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 0.05

        self.board_size = 20

        # iter max
        self.max_iter = int(50e3)

        # threshold score of superAgent below which it splits
        # self.threshold = 3.0

        # mutate policy to that of a better neighbor
        self.policy_mutation_rate = 0.7

        # save frequency
        self.save_every = self.max_iter//1000

        # n_runs
        self.n_runs = 5

        self.payTable = {
            "CC": 8.0,
            "CD": 0.0,
            "CM": 8.0,
            "CS": 0.0,
            "DD": 5.0,
            "DC": 10.0,
            "DM": 10.0,
            "DS": 0.0,
            "MM": 0.0,
            "MC": 8.0,
            "MD": 0.0,
            "MS": 0.0,
            "SC": 0.0,
            "SD": 0.0,
            "SM": 0.0,
            "SS": 0.0,
        }  # format: me,you

        # qlearning or static
        self.policy_type = "qlearning"


class HP4Act(HP):
    def __init__(self):
        super().__init__()
        self.n_actions = 4
        self.mode = "fixed"
        self.max_agentMemory = 4
        # state will a combination of my memory+opponent memory
        self.max_stateLen = self.max_agentMemory*2


class HP2Act(HP):
    def __init__(self):
        super().__init__()
        self.n_actions = 2
        self.mode = "range_memory"
        self.max_agentMemory = 4
        # state will a combination of my memory+opponent memory
        self.max_stateLen = self.max_agentMemory*2


class HPTfT(HP):
    def __init__(self):
        super().__init__()
        self.n_actions = 2
        # memory setting (see _init_tree in env for context)
        self.mode = "fixed"
        self.max_agentMemory = 1
        # state will a combination of my memory+opponent memory
        self.max_stateLen = self.max_agentMemory*2
