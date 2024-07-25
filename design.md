### Modules (interface):

1. Agent: provides routines to maintain and manipulate an independent RL agent. It maintains the following:

   - memory (its own)
   - a policy table
   - a unique id (set during its instantiation)
   - memory length
   - normalized score (from all time steps)

2. BoardState: creates a game board and subsequently keeps track of agents and superagents in each round. Tracking involves the following:

   - maintaining a tree whose root nodes hold individual agent id's (this is level 0), level 1 will hold super-agent id's. Individual agents which merge act as parents to a particular super agent. Once merged, individual agents  of a super-agent share a common memory/policy table. Similarly, level-2 connects two super-agents to super-super-agent ids. Each node in this tree (at any level) also carries a list of its neighbors, so each agent/super-agent/super-super-agent... carries a list of its neighbors. 
     Note: each node can have only one child

   This module also provides a map of agents: a list of different agents within the 2D grid which can play against each other and is updated every round.

   - To create such a play-list, we traverse the tree in bottom-up order (leaves to root). The bottommost level provies the list of super-agents, chidless nodes at higher levels provide us with other agent ID's.

3. Policy: a module which takes as input a memory string, returns an action, and updates a transition function. Different kinds of policies exist based on the type of strategy. Each policy type can be a subclass. By default the parent class will implement Q-learning.

   - Given the maximum memory length, a sub-routine exists which enumerates all possible combinations of memories of the agent+opponent and initilizes its q-values to random values. ($ N_combinations = N_actions^{mem_len} $), where mem_len = agent_len + opp_len. (you'll have to redesign this in case the maximum memory length exceeds 10; because the number of combinations is exponential and it quickly blows up)

   - Provides a routine to apply a learning rule to the q-table
   - Provides a routine to sample an action from the q-table.

Derived policy classes can implement deterministic policies (eg: tft, where we maintain a table mapping opposite actions to each other), or other stochastic policies (involving fn approximation with nn's).

4. Simulate: Creates a BoardState with a number of Agents. Receives turn order, and runs an iterative loop which pits an agent against an opponent, selects an action, updates its policy, and score;
    Note: initially both play random actions, from the next timestep onwards, the agent plays; then the opponent plays based on their previous memory states. Once they've both played their hand, rewards get set and their policies are accordingly updated.

    - Two agents merge if their last actions are "merge". A merge updates the BoardState tree to include a super-agent and its neighbors. Merged policy will be that of the agent with the higest score so far.

   - If a super agent's score drops below a threshold, it splits. A split deletes the respective super-agent node (identified by its ID and level) in the BoardState tree.

   - If any agent's score is worse than all of its neighbors for a time duration of t_rounds, then it copies the policy table of a neighbor's with the highest score so far.

5. Visualize: Creates a real-time panel of: 

   * The BoardState with stained super-agents, 
   * Avg score graph of all agents, 
   * Avg pair of agents cooperating, merging, and defecting; 
   * Avg, min, and max memory length of agents.

6. LogWriter: After every iteration, store the:
   _ BoardState tree,
   _ Policy tables of each agent,
   _ Memories of each agent,
   _ Memory length, 
   -  Score.

### Modules (implementation):

1. Agent
   member functions

   - constructor (agent_id::int, tree-level::int, memory_length::int, memory:str) -> None
     (initialize new Agent)

   - destructor
     (deletes object from memory)
     (call Policy's destructor)

   - set_memory_length(int) -> None
   - add_memory(mem:str) -> None
   - set_policy(Policy) -> None
   - set_score(float) -> None
   - act_given(state) -> str

2. Policy
   member functions

   - constructor (n_actions:int, memLen:int) -> None
    * (initialize a q-table with all possible three-actions-comibinations. This will be a geometric series: $\sum_{i=1}^{max_memLen}x^i = (n_actions**maxLen -1)/ (n_actions -1) - 1$ 

     Note: redesign this class for agent_memLen >7

   - Destructor
     free table memory
     (needs to get called when Agent's destructor get's called)

   - get_action(memory_state::string) -> None

     - Note memory_state is a combination of agents' and opponents' memory (in that order). Eg: if agent_men = [C,C,D] and opponent_mem = [D,M,C], and mem_len = 1, then memory_state from from 1 time-step ago will be: "DC".

   - _generate_qTable(self) -> None
   - update_qTable(reward, next_state) -> None

   - Note: to create a new strategy, create a new class derived from Policy, and rewrite these member functions. Eg: for tft, set Q-table to be a diagonal matrix and only rewrite get_action to return an action from the q-table.

3. BoardState
   member functions

   - constructor (board_size::int, \*\*hyperparameters::list) -> None
     (initialize new game, gameTree, playList)

   - get_neighbor_of(agent_id::int) -> list[n]
   - initialize_gameTree(self) -> Tree (data structure)
   - grow_tree(parent_list::list[2]) -> None
   - prune_tree(agent_id::int, level_id::int) -> None
   - update_playList(self) -> None


4. Simulate
   script (not a module)
   - implement as you see fit using BoardState, Agent, and Policy
   - Pick a random agent, and a random neighbor as opponent in each iteration. 
   - based on their individual memory lengths combine their memories (if different choose the shorter one) Note: pick memories in order of the past. Eg: you: ['D', 'D', 'M'] and opp: ['C', 'M', 'M'], then for a mem length of 3, state = ['MM DM DC'] 
    Note: at each time-step, the agent plays; then the opponent plays based on their previous memory states. Once they've both played their hand, rewards get set and their policies are updated.

    - Update BoardState, repeat for N rounds.
    - make sure you use a random generator

5. Visualize

   - matplotlib panels
   - use the Animate class to showcase real-time dynamics.

6. LogWriter
   (a module; implement as you see fit)
   - creates output files in a separate directory based on time-stamps
   - member functions to save BoardState tree, policy table of each agent, memory of each agent, memory_length, and scores of each agent. Corresponding data must be written to output files in real-time.

