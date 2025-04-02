### File organization

- Core/ (dir): contains common modules (see design below)

- cdm.py: simulation of cooperate-defect-merge-split IPD w or w/o varying memory length ([0:5])
- twoAct.py: simulation of classic IPD w or w/o varying memory length ([0:5])

### Modules (interface):

1. Agent: provides routines to maintain and manipulate an independent RL agent. It maintains the following:

   - memory (its own)
   - a policy table
   - a unique id (set during creation)
   - memory length
   - normalized score (from all time steps)

2. BoardState: creates a game board tree and subsequently keeps track of agents and superagents in each round.

   - The main data structure is a tree. Each node of this tree represents an agent with three properties:

     - a set of neighbors
     - a set of parents
     - a set of children

   - During a merge, we update the tree by creating a new node, setting its parents (and consequently setting its parents's "child" attribute)
   - During a split, we delete a node, and update its parents' child node.
   - The tree allows us to track agents of any size (and its neighbors) by traversing the tree (from root to leaf)

3. Policy: a module which takes as input a memory string, returns an action, and updates a transition function. The transition funciton is a q-table based on off-policy Q-learning.

   - Given the maximum memory length, a sub-routine exists which enumerates all possible combinations of memories of the agent+opponent and initilizes its q-values to random values (you'll have to redesign this in case the maximum memory length exceeds 10; because the number of combinations is exponential and it quickly blows up).

   - Provides a routine update the q-table.
   - Provides a routine to sample an action from the q-table.

4. simulation files (cdm, twoAct): creates a BoardState with a number of Agents. Receives turn order (agent id's in the current board state), and runs an iterative loop which pits an agent against an opponent, selects an action, updates its policy, and score.

   - Two agents merge if either choose "merge". A merge updates the BoardState tree to include a super-agent and its neighbors. The Merged-agent's policy will be that of the sub-agent with the higest payoff-score so far.

   - If a super agent chooses the "split" action, irrespective of the opponent's action, it splits. A split deletes the respective super-agent node (identified by its ID and level) in the BoardState tree, and restores its sub-agents with its policy and memory.

   - If any agent's score is worse than all of its neighbors for a time duration of t_rounds, then it copies the policy table of a neighbor's with the highest score so far: the mutation operator

5. Visualize: Creates a real-time panel of:

   - The BoardState with stained super-agents,
   - Graphs of :

   * Avg score graph of all agents,
   * Avg pair of agents cooperating, merging, defecting, or splitting;
   * Avg, min, and max memory length of agents.

6. LogWriter: After every iteration, store the:
   _ BoardState tree,
   _ Policy tables of each agent,
   _ Memories of each agent,
   _ Memory length,
   - Score.

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

   * (initialize a q-table with all possible four-actions-comibinations. Note: limit to only even-lengthed strings.

   Note: redesign the entire class when men_len >10

   - Destructor
     free table memory
     (needs to get called when Agent's destructor get's called)

   - get_action(memory_state::string) -> None

     - Note memory_state is a combination of agents' and opponents' memory (in that order). Eg: if agent_men = [C,C,D] and opponent_mem = [D,M,C], and mem_len = 1, then memory_state from from 1 time-step ago will be: "DC".

   - \_generate_qTable(self) -> None
   - update_qTable(reward, next_state) -> None

3. BoardState
   member functions

   - constructor (board_size::int, hyperparameters::list) -> None
     (initialize new game, gameTree, playList)

   - get_neighbor_of(agent_id::int) -> list[n]
   - initialize_gameTree(self) -> Tree (data structure)
   - grow_tree(parent_list::list[2]) -> None
   - prune_tree(agent_id::int, level_id::int) -> None
   - update_playList(self) -> None

4. Simulate
   script (not a module) (cdm, or twoAct)

   - implement as you see fit using BoardState, Agent, and Policy
   - Pick a random agent, and a random neighbor as opponent in each iteration.
   - based on their individual memory lengths, combine their memories (if different choose the shorter memory-length) Note: pick memories in order of the past. Eg: you: ['D', 'D', 'M'] and opp: ['C', 'M', 'M'], then for a mem length of 3, state = ['MMDMDC']
     Note: at each time $t$, the agent and opponent play based on memories recorded in time steps $mem_len...t-1$
   - Update BoardState, repeat for N rounds.

5. Visualize

   - webgl simulation (to-do)
   - use the Animate class to showcase real-time dynamics.

6. LogWriter
   (a module; implement as you see fit)
   - creates output files in a separate directory based on time-stamps
   - member functions to save BoardState tree, policy table of each agent, memory of each agent, memory_length, and scores of each agent. Corresponding data must be written to output files in real-time.