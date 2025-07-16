### **Spatialized Iterated Prisoners Dilemma with a Varying Number of Agents**

<div align="center">
    <a href="https://ieeexplore.ieee.org/document/10970107">Paper</a>
    <a href="https://lksshw.github.io/">Demo</a>
    <a href="https://www.youtube.com/watch?v=iOFHyNliS9M&ab_channel=MichaelLevin%27sAcademicContent">Talk</a>
</div>

<br>

This repository contains code corresponding to the work titled: "Extending Iterated, Spatialized Prisoners’ Dilemma to Understand Multicellularity: game theory with self-scaling players"

The general idea is to run a simulation where independent RL agents play Iterated prisoners dilemma (IPD) games with one another. But with a twist: instead of their restricted ability to _cooperate_ or _defect_, agents in-addition, have the option to _merge_ or _split_. These operators allow agents the ability to forego their individuality in exchange for that of another, better performant individual, with their 2D spatial arrangement growing/shrinking accordingly.

Here's a visualization: (the colorbar is representative of agents' size).

<div style= "text-align:center;">
    <img src="./metadata/ipdms-sim.gif" alt="sim" width="450"/>
</div>

<br>

The [interactive demo](https://lksshw.github.io/) might provide a better understanding of the matter.

### **Simulation**

In a virtualenv:

```bash
pip3 install -r metadata/requirements.txt
```

A simulation of merge based IPD (termed ipd-ms) can be run as:

```python
python3 ipd-ms.py --mode "fixed" --mem_len 4 -bs 20
```
and,

A simulation of classic IPD (with _cooperate_ and _defect_ as possible actions) can be run as:

```python
python3 ipd.py --mode "fixed" --mem_len 4 -bs 20
```

These hyperparameters require some background on the design of our RL agents:

* Each RL agent carries with it two data structures: a list of its memories (actions played so far), and a policy table (discrete map from memory states to actions).

* The memory size (size of the list) can be pre-set to a fixed capacity (represented by the --mem_len hyperparameter).

* Given that each simulation involves multiple agents (determined by the -bs hyperparmeter), the --mode hyperparameter provides a way to set the --mem_len hyperparameter of each agent.

Specifically, in the code snippet above:

- The _--mode_ option specifies if agents are to be uniformly set to a constant memory capacity (mode = _fixed_) or heterogenously set to sizes from a uniform random distribution (mode = _range_memory_ ) whose upper-bound is decided by the _--mem_len_ parameter (_--mem_len = 4_, set here).

- The _-bs_ option specifies the number of RL agents to initialize. Here we set it to a value of 20 (implying a 20x20 grid composed of 400 agents).

The file, [hyperparams.py](https://github.com/lksshw/IPDm/blob/main/core/hyperParams.py) maintains a list of invariant hyperparameters values related to the q-learning algorithm, mutation rate etc.

### **Results**

The results we report focuses on the relationship between memory size (varied by the _--mem_len_ option) and merge tendency (varied by enabling or disabling merge/split actions: by running either ipd-ms or ipd simulations)

running [plot.sh](https://github.com/lksshw/IPDm/blob/main/metadata/plot.sh) replicates our result:

```bash
cd metadata
chmod +x plot.sh
./plot.sh
```

note: plot.sh fetches cached simulation data (running them from scratch is time-consuming). A re-run of these simulations should yield ~the same results.
