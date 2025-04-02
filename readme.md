### **Spatialized Iterated Prisoners Dilemma with a Varying Number of Agents**
---

This repository contains code corresponding to the work known by the title of: "Extending Iterated, Spatialized Prisoners’ Dilemma to Understand Multicellularity: game theory with self-scaling players"

The general idea is to run a simulation where a number of independent RL agents play Iterated prisoners dilemma (IPD) games with one another, but with a twist: instead of just the ability to _cooperate_ or _defect_, agents in-addition, have the option to _merge_ or _split_. These operators allow independent agents the ability to forego their own individuality in exchange for that of another, better performant individual, with their 2D spatial arrangement growing/shrinking accordingly.

Here's a visualization: (the colorbar is representative of agents' size at any simulation step):

<img src="./metadata/ipdms-sim.gif" alt="sim" width="500" display='block'/>

### **Simulation**

---

Install requisites (preferably in a virtualenv):

```bash
pip3 install -r metadata/requirements.txt
```

A simulation of merge based IPD (termed ipd-ms) can be run as:

```python
python3 ipd-ms.py --mode "fixed" --mem_len 4 --bs 20
```

The _--mode_ option specifies if agents are to be homogenously initialized with a constant memory capacity (mode = _fixed_) or heterogenously from a uniform random distribution (mode = _range_memory_ ) whose bounds are decided by the _--mem_len_ parameter (_--mem_len = 4_, set here).

The size of the arena in which agents play merge based IPD games is set using the boardsize (_bs_) option. Here we set it to a value of 20 (implying a 20x20 grid composed of 400 agents).

A simulation of classic IPD (with _cooperate_ and _defect_ as possible actions) can be run as:

```python
python3 ipd.py --mode "fixed" --mem_len 4 --bs 20
```

The file, [hyperparams.py](https://github.com/lksshw/IPDm/blob/main/core/hyperParams.py) maintains a list of invariant hyperparameters values related to the q-learning algorithm, mutation rate etc.

### **Results**

---

The results we report focuses on the relationship between memory size (varied by the _--mem_len_ option) and merge tendency (varied by enabling or disabling merge/split actions: by running either ipd-ms or ipd simulations)

```bash
chmod +x plot.sh
./plot.sh
```

note: plot.sh fetches cached simulation data (running them from scratch is time-consuming). A re-run of these simulations should yield ~the same results.
