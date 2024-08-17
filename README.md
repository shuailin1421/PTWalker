# PTWalker

This repository holds the manuscript of "Towards Probabilistic Termination Random Walks with Efficient Walker Updating and Management".

## System Specification

All experiments are run on a Dell PowerEdge R740 server running Ubuntu 20.04.2 LTS. It has two Intel(R) Xeon(R) Gold 5218R processors, each with 20 cores and 64GB DRAM. Each core has a private 1MB L2 Cache.
All datasets are undirected graph, which means for a vertex i has an edge connecting with a vertex j, then vertex j also has an edge connecting with vertex i.
Each dataset should be preprocessed as a directed edge lists format.

## Setup

### Install Dependencies

```bash
sudo apt-get update
sudo apt-get install cmake g++ autoconf libtool libnuma-dev -y
```

### Compile PTWalker

```bash
cd PTWalker
mkdir build && cd build
cmake ..
make && make install
```

## Evaluation

Evaluate PageRank:

```bash
./bin/pagerank -f text -g datapath -w walker_num --sp termination_prb -s 1 -t 20
```

Evaluate PPR:

```bash
./bin/PPR -f text -g datapath -w walker_num --sp termination_prb -s 1 -t 20
```

Evaluate RWR:

```bash
./bin/RWR -f text -g datapath -w walker_num --sp termination_prb -s 1 -t 20
```

Evaluate DeepWalk:

```bash
./bin/deepwalk -f text -g datapath -w walker_num -l num_of_steps -s 1 -t 20
```
