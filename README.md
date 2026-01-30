# ReSketch: A Mergeable, Partitionable, and Resizable Sketch

## Introduction

Tracking items' frequency in data streams is a fundamental problem with applications ranging from network monitoring to database query optimization, machine learning, and more. Sketches offer practical, sublinear-memory solutions that provide high-throughput updates and queries with provable accuracy approximation bounds. Furthermore, sketches are mergeable, which allows multiple sketches of identical parameters to be combined into a single, representative sketch, which enables their use in parallel and distributed systems.

Still, there are limitations in known sketch designs that restrict their applicability in systems characterized by resource heterogeneity across nodes, workload fluctuation over time, and the need for efficient distributed data aggregation. We identify and formalize three critical properties that can address these limitations: **resizability**, **enhanced mergeability**, and **partitionability**. We propose ReSketch, a matrix-based sketch algorithmic design, which, through a combination of consistent hashing and quantile sketching, fused with a partition-aware hashing technique, leads to the ability to satisfy all three properties, with a beneficial memory-to-accuracy ratio. We propose an analysis methodology for dynamic sketches and apply it to investigate the costs and benefits of ReSketch, in conjunction with a detailed empirical study that also includes its time-associated behavior. 

As ReSketch is orthogonal to other matrix-based sketches, we expect it can enable them to support the aforementioned properties and, in turn, lead to new significant use cases for frequency estimation sketches in modern systems.

## Key Implementation

The core implementation is in [`src/frequency_summary/resketchv2.hpp`](src/frequency_summary/resketchv2.hpp), which provides:
- **Resizability**: `expand()` and `shrink()` operations.
- **Enhanced Mergeability**: `merge()` operation.
- **Partitionability**: `split()` operation.

## Quick Start

### Clone and Setup
```bash
git clone https://github.com/vinhqngo5/ReSketch.git
cd ReSketch
./scripts/setup_submodules.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build
```bash
cmake --preset=release
cmake --build build/release -j$(nproc)
```

### Run Single Experiment
```bash
# Example: Run expansion experiment
./build/release/bin/release/expansion_experiment -h 
./build/release/bin/release/expansion_experiment

# Example: Run shrinking experiment
./build/release/bin/release/shrinking_experiment -h 
./build/release/bin/release/shrinking_experiment

# Example: Run expansion then shrinking experiment
./build/release/bin/release/expansion_shrinking_experiment  -h 
./build/release/bin/release/expansion_shrinking_experiment 

# Example: Run merge experiment
./build/release/bin/release/merge_experiment -h
./build/release/bin/release/merge_experiment

# Example: Run partition experiment
./build/release/bin/release/partition_experiment -h
./build/release/bin/release/partition_experiment

# Example: Run DAG experiment
./build/release/bin/release/dag_experiment -h
./build/release/bin/release/dag_experiment
```

### Visualize Single Experiment
```bash
# Visualize expansion results
python3 scripts/visualize_expansion.py --input output/expansion_results_*.json --output expansion.png

# Visualize shrinking results
python3 scripts/visualize_shrinking.py --input output/shrinking_results_*.json --output shrinking.png

# Visualize expansion_shrinking results
python3 scripts/visualize_expansion_shrinking.py --input output/expansion_shrinking_results_*.json --output expansion_shrinking.png

# Visualize merge results
python3 scripts/visualize_merge.py --input output/merge_results_*.json --output merge.png

# Visualize partition results
python3 scripts/visualize_partition.py --input output/partition_results_*.json --output partition.png

# Visualize DAG results
python3 scripts/visualize_dag_results.py --input output/dag_results_*.json --output dag.png
```

### Run and Visualize All Experiments
```bash
./scripts/run_experiment.sh all
```
This will run and visualize all experiments (expansion, merge, partition, shrinking, sensitivity) using default settings and save results to `output/run_all_<timestamp>/`.

### Reproduce experimental results from the paper
```bash
./scripts/reproduce.sh
```
This will run and visualize all experiments (expansion, merge, partition, shrinking, sensitivity) using settings specified in the paper and save results to `output/reproduce_all_<timestamp>/`.

## Requirements

- CMake 3.10+
- C++20 compiler
- Python 3.x (for visualization)
- Required Python packages: `matplotlib`, `numpy`, `scipy`
