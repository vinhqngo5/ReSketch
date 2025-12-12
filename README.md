# ReSketch: A Mergeable, Partitionable, and Resizable Sketch

## Introduction

Tracking items' frequency in data streams is a fundamental problem with applications ranging from network monitoring to database query optimization, machine learning, and more. Sketches offer practical, sublinear-memory solutions that provide high-throughput updates and queries with provable error bounds. Furthermore, sketches are mergeable, which allows multiple sketches of identical parameters to be combined into a single, representative sketch, which enables their use in parallel and distributed systems.

Still, there are important limitations in existing sketch designs that restrict their applicability in modern systems characterized by resource heterogeneity across nodes, workload dynamicity over time, and the need for efficient distributed data aggregation. In this paper, we identify and formalize three critical properties addressing these limitations: **resizability**, **enhanced mergeability**, and **partitionability**. We propose ReSketch, a *sketch base model* that simultaneously achieves all three properties for the first time. As a base model, ReSketch is orthogonal to many existing sketch designs and enables them to support these properties. We believe ReSketch can enable new significant use cases for frequency estimation sketches in modern systems.

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

# Example: Run merge experiment
./build/release/bin/release/merge_experiment -h
./build/release/bin/release/merge_experiment

# Example: Run split experiment
./build/release/bin/release/split_experiment -h
./build/release/bin/release/split_experiment

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

# Visualize merge results
python3 scripts/visualize_merge.py --input output/merge_results_*.json --output merge.png

# Visualize split results
python3 scripts/visualize_split.py --input output/split_results_*.json --output split.png

# Visualize DAG results
python3 scripts/visualize_dag_results.py --input output/dag_results_*.json --output dag.png
```

### Run and Visualize All Experiments
```bash
./scripts/run_all_experiments.sh
```
This will run and visualize all experiments (expansion, merge, split, shrinking, sensitivity) and save results to `output/run_all_<timestamp>/`.

## Requirements

- CMake 3.10+
- C++20 compiler
- Python 3.x (for visualization)
- Required Python packages: `matplotlib`, `numpy`, `scipy`
