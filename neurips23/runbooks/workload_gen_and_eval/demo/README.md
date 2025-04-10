# Dynamic Workload Generator Demo

This document demonstrates the capabilities of the dynamic workload generator. The generator produces a sequence of vector search operations—**insert**, **delete**, and **query**. This demo shows how different sampling strategies impact the overall simulation.


---

## Table of Contents

- [Overview](#overview)
- [Key Simulation Parameters](#key-simulation-parameters)
- [Workload Examples](#configuration-examples)
- [How to Run the Generator](#how-to-run-the-generator)

---

## Overview

The workload generator operates as follows:

1. **Loading the Data:**  
   It loads a dataset (for this demo, a 2D random vector dataset) and any corresponding query vectors.

2. **Clustering & Resident Set Initialization:**  
   Base vectors are clustered using Faiss, and an initial resident set is sampled.

3. **Operation Generation:**  
   A series of operations is generated:
   - **Insert:** New vectors are sampled from the dataset added to the resident set.
   - **Delete:** Vectors are removed from the resident set.
   - **Query:** A set of query vectors are sampled from the queries provided by the dataset.

4. **Output Generation:**  
   The generator produces:
   - JSON runbook file containing the operations and a summary of the workload.
   - Numpy arrays containing the ids in the initial resident set and the per operation ids.
---

## Key Simulation Parameters

The generator’s behavior is controlled by several parameters:

- **Operation Ratios:**  
  - **Insert Ratio:** Fraction of operations that insert vectors.
  - **Delete Ratio:** Fraction that delete vectors.
  - **Query Ratio:** Fraction that issue queries.  
  _Note: The sum must equal 1.0._

- **Batch Sizes:**  
  - **Update Batch Size:** Number of vectors processed per insert or delete.
  - **Query Batch Size:** Number of queries per operation.

- **Initialization & Operation Count:**  
  - **Initial Resident Set Size:** Number of vectors in the resident set at startup.
  - **Number of Operations:** Total operations to generate. If there are not enough vectors in the dataset, the generator will stop early.

- **Sampling Strategies:**  
  Choose between:
  - **Uniform:** Pure random selection.
  - **Clustered Drift:** A structured, drifting sampling based on clusters.
  - **Clustered Random:** Random sampling of clusters.

- **Other Parameters:**  
  - **Cluster Size:** Controls clustering granularity.
  - **Metric:** Choice of `"l2"` or `"ip"` (inner product).
  - **Random Seed:** Ensures reproducibility.

---

## Configuration Examples

Below are three demo configurations using the `"random-2d"` dataset, with differing sampling strategies.

### 1. Clustered Drift

```yaml
dataset: "random-2d"
workload_dir: "workload/demo2d_clustered_drift"
metric: "l2"
insert_ratio: 0.5
delete_ratio: 0.25
query_ratio: 0.25
update_batch_size: 100
query_batch_size: 10
num_operations: 1000
initial_size: 1000
cluster_size: 100
update_sample_distribution: "clustered_drift"
query_sample_distribution: "clustered_drift"
seed: 1738
```

Expected Behavior: The resident set will exhibit structured drift over time, with clusters gradually shifting as the sampling prioritizes specific clusters.

⸻

2. Clustered Random

```yaml
dataset: "random-2d"
workload_dir: "workload/demo2d_clustered_random"
metric: "l2"
insert_ratio: 0.5
delete_ratio: 0.25
query_ratio: 0.25
update_batch_size: 100
query_batch_size: 10
num_operations: 1000
initial_size: 1000
cluster_size: 100
update_sample_distribution: "clustered_random"
query_sample_distribution: "clustered_random"
seed: 1738
```

Expected Behavior: Sampling is based on cluster structure but chosen randomly. The resident set will show a mixture of randomness with some clustering influence.

⸻

3. Uniform Sampling

```yaml
dataset: "random-2d"
workload_dir: "workload/demo2d_uniform"
metric: "l2"
insert_ratio: 0.5
delete_ratio: 0.25
query_ratio: 0.25
update_batch_size: 100
query_batch_size: 10
num_operations: 1000
initial_size: 1000
update_sample_distribution: "uniform"
query_sample_distribution: "uniform"
seed: 1738
```

Expected Behavior: No cluster-based bias is applied; the resident set evolution is driven by uniform random sampling.

⸻
## How to Run the Generator

1.	Prepare the Configuration File:
Create a YAML file (e.g., config.yaml) using one of the configuration examples above.

2. Run the Generator:

```bash
python workload_generator.py --config config.yaml [--verbose]
```
The generator writes the operations, runbook, clustering data, and other outputs to the specified workload_dir.

3. Generate the Animation:

```bash
python viz_workload.py --workload_dir workload/demo2d_clustered_drift
```