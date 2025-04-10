# Workload Generator & Evaluator

## Overview

This project provides two key modules for streaming benchmarks on vector search datasets:
	
1.	Dynamic Workload Generator
   - Generates a series of operations (insert, delete, query) on a vector dataset.
   - Supports loading datasets by name (using benchmark.datasets) or from user-specified files.
   - Applies Faiss clustering to the base vectors to create a stratified initial resident set and guide sampling.
   - Saves each operation as a separate .npy file inside an operations folder.
   - Logs timings, computes ground truth (using Faiss), and records a detailed runbook (in both JSON and YAML) with all operation metrics.
	
2.	Streaming Workload Evaluator
- Reads the runbook generated above and processes the operations using an index.
- Builds a default Faiss Flat index (if no external index is provided) or uses a supplied index.
- Computes or loads exhaustive ground truth for query vectors and evaluates recall and latency for query operations.

⸻

## Configuration Files

Two YAML configuration files are required—one for generating the workload and one for evaluating it.

### Workload Generator (e.g., config.yaml)

Below is an example configuration:
```yaml
# Dataset selection: use a dataset name from benchmark.datasets OR provide file paths.
dataset: "random-2d"  
# Alternatively, use file-based loading:
# base_vectors_file: "data/base_vectors.npy"
# queries_file: "data/query_vectors.npy"

workload_dir: "workload/demo2d_clustered_drift"
metric: "l2"                                     # Options: "l2" or "ip"

# Operation Ratios (sum must equal 1)
insert_ratio: 0.5
delete_ratio: 0.25
query_ratio: 0.25

# Batch sizes and number of operations
update_batch_size: 100                           # Number of vectors for insert/delete operations per op.
query_batch_size: 10                             # Number of queries per op.
num_operations: 1000                             # Total operations to generate.
initial_size: 1000                               # Size of the initial resident set.

# Clustering and Sampling Parameters
cluster_size: 100                                # Used to determine the number of clusters.
update_sample_distribution: "clustered_drift"    # Options: "uniform", "clustered_drift", "clustered_random".
query_sample_distribution: "clustered_drift"     # Options: "uniform", "clustered_drift", "clustered_random".

seed: 1738                                       # Random seed for reproducibility.
```

### Evaluator (e.g., eval_config.yaml)

Below is an example configuration for the evaluator:

```yaml
workload_dir: "workload/demo2d_clustered_drift"
metric: "l2"                 # Options: "l2" or "ip"
k: 10                        # Number of nearest neighbors to use for queries.
use_precomputed_gt: true     # Whether to use precomputed ground truth if available.
```

⸻

### Detailed Parameter Reference

The following table summarizes the key configuration parameters for both modules:

#### Workload Generator

| Parameter                  | Type    | Default           | Allowed Values / Notes                                        | Description                                                                                          |
|----------------------------|---------|-------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| dataset                    | string  | None              | Any valid dataset name from `benchmark.datasets`              | Specifies the dataset to be used. If provided, files aren’t required.                                |
| base_vectors_file          | string  | N/A               | File path to a `.npy` file.                                   | Used if `dataset` is not provided—path to the base vectors.                                          |
| queries_file               | string  | N/A               | File path to a `.npy` file.                                   | Used if `dataset` is not provided—path to query vectors.                                             |
| workload_dir               | string  | "workload"        | Any valid directory path.                                     | Output directory where operations, runbook, and clustering data are stored.                          |
| metric                     | string  | "l2"              | "l2", "ip"                                                   | Metric used for distance computation.                                                              |
| insert_ratio               | float   | 0.5               | Should be in [0, 1].                                          | Proportion of insert operations. Must combine with others to equal 1.                                |
| delete_ratio               | float   | 0.0               | Should be in [0, 1].                                          | Proportion of delete operations.                                                                   |
| query_ratio                | float   | 0.5               | Should be in [0, 1].                                          | Proportion of query operations. Must combine with others to equal 1.                                 |
| update_batch_size          | integer | 100               | Must be a positive integer.                                   | Number of vectors inserted or deleted per operation.                                               |
| query_batch_size           | integer | 50                | Must be a positive integer.                                   | Number of queries processed per query operation.                                                   |
| num_operations             | integer | 1000              | Must be a positive integer.                                   | Total number of operations to generate.                                                            |
| initial_size               | integer | 10000             | Must be a positive integer.                                   | Size of the resident vector set initialized prior to operations.                                   |
| cluster_size               | integer | 100               | Must be a positive integer.                                   | Used to determine the number of clusters for stratified sampling.                                  |
| update_sample_distribution | string  | "clustered_random"| Options: "uniform", "clustered_drift", "clustered_random"     | Determines the sampling strategy for insert/delete operations.                                     |
| query_sample_distribution  | string  | "clustered_random"| Options: "uniform", "clustered_drift", "clustered_random"     | Determines the sampling strategy for query operations.                                             |
| seed                       | integer | 1738              | Any integer value.                                            | Random seed used for reproducibility.                                                              |
| initial_clustering_path    | string  | Derived from workload_dir | File path to store or load the clustering data.        | Optional path for loading precomputed clustering; if absent, a default path is used.                 |

#### Evaluator

| Parameter          | Type    | Default | Allowed Values / Notes                  | Description                                                                                         |
|--------------------|---------|---------|-----------------------------------------|-----------------------------------------------------------------------------------------------------|
| workload_dir       | string  | N/A     | Any valid directory containing workload files | Directory where the runbook, operations, and vector data are stored.                                |
| metric             | string  | "l2"    | "l2", "ip"                             | The metric used by Faiss for indexing and search operations.                                        |
| k                  | integer | 10      | Must be a positive integer              | Number of nearest neighbors to retrieve per query.                                                 |
| use_gpu            | boolean | false   | true or false                           | Indicates whether to use GPU resources for Faiss (requires GPU support and proper drivers).         |
| use_precomputed_gt | boolean | true    | true or false                           | Whether to load precomputed ground truth, if available, to speed up evaluation.                     |

⸻

## Usage

### Running the Workload Generator
#### 1.	Prepare the configuration file (e.g., config.yaml)
Follow the format above to specify your dataset (or provide file paths), ratios, batching parameters, and other settings.

### 2. Execute the generator

```bash
python neurips23/streaming/workload_generator.py --config config.yaml [--verbose]
```

With --verbose, detailed debug logging will be enabled.

### 3.	Outputs
- Operations: Saved as individual .npy files in the <workload_dir>/operations folder.
- Runbook: Recorded in both runbook.json and runbook.yaml in the specified workload directory.
- Clustering Data: Saved as <workload_dir>/clustered_index.npy (unless an existing file is provided).

### Running the Workload Evaluator
#### 1.	Prepare the evaluation configuration file (e.g., eval_config.yaml)
Specify the workload directory and search_parameters

#### 2. Execute the evaluator:

```bash
python neurips23/streaming/workload_evaluator.py --config eval_config.yaml
```
The evaluator processes each operation—updates the index, runs queries, and computes recall against the ground truth.

#### 3.	Outputs:
- Evaluation Metrics: Logged for each operation (latency, query recall, etc.).
- Updated Runbook: The runbook is augmented with evaluation metrics.
- Query Results: For query operations, predicted neighbor IDs are saved to <operation_key>_pred_ids.npy and distances to <operation_key>_pred_dists.npy.

⸻

### Output Structure

Within the workload_dir, expect to find:
- operations/ folder: Contains .npy files for each operation.
- runbook.json: Detailed logs of operation parameters, timings, and evaluation results.
- Clustering Data: clustered_index.npy, if generated.
- Plots (Optional): A heatmap (e.g., resident_history.png) visualizing the evolution of resident fractions across clusters (requires Matplotlib).

⸻
