from __future__ import absolute_import
import os
import time
import numpy as np
import yaml
import json
import pickle

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS

from neurips23.streaming.faiss_ivf.faiss_ivf import FaissIVF
from neurips23.streaming.faiss_flat.faiss_flat import FaissFlat
def select_index(index_name: str):
    if index_name == "faiss-ivf":
         index_params = {"n_clusters": 50, "k": 10}  # adjust default parameters as needed
         return FaissIVF(metric="l2", index_params=index_params)
    elif index_name == "faiss-flat":
        index_params = {"k": 10}
        return FaissFlat(metric="l2", index_params=index_params)
    # other index cases...
    # For example, if index_name == "scann", return the Scann index, etc.
    raise ValueError(f"Index '{index_name}' not found in the factory.")

class WorkloadRunner(BaseRunner):
    """
    A runner that extends BaseRunner to support a runbook with operations:
      - "insert": calls algo.insert(data, ids)
      - "delete": calls algo.delete(ids)
      - "replace": calls algo.replace(data, ids) (or similar)
      - "search": applies search parameters, performs a query, saves result distances and IDs,
                  and (if a ground truth file is provided) computes recall.

    The configuration file (YAML) should contain:
      - workload_dir: Directory containing base_vectors.npy, query_vectors.npy, etc.
      - distance: A descriptor (e.g. "l2")
      - k: Number of neighbors to retrieve
      - run_count: Number of evaluation runs (for reporting purposes)
      - search_type: "knn" or "range"
      - runbook_file: Path to the runbook YAML file (which must include a "dataset" key)
      - ground_truth_file: (Optional) Path to a .npy file with precomputed ground truth
      - results_dir: Directory where search results and the updated runbook will be saved
      - indexes: A list of index configurations. Each index entry must include:
           - index_name: Name of the index to select
           - build_params: dict of attributes to set after building the index
           - search_params: dict of attributes to set before search operations.
    """

    @classmethod
    def build(cls, algo, workload_dir, build_params=None):
        """
        Build the algorithm using a dataset loaded from the workload directory.
        The dataset name is obtained from the runbook.

        Parameters:
          algo: The algorithm instance.
          workload_dir: Directory containing dataset files.
          build_params: Optional dict of additional algorithm attributes.

        Returns:
          Setup time in seconds.
        """
        t0 = time.time()
        # Load the runbook to extract the dataset name.
        runbook_path = os.path.join(workload_dir, "runbook.json")
        with open(runbook_path, "r") as f:
            runbook = json.load(f)
        dataset_name = runbook.get("parameters").get("dataset")
        if not dataset_name:
            raise ValueError("Runbook must contain a 'dataset' key to specify the dataset name.")
        ds = DATASETS[dataset_name]()

        # get initial resident set
        initial_indices_path = os.path.join(workload_dir, "initial_indices.npy")
        if not os.path.exists(initial_indices_path):
            raise FileNotFoundError(f"Initial indices file not found: {initial_indices_path}")

        initial_indices = np.load(initial_indices_path)
        initial_vectors = ds.get_dataset()[initial_indices]

        algo.setup(initial_vectors.dtype, -1, initial_vectors.shape[1])

        if build_params:
            for key, value in build_params.items():
                try:
                    setattr(algo, key, value)
                except Exception as e:
                    print(f"Warning: Could not set build parameter '{key}': {e}")

        print(f"Building index with {initial_vectors.shape[0]} vectors.")
        algo.insert(initial_vectors, initial_indices)
        print("Index built.")

        build_time = time.time() - t0
        return build_time

    @staticmethod
    def compute_recall(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
        """
        Compute average recall over queries.
        Recall for each query is defined as the number of overlapping elements between
        predicted and ground truth divided by k.

        Parameters:
          pred_ids: np.ndarray (n_queries x k) of predicted neighbor IDs.
          gt_ids: np.ndarray (n_queries x k) of ground truth neighbor IDs.
          k: Number of neighbors.

        Returns:
          The average recall.
        """
        n_queries = gt_ids.shape[0]
        recalls = []
        for i in range(n_queries):
            inter = np.intersect1d(pred_ids[i], gt_ids[i])
            recalls.append(len(inter) / k)
        return float(np.mean(recalls))

    @classmethod
    def run_task(cls, algo, workload_dir, distance, k, run_count, search_type,
                 runbook, search_params=None, results_dir="results"):
        """
        Run the workload task by processing each operation in the runbook.
        At the end, save the updated runbook (with timings) in the results directory.

        Parameters:
          algo: The algorithm instance.
          workload_dir: Directory containing dataset files.
          distance: Descriptor of the distance metric.
          k: Number of neighbors to retrieve.
          run_count: Number of runs (for reporting purposes).
          search_type: "knn" or "range".
          runbook: List of operation dictionaries from the runbook.
          search_params: Optional dict of attributes to set on algo before searching.
          results_dir: Directory where search results and the updated runbook will be saved.

        Returns:
          A tuple (attrs, best_results) where attrs is a summary dictionary and best_results
          is the result from the fastest search operation.
        """
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Get dataset name from the runbook and load the dataset.
        dataset_name = runbook.get("parameters").get("dataset")
        if not dataset_name:
            raise ValueError("Runbook must include a 'dataset' key with the dataset name.")
        ds = DATASETS[dataset_name]()
        X = ds.get_queries()
        XB = ds.get_dataset()

        cls.build(algo, workload_dir)

        print(f"Got {X.shape[0]} queries")

        best_search_time = float("inf")
        best_results = None
        search_times = []

        operations = runbook.get("operations", [])

        # Process each operation in the runbook.
        for step, op_id in enumerate(operations):
            entry = operations[op_id]
            print(step, entry)
            step_start = time.time()
            op = entry.get("operation", "").lower()

            # load operation ids
            op_file = os.path.join(workload_dir, "operations", f"{op_id}.npy")
            if not os.path.exists(op_file):
                raise FileNotFoundError(f"Operation file not found: {op_file}")

            ids = np.load(op_file)

            if op == "insert":
                algo.insert(XB[ids], ids)
            elif op == "delete":
                algo.delete(ids)
            elif op == "replace":
                raise NotImplementedError("Replace operation is not implemented.")
            elif op == "search":
                query_vectors = X[ids]
                if search_params:
                    for key, value in search_params.items():
                        try:
                            setattr(algo, key, value)
                        except Exception as e:
                            print(f"Warning: Could not set search parameter '{key}': {e}")
                if search_type == "knn":
                    algo.query(query_vectors, k)
                    results = algo.get_results()  # Expected to return (result_dists, result_ids)
                elif search_type == "range":
                    algo.range_query(X, k)
                    results = algo.get_range_results()  # Expected to return a tuple
                else:
                    raise NotImplementedError(f"Search type {search_type} not available.")
                search_time = time.time() - step_start
                search_times.append(search_time)
                result_dists, result_ids = results
                np.save(os.path.join(results_dir, f"{step}_result_dists.npy"), result_dists)
                np.save(os.path.join(results_dir, f"{step}_result_ids.npy"), result_ids)

                # check if gt for this operation exists
                gt_file = os.path.join(workload_dir, "ground_truth", f"{op_id}_gt.npy")
                if os.path.exists(gt_file):
                    gt = np.load(gt_file)
                else:
                    gt = None

                if gt is not None:
                    try:
                        recall = cls.compute_recall(result_ids, gt[:result_ids.shape[0]], k)
                        entry["recall"] = recall
                        print(f"Step {step + 1} recall: {recall}")
                    except Exception as e:
                        print(f"Warning: Could not compute recall for step {step + 1}: {e}")
                if search_time < best_search_time:
                    best_search_time = search_time
                    best_results = results
            else:
                raise NotImplementedError("Invalid runbook operation {}: ".format(op))
            entry["latency"] = time.time() - step_start
            print(f"Step {step + 1} took {time.time() - step_start}s.")

        # Save the updated runbook with timings into the results directory.
        runbook_output_path = os.path.join(results_dir, "runbook_results.json")
        with open(runbook_output_path, "w") as f:
            json.dump(runbook, f, indent=4)
        print(f"Updated runbook saved to {runbook_output_path}")

        attrs = {
            "best_search_time": best_search_time,
            "name": str(algo),
            "run_count": run_count,
            "distance": distance,
            "type": search_type,
            "k": int(k),
            "search_times": search_times,
        }
        additional = algo.get_additional()
        for key, value in additional.items():
            attrs[key] = value
        return (attrs, best_results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run WorkloadRunner evaluation.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    workload_dir = config.get("workload_dir")
    distance = config.get("distance", "l2")
    k = config.get("k", 10)
    run_count = config.get("run_count", 1)
    search_type = config.get("search_type", "knn")
    runbook_file = config.get("runbook_file")
    ground_truth_file = config.get("ground_truth_file")
    results_dir = config.get("results_dir", "results")

    # Load runbook.
    with open(runbook_file, "r") as f:
        runbook = yaml.safe_load(f)

    # The runbook must include a "dataset" key.
    if "dataset" not in runbook:
        raise ValueError("The runbook must include a 'dataset' key to specify the dataset name.")

    # Process each index configuration.
    indexes = config.get("indexes", [])
    for index_conf in indexes:
        index_name = index_conf.get("index_name", "default_index")
        build_params = index_conf.get("build_params", {})
        search_params = index_conf.get("search_params", {})

        print(f"=== Evaluating index: {index_name} ===")
        # Select the index using our selection function.
        algo = select_index(index_name)

        # Run the workload task.
        attrs, best_results = WorkloadRunner.run_task(
            algo, workload_dir, distance, k, run_count, search_type,
            runbook, search_params, ground_truth_file, results_dir
        )

        print(f"Results for index {index_name}:")
        print(attrs)
        print("Best search results obtained (saved in the results directory).")
        print("=======================================")


if __name__ == "__main__":
    main()