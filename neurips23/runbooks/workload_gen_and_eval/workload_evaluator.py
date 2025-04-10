from __future__ import absolute_import
import os
import time
import numpy as np
import yaml
import json

from benchmark.algorithms.base_runner import BaseRunner
from benchmark.datasets import DATASETS

from neurips23.streaming.faiss_ivf.faiss_ivf import FaissIVF
from neurips23.streaming.faiss_flat.faiss_flat import FaissFlat


def select_index(index_type: str):
    if index_type == "faiss-ivf":
        index_params = {"n_clusters": 50, "k": 10}  # adjust default parameters as needed
        return FaissIVF(metric="l2", index_params=index_params)
    elif index_type == "faiss-flat":
        index_params = {"k": 10}
        return FaissFlat(metric="l2", index_params=index_params)
    raise ValueError(f"Index '{index_type}' not found in the factory.")


class WorkloadRunner(BaseRunner):
    @classmethod
    def build(cls, algo, workload_dir, build_params=None):
        """
        Build the algorithm using a dataset loaded from the workload directory.
        The dataset name is obtained from the runbook.
        """
        t0 = time.time()
        runbook_path = os.path.join(workload_dir, "runbook.json")
        with open(runbook_path, "r") as f:
            runbook = json.load(f)
        dataset_name = runbook.get("parameters").get("dataset")
        if not dataset_name:
            raise ValueError("Runbook must contain a 'dataset' key to specify the dataset name.")
        ds = DATASETS[dataset_name]()

        # Get initial resident set.
        initial_indices_path = os.path.join(workload_dir, "initial_indices.npy")
        if not os.path.exists(initial_indices_path):
            raise FileNotFoundError(f"Initial indices file not found: {initial_indices_path}")
        initial_indices = np.load(initial_indices_path)
        initial_vectors = ds.get_dataset()[initial_indices]

        # Use the actual number of initial vectors for max_pts.
        algo.setup(initial_vectors.dtype, initial_vectors.shape[0], initial_vectors.shape[1])

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
        """

        gt_ids = gt_ids[:, :k]  # Ensure gt_ids is limited to k nearest neighbors

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
        Process each operation from the runbook using the provided index algorithm.

        Changes:
         - Operations are ordered based on numeric keys.
         - Results for each index are saved in a subdirectory named for the index_config_name.
         - Ground truth is loaded from a global precomputed file if available, otherwise recall is skipped.
         - The call to get_additional() has been removed.
        """
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Determine the index config name and create a sub-directory.
        # Use runbook parameters "index_config_name" if available, or fall back to str(algo).
        index_config_name = runbook.get("parameters", {}).get("index_config_name", str(algo))
        results_subdir = os.path.join(results_dir, index_config_name)
        os.makedirs(results_subdir, exist_ok=True)

        dataset_name = runbook.get("parameters").get("dataset")
        if not dataset_name:
            raise ValueError("Runbook must include a 'dataset' key with the dataset name.")
        ds = DATASETS[dataset_name]()
        X = ds.get_queries()
        XB = ds.get_dataset()

        # Build the index.
        cls.build(algo, workload_dir)
        print(f"Got {X.shape[0]} queries")

        best_search_time = float("inf")
        best_results = None
        search_times = []

        # Order the operations by numeric key.
        operations = runbook.get("operations", {})
        sorted_keys = sorted(operations.keys(), key=lambda x: int(x))
        for step, op_key in enumerate(sorted_keys):
            entry = operations[op_key]
            print(f"Operation {op_key}: {entry}")
            step_start = time.time()
            op = entry.get("operation", "").lower()

            # Load the operation's IDs.
            op_file = os.path.join(workload_dir, "operations", f"{op_key}.npy")
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
                    results = algo.get_range_results()
                else:
                    raise NotImplementedError(f"Search type {search_type} not available.")
                search_time = time.time() - step_start
                search_times.append(search_time)
                result_dists, result_ids = results
                np.save(os.path.join(results_subdir, f"{step}_result_dists.npy"), result_dists)
                np.save(os.path.join(results_subdir, f"{step}_result_ids.npy"), result_ids)
                # Compute recall only if a ground truth file exists.

                # load the gt ids for the current operation
                gt_file = os.path.join(workload_dir, "operations", f"{op_key}_gt_ids.npy")
                if os.path.exists(gt_file):
                    gt_ids = np.load(gt_file)
                else:
                    gt_ids = None

                if gt_ids is not None:
                    try:
                        recall = cls.compute_recall(result_ids, gt_ids, k)
                        entry["recall"] = recall
                        print(f"Operation {op_key} recall: {recall}")
                    except Exception as e:
                        print(f"Warning: Could not compute recall for operation {op_key}: {e}")
                if search_time < best_search_time:
                    best_search_time = search_time
                    best_results = results
            else:
                raise NotImplementedError(f"Invalid runbook operation {op}:")

            entry["latency"] = time.time() - step_start
            print(f"Operation {op_key} took {entry['latency']} seconds.")

        # Save the updated runbook with timings.
        runbook_output_path = os.path.join(results_subdir, "runbook_results.json")
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
        return (attrs, best_results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run WorkloadRunner evaluation.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

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

    with open(runbook_file, "r") as f:
        runbook = yaml.safe_load(f)

    if "dataset" not in runbook.get("parameters", {}):
        raise ValueError("The runbook must include a 'dataset' key to specify the dataset name.")

    indexes = config.get("indexes", [])
    for index_conf in indexes:

        # Use two config keys: one for directory labeling and one for selecting the index. Fallback to index_name if not provided.
        index_config_name = index_conf.get("index_config_name", index_conf.get("index_name"))
        index_type = index_conf.get("index_type", index_conf.get("index_name"))
        build_params = index_conf.get("build_params", {})
        search_params = index_conf.get("search_params", {})

        print(f"=== Evaluating index config: {index_config_name} (type: {index_type}) ===")
        algo = select_index(index_type)
        build_time = WorkloadRunner.build(algo, workload_dir, build_params)
        print(f"Build time for index '{index_config_name}': {build_time:.4f} seconds.")
        attrs, best_results = WorkloadRunner.run_task(
            algo, workload_dir, distance, k, run_count, search_type,
            runbook, search_params, results_dir=results_dir, ground_truth_file=ground_truth_file
        )
        print(f"Results for index config {index_config_name}:")
        print(attrs)
        print("Best search results obtained (saved in the results directory).")
        print("=======================================")


if __name__ == "__main__":
    main()