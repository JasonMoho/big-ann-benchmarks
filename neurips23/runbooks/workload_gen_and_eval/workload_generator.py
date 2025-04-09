#!/usr/bin/env python3
"""
Dynamic Workload Generator for Streaming Benchmarks

This module generates a sequence of operations (insert, delete, query)
for vector search workloads. It loads the dataset (base vectors and queries)
by dataset name (from benchmark.datasets) if provided in the config; otherwise,
it loads from user-specified files. The base vectors are clustered using Faiss
(with no dependency on SciPy) to drive stratified sampling and form the initial
resident set. Each operation is saved as a separate .npy file in an "operations"
folder. A runbook is recorded and saved in both JSON and YAML. Ground truth for
each query operation is computed immediately using Faiss, and timing information
is recorded in the runbook.

Usage:
  python neurips23/streaming/workload_generator.py --config config.yaml [--verbose]
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
import faiss

from benchmark.datasets import DATASETS

# === Default Constants ===
DEFAULT_WORKLOAD_DIR = "workload"
DEFAULT_METRIC = "l2"
DEFAULT_INSERT_RATIO = 0.5
DEFAULT_DELETE_RATIO = 0.0
DEFAULT_QUERY_RATIO = 0.5  # Must add to 1.0.
DEFAULT_UPDATE_BATCH_SIZE = 100
DEFAULT_QUERY_BATCH_SIZE = 50
DEFAULT_NUM_OPERATIONS = 1000
DEFAULT_INITIAL_SIZE = 10000
DEFAULT_CLUSTER_SIZE = 100
DEFAULT_CLUSTER_SAMPLE_DISTRIBUTION = "uniform"
DEFAULT_QUERY_CLUSTER_SAMPLE_DISTRIBUTION = "uniform"
DEFAULT_SEED = 1738
GT_K = 100  # Number of nearest neighbors for ground truth computation

VALID_SAMPLE_DISTRIBUTIONS = ["uniform", "clustered_drift", "clustered_random"]

logger = logging.getLogger("WorkloadGenerator")

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_from_name(dataset_name: str) -> (np.ndarray, Optional[np.ndarray]):
    ds = DATASETS[dataset_name]()
    base_vectors = ds.get_dataset()  # Expects a NumPy array
    queries = ds.get_queries() if hasattr(ds, "get_queries") else None
    return base_vectors, queries


class UniformSampler:
    """Uniformly samples indices from a NumPy array."""

    @staticmethod
    def sample(pool: np.ndarray, size: int) -> np.ndarray:
        if pool.size == 0:
            return np.array([], dtype=int)
        size = min(size, pool.size)
        perm = np.random.permutation(pool.size)
        return pool[perm[:size]]


class StratifiedClusterSampler:
    """
    Stratified sampler based on cluster assignments.

    Expects:
      - assignments: 1D array of cluster IDs.
      - centroids: 2D array of centroids.
    """

    def __init__(self, assignments: np.ndarray, centroids: np.ndarray,
                 sample_distribution: str = "clustered_random") -> None:
        self.assignments = assignments
        self.centroids = centroids
        self.n_clusters = centroids.shape[0]
        unique = np.unique(assignments)
        self.root_cluster = int(np.random.choice(unique))
        self.cluster_ranks = self._compute_cluster_ranks(self.root_cluster)

        self.upper_random = self.n_clusters
        if sample_distribution == "clustered_random":
            self.fixed_ranks = False
        elif sample_distribution == "clustered_drift":
            self.fixed_ranks = False
            self.upper_random = 5
        else:
            raise ValueError(f"Invalid sample distribution: {sample_distribution}")

    def _compute_cluster_ranks(self, root_cluster: int) -> np.ndarray:
        root = self.centroids[root_cluster]
        distances = np.linalg.norm(self.centroids - root, axis=1)
        return np.argsort(distances)

    def update_ranks(self, new_root: int = -1) -> None:
        logger.debug("Updating cluster ranks; new root: %d", new_root)

        # select random if not given
        if new_root == -1:
            new_root = np.random.randint(0, self.n_clusters, 1)

        self.root_cluster = new_root
        self.cluster_ranks = self._compute_cluster_ranks(new_root)

    def sample(self, pool: np.ndarray, size: int, update_ranks: bool = True) -> np.ndarray:
        pool_clusters = self.assignments[pool]
        ordered = [c for c in self.cluster_ranks if c in np.unique(pool_clusters)]
        collected: List[np.ndarray] = []
        num_collected = 0
        for cluster in ordered:
            indices = pool[pool_clusters == cluster]
            if indices.size == 0:
                continue
            n_to_sample = min(size - num_collected, indices.size)
            sampled = UniformSampler.sample(indices, n_to_sample)
            collected.append(sampled)
            num_collected += sampled.size
            if num_collected >= size:
                break
        result = np.concatenate(collected) if collected else np.array([], dtype=int)
        if update_ranks and len(ordered) > 1 and not self.fixed_ranks:
            high = min(len(ordered), self.upper_random)
            self.update_ranks(ordered[np.random.randint(1, high, 1)[0]])
        return np.unique(result)

class DynamicWorkloadGenerator:
    """
    Generates a dynamic workload on a vector search dataset.

    Process:
      1. Load dataset (by name or from file paths).
      2. Cluster the base vectors using Faiss (or reuse saved clustering).
         Clustering is saved to <workload_dir>/clustered_index.npy.
      3. Sample an initial resident set.
      4. Generate operations (insert, delete, query) saved as .npy files.
      5. For each query op, compute ground truth on the currently resident vectors.
      6. Save a runbook (JSON and YAML) that records operation info and timing.

    INFO-level logs show a one-line summary per operation.
    """

    def __init__(
            self,
            workload_dir: Union[str, Path],
            metric: str,
            insert_ratio: float,
            delete_ratio: float,
            query_ratio: float,
            update_batch_size: int,
            query_batch_size: int,
            num_operations: int,
            initial_size: int,
            cluster_size: int,
            update_sample_distribution: str,
            query_sample_distribution: str,
            seed: int,
            dataset: Optional[str] = None,
            base_vectors_file: Optional[str] = None,
            queries_file: Optional[str] = None,
            initial_clustering_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.workload_dir = Path(workload_dir)
        self.metric = metric.lower()
        self.insert_ratio = insert_ratio
        self.delete_ratio = delete_ratio
        self.query_ratio = query_ratio
        self.update_batch_size = update_batch_size
        self.query_batch_size = query_batch_size
        self.num_operations = num_operations
        self.initial_size = initial_size
        self.cluster_size = cluster_size
        self.update_sample_distribution = update_sample_distribution.lower()
        self.query_sample_distribution = query_sample_distribution.lower()
        self.seed = seed
        # Default clustering file path if not provided.
        if initial_clustering_path:
            self.initial_clustering_path = Path(initial_clustering_path)
        else:
            self.initial_clustering_path = self.workload_dir / "clustered_index.npy"

        # Load dataset by name or file.
        if dataset:
            base_vectors, queries = load_dataset_from_name(dataset)
        else:
            if base_vectors_file is None:
                raise ValueError("Either 'dataset' or 'base_vectors_file' must be provided in the config.")
            base_vectors = np.load(base_vectors_file)
            queries = np.load(queries_file) if queries_file else None
        self.base_vectors = base_vectors
        self.queries = queries

        np.random.seed(self.seed)
        random.seed(self.seed)
        self.workload_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir = self.workload_dir / "operations"
        self.operations_dir.mkdir(exist_ok=True)
        self.n_vectors = self.base_vectors.shape[0]
        self.all_ids = np.arange(self.n_vectors)
        self.resident_set = np.zeros(self.n_vectors, dtype=bool)
        self.runbook: Dict[str, Any] = {"operations": {}}
        self.resident_history: List[np.ndarray] = []

    def validate_parameters(self) -> None:
        if self.metric not in ["l2", "ip"]:
            raise ValueError("Metric must be 'l2' or 'ip'.")
        total = self.insert_ratio + self.delete_ratio + self.query_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("The sum of insert, delete, and query ratios must equal 1.")
        if self.update_batch_size <= 0 or self.query_batch_size <= 0 or self.num_operations <= 0:
            raise ValueError("Batch sizes and number of operations must be positive.")
        if self.initial_size <= 0 or self.cluster_size <= 0:
            raise ValueError("Initial size and cluster size must be positive.")
        
        if self.update_sample_distribution not in VALID_SAMPLE_DISTRIBUTIONS:
            raise ValueError(f"Invalid update sample distribution: {self.update_sample_distribution}. "
                             f"Valid options are: {VALID_SAMPLE_DISTRIBUTIONS}")
        if self.query_sample_distribution not in VALID_SAMPLE_DISTRIBUTIONS:
            raise ValueError(f"Invalid query cluster sample distribution: {self.query_sample_distribution}. "
                             f"Valid options are: {VALID_SAMPLE_DISTRIBUTIONS}")

    def initialize_clustered_index(self) -> (np.ndarray, np.ndarray):
        """
        Clusters the base vectors using Faiss clustering.
        If a clustering file exists, it is loaded.

        Returns:
          assignments: 1D array of cluster IDs.
          centroids: 2D array of centroids.
        The clustering time is recorded.
        """
        start_time = time.time()
        if self.initial_clustering_path.exists():
            logger.info("Reusing existing clustering from '%s'", self.initial_clustering_path)
            data = np.load(self.initial_clustering_path, allow_pickle=True).item()
            assignments = data["assignments"]
            centroids = data["centroids"]
        else:
            n_clusters = max(2, self.n_vectors // self.cluster_size)
            logger.info("Clustering %d vectors into %d clusters using Faiss...", self.n_vectors, n_clusters)
            d = self.base_vectors.shape[1]
            X = self.base_vectors.astype(np.float32)
            clustering = faiss.Clustering(d, n_clusters)
            clustering.niter = 20
            clustering.max_points_per_centroid = 10000000
            index_flat = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
            clustering.train(X, index_flat)
            centroids = faiss.vector_float_to_array(clustering.centroids).reshape(n_clusters, d)
            index_centroids = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
            index_centroids.add(centroids)
            _, assignments = index_centroids.search(X, 1)
            assignments = assignments.flatten()
            np.save(self.initial_clustering_path, {"assignments": assignments, "centroids": centroids})
            logger.info("Clustering completed and saved to '%s'", self.initial_clustering_path)
        elapsed = time.time() - start_time
        self.runbook["clustering_time"] = elapsed
        logger.info("Clustering elapsed time: %.2f s", elapsed)
        return assignments, centroids

    def initialize_workload(self) -> None:
        self.validate_parameters()
        self.assignments, self.centroids = self.initialize_clustered_index()
        if self.update_sample_distribution in ["clustered_drift", "clustered_random"]:
            self.sampler = StratifiedClusterSampler(self.assignments, self.centroids, self.update_sample_distribution)
        else:
            self.sampler = UniformSampler
        if self.query_sample_distribution in ["clustered_drift", "clustered_random"] and self.queries is not None:
            d = self.queries.shape[1]
            index_centroids = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
            index_centroids.add(self.centroids)
            _, q_assignments = index_centroids.search(self.queries.astype(np.float32), 1)
            q_assignments = q_assignments.flatten()
            self.query_sampler = StratifiedClusterSampler(q_assignments, self.centroids, self.query_sample_distribution)
        else:
            self.query_sampler = UniformSampler

        non_resident = self.all_ids[~self.resident_set]
        init_indices = self.sampler.sample(non_resident, self.initial_size)
        self.resident_set[init_indices] = True
        np.save(self.workload_dir / "initial_indices.npy", init_indices)
        np.save(self.workload_dir / "base_vectors.npy", self.base_vectors)
        if self.queries is not None:
            np.save(self.workload_dir / "query_vectors.npy", self.queries)
        self.runbook["parameters"] = {
            "n_base_vectors": self.n_vectors,
            "vector_dimension": self.base_vectors.shape[1],
            "metric": self.metric,
            "insert_ratio": self.insert_ratio,
            "delete_ratio": self.delete_ratio,
            "query_ratio": self.query_ratio,
            "update_batch_size": self.update_batch_size,
            "query_batch_size": self.query_batch_size,
            "num_operations": self.num_operations,
            "initial_size": self.initial_size,
            "cluster_size": self.cluster_size,
            "update_sample_distribution": self.update_sample_distribution,
            "query_sample_distribution": self.query_sample_distribution,
            "seed": self.seed,
        }
        self.runbook["initialize"] = {"size": self.initial_size}
        logger.info("Workload initialization complete. Initial resident set size: %d", int(np.sum(self.resident_set)))

    def sample_indices(self, size: int, op_type: str) -> np.ndarray:
        if op_type == "insert":
            pool = self.all_ids[~self.resident_set]
        elif op_type == "delete":
            pool = self.all_ids[self.resident_set]
        elif op_type == "query":
            pool = np.arange(self.queries.shape[0]) if self.queries is not None else self.all_ids[~self.resident_set]
        else:
            raise ValueError(f"Invalid op type: {op_type}")
        if pool.size == 0:
            return np.array([], dtype=int)
        if op_type in ["insert", "delete"]:
            return self.sampler.sample(pool, size)
        else:
            return self.query_sampler.sample(pool, size)

    def compute_ground_truth_for_query(self, q_indices: np.ndarray) -> Dict[str, Any]:
        """
        Computes ground truth for the queries corresponding to q_indices over the
        current resident set. Returns a dict with the ground truth time and (optionally)
        other info.
        """
        # Use the currently resident vectors.
        current_ids = np.where(self.resident_set)[0]
        current_vectors = self.base_vectors[current_ids]
        # Select the queries for this op.
        if self.queries is not None:
            query_vectors = self.queries[q_indices]
        else:
            query_vectors = self.base_vectors[q_indices]
        d = self.base_vectors.shape[1]
        gt_index = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
        gt_index.add(current_vectors)
        start = time.time()
        _, gt_ids = gt_index.search(query_vectors.astype(np.float32), GT_K)
        elapsed = time.time() - start
        return {"gt_time": elapsed, "gt_ids": gt_ids}

    def generate_workload(self) -> None:
        overall_start = time.time()
        self.initialize_workload()
        op_times = {"insert": 0.0, "delete": 0.0, "query": 0.0}
        counts = {"insert": 0, "delete": 0, "query": 0}

        n_clusters = int(self.assignments.max() + 1)
        cluster_sizes = np.bincount(self.assignments, minlength=n_clusters)

        for i in range(self.num_operations):
            op_start = time.time()
            op_type = np.random.choice(
                ["insert", "delete", "query"],
                p=[self.insert_ratio, self.delete_ratio, self.query_ratio]
            )
            entry: Dict[str, Any] = {}
            if op_type == "insert":
                indices = self.sample_indices(self.update_batch_size, "insert")
                if indices.size == 0:
                    logger.info("Op %d [INSERT]: No indices available. Skipping generation.", i)
                    break
                self.resident_set[indices] = True
                counts["insert"] += 1
                entry = {"operation": "insert",
                         "sample_size": int(indices.size),
                         "n_resident": int(np.sum(self.resident_set))}
                np.save(self.operations_dir / f"{i}.npy", indices)
            elif op_type == "delete":
                indices = self.sample_indices(self.update_batch_size, "delete")
                if indices.size == 0:
                    logger.info("Op %d [DELETE]: No indices available. Terminating generation.", i)
                    break
                self.resident_set[indices] = False
                counts["delete"] += 1
                entry = {"operation": "delete",
                         "sample_size": int(indices.size),
                         "n_resident": int(np.sum(self.resident_set))}
                np.save(self.operations_dir / f"{i}.npy", indices)
            elif op_type == "query":
                q_indices = self.sample_indices(self.query_batch_size, "query")
                if q_indices.size == 0:
                    logger.info("Op %d [QUERY]: No query indices available. Terminating generation.", i)
                    break
                counts["query"] += 1
                entry = {"operation": "query",
                         "sample_size": int(q_indices.size),
                         "n_resident": int(np.sum(self.resident_set))}
                np.save(self.operations_dir / f"{i}.npy", q_indices)
                # Compute ground truth for this query op.
                gt_info = self.compute_ground_truth_for_query(q_indices)
                entry["gt_time"] = gt_info["gt_time"]
            else:
                raise ValueError(f"Unknown op type: {op_type}")

            self.runbook["operations"][i] = entry
            op_elapsed = time.time() - op_start
            op_times[op_type] += op_elapsed


            # Compute resident fractions using vectorized operations.
            res_ids = self.all_ids[self.resident_set]
            if res_ids.size > 0:
                counts_arr = np.bincount(self.assignments[res_ids], minlength=n_clusters)
                fractions = counts_arr / cluster_sizes
                self.resident_history.append(fractions)

            logger.info("Op %d [%s]: resident_size=%d, sample_size=%d, op_time=%.3f s%s",
                        i, op_type.upper(), res_ids.shape[0], entry["sample_size"], op_elapsed,
                        f", gt_time=%.3f s" % entry["gt_time"] if op_type == "query" and "gt_time" in entry else "")

        total_ops = i + 1
        summary = {
            "n_inserts": counts["insert"],
            "n_deletes": counts["delete"],
            "n_queries": counts["query"],
            "n_operations": total_ops,
            "average_op_time": {op: (op_times[op] / counts[op] if counts[op] > 0 else 0) for op in op_times},
            "total_generation_time": time.time() - overall_start
        }
        self.runbook["summary"] = summary
        logger.info("Operation timings: %s", summary["average_op_time"])
        logger.info("Operations: %d (insert: %d, delete: %d, query: %d)",
                    total_ops,
                    counts["insert"],
                    counts["delete"],
                    counts["query"])
        logger.info("Total workload generation time: %.2f s", summary["total_generation_time"])

        # Save resident history heatmap.
        try:
            import matplotlib.pyplot as plt
            # Suppress matplotlib debug logs.
            import logging as mpl_logging
            mpl_logging.getLogger("matplotlib").setLevel(mpl_logging.WARNING)
            heatmap = np.array(self.resident_history).T
            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(heatmap, cmap="viridis", aspect="auto")
            ax.set_xlabel("Operation Number")
            ax.set_ylabel("Cluster ID")
            fig.colorbar(cax, label="Resident Fraction")
            plt.tight_layout()
            heatmap_file = self.workload_dir / "resident_history.png"
            plt.savefig(heatmap_file)
            plt.close()
            logger.info("Resident history heatmap saved to '%s'", heatmap_file)
        except ImportError:
            logger.info("matplotlib not installed; skipping resident history plot.")

        runbook_json_path = self.workload_dir / "runbook.json"
        with runbook_json_path.open("w") as f:
            json.dump(self.runbook, f, indent=4)
        logger.info("Runbook saved to '%s'", runbook_json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic workload runbook using YAML configuration."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[WorkloadGenerator] %(levelname)s: %(message)s"
    )
    # Suppress unnecessary matplotlib logging.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.verbose:
        logger.debug("Verbose mode enabled.")

    config = load_yaml_config(args.config)
    workload_dir = config.get("workload_dir", DEFAULT_WORKLOAD_DIR)

    generator = DynamicWorkloadGenerator(
        dataset=config.get("dataset", None),
        workload_dir=workload_dir,
        metric=config.get("metric", DEFAULT_METRIC),
        insert_ratio=config.get("insert_ratio", DEFAULT_INSERT_RATIO),
        delete_ratio=config.get("delete_ratio", DEFAULT_DELETE_RATIO),
        query_ratio=config.get("query_ratio", DEFAULT_QUERY_RATIO),
        update_batch_size=config.get("update_batch_size", DEFAULT_UPDATE_BATCH_SIZE),
        query_batch_size=config.get("query_batch_size", DEFAULT_QUERY_BATCH_SIZE),
        num_operations=config.get("num_operations", DEFAULT_NUM_OPERATIONS),
        initial_size=config.get("initial_size", DEFAULT_INITIAL_SIZE),
        cluster_size=config.get("cluster_size", DEFAULT_CLUSTER_SIZE),
        update_sample_distribution=config.get("update_sample_distribution", DEFAULT_CLUSTER_SAMPLE_DISTRIBUTION),
        query_sample_distribution=config.get("query_sample_distribution",
                                                     DEFAULT_QUERY_CLUSTER_SAMPLE_DISTRIBUTION),
        seed=config.get("seed", DEFAULT_SEED),
        initial_clustering_path=config.get("initial_clustering_path", None),
    )

    logger.info("Starting workload generation...")
    overall_start = time.time()
    generator.generate_workload()
    total_elapsed = time.time() - overall_start
    logger.info("Workload generated in %.2f s", total_elapsed)


if __name__ == "__main__":
    main()