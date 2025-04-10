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

Run from the repo root directory with:
Usage:
  python3 -m neurips23.streaming.workload_generator --config config.yaml [--verbose]
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
import faiss

from benchmark.datasets import DATASETS

# === Default Constants ===
DEFAULT_WORKLOAD_DIR = "workload"
DEFAULT_METRIC = "l2"
DEFAULT_INSERT_RATIO = 0.5
DEFAULT_DELETE_RATIO = 0.0
DEFAULT_QUERY_RATIO = 0.5  # Sum must equal 1.0.
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
    """Load YAML configuration file from the given path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_from_name(dataset_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a dataset using the provided dataset name from benchmark.datasets."""
    ds = DATASETS[dataset_name]()
    base_vectors = ds.get_dataset()  # Expected to be a NumPy array.
    queries = ds.get_queries() if hasattr(ds, "get_queries") else None
    return base_vectors, queries


class UniformSampler:
    """Uniformly samples indices from a NumPy array."""

    @staticmethod
    def sample(pool: np.ndarray, size: int) -> np.ndarray:
        """Return a uniform random sample of indices from the given pool."""
        if pool.size == 0:
            return np.array([], dtype=int)
        sample_size = min(size, pool.size)
        perm = np.random.permutation(pool.size)
        return pool[perm[:sample_size]]


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
        unique_clusters = np.unique(assignments)
        self.root_cluster = int(np.random.choice(unique_clusters))
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
        """Compute cluster ranks based on distance from the root cluster centroid."""
        root = self.centroids[root_cluster]
        distances = np.linalg.norm(self.centroids - root, axis=1)
        return np.argsort(distances)

    def update_ranks(self, new_root: int) -> None:
        """Update cluster ranks, optionally selecting a new root cluster."""
        logger.debug("Updating cluster ranks; new root: %d", new_root)
        self.root_cluster = new_root
        self.cluster_ranks = self._compute_cluster_ranks(new_root)

    def sample(self, pool: np.ndarray, size: int, update_ranks: bool = True) -> np.ndarray:
        """
        Sample indices from the pool based on stratified cluster assignments.

        Parameters:
            pool: Available indices to sample from.
            size: Number of indices to sample.
            update_ranks: Whether to update cluster rankings after sampling.

        Returns:
            Unique indices sampled from the pool.
        """
        pool_clusters = self.assignments[pool]
        unique_clusters_in_pool = np.unique(pool_clusters) # get clusters that are present in the sample pool
        ordered_clusters = [c for c in self.cluster_ranks if c in unique_clusters_in_pool]
        collected_samples: List[np.ndarray] = []
        num_collected = 0
        for cluster in ordered_clusters:
            cluster_indices = pool[pool_clusters == cluster]
            if cluster_indices.size == 0:
                continue
            n_to_sample = min(size - num_collected, cluster_indices.size)
            sampled = UniformSampler.sample(cluster_indices, n_to_sample)
            collected_samples.append(sampled)
            num_collected += sampled.size
            if num_collected >= size:
                break
        result = np.concatenate(collected_samples) if collected_samples else np.array([], dtype=int)
        if update_ranks and len(ordered_clusters) > 1 and not self.fixed_ranks:
            high = min(len(ordered_clusters), self.upper_random)
            self.update_ranks(ordered_clusters[np.random.randint(1, high)])
        return np.unique(result)


class DynamicWorkloadGenerator:
    """
    Generates a dynamic workload on a vector search dataset.

    Process:
      1. Load dataset (by name or from file paths).
      2. Cluster the base vectors using Faiss.
      3. Sample an initial resident set.
      4. Generate operations (insert, delete, query) and save them as .npy files.
      5. Compute ground truth for query operations.
      6. Save a runbook with parameters, operation details, and timings.
    """

    def __init__(
            self,
            dataset: str,
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
    ) -> None:
        self.dataset = dataset
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

        self.initial_clustering_path = self.workload_dir / "clustered_index.npy"

        # Load dataset by name or from provided file paths.
        self.base_vectors, self.queries = load_dataset_from_name(dataset)

        # Set random seeds for reproducibility.
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Create required directories.
        self.workload_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir = self.workload_dir / "operations"
        self.operations_dir.mkdir(exist_ok=True)

        self.n_vectors = self.base_vectors.shape[0]
        self.all_ids = np.arange(self.n_vectors)
        self.resident_set = np.zeros(self.n_vectors, dtype=bool)
        self.runbook: Dict[str, Any] = {"operations": {}}
        self.resident_history: List[np.ndarray] = []

    def validate_parameters(self) -> None:
        """Validate configuration parameters and raise errors if misconfigured."""
        if self.metric not in ["l2", "ip"]:
            raise ValueError(f"Invalid metric '{self.metric}'. Must be 'l2' or 'ip'.")
        total_ratio = self.insert_ratio + self.delete_ratio + self.query_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError("The sum of insert, delete, and query ratios must equal 1.")
        if self.update_batch_size <= 0 or self.query_batch_size <= 0 or self.num_operations <= 0:
            raise ValueError("Batch sizes and number of operations must be positive.")
        if self.initial_size <= 0 or self.cluster_size <= 0:
            raise ValueError("Initial size and cluster size must be positive.")
        if self.update_sample_distribution not in VALID_SAMPLE_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid update_sample_distribution '{self.update_sample_distribution}' in config. "
                f"Valid options are: {', '.join(VALID_SAMPLE_DISTRIBUTIONS)}."
            )
        if self.query_sample_distribution not in VALID_SAMPLE_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid query_sample_distribution '{self.query_sample_distribution}' in config. "
                f"Valid options are: {', '.join(VALID_SAMPLE_DISTRIBUTIONS)}."
            )

    def initialize_clustered_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster the base vectors using Faiss. If a precomputed clustering exists,
        load it; otherwise, perform clustering and save the results.

        Returns:
            assignments: 1D array of cluster IDs.
            centroids: 2D array of centroids.
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
            try:
                np.save(self.initial_clustering_path, {"assignments": assignments, "centroids": centroids})
                logger.info("Clustering completed and saved to '%s'", self.initial_clustering_path)
            except Exception as e:
                logger.error("Failed to save clustering: %s", e)
        elapsed = time.time() - start_time
        self.runbook["clustering_time"] = elapsed
        logger.info("Clustering elapsed time: %.2f s", elapsed)
        return assignments, centroids

    def initialize_workload(self) -> None:
        """Initialize the workload by setting up clustering and the initial resident set."""
        self.validate_parameters()
        self.assignments, self.centroids = self.initialize_clustered_index()
        # Setup samplers based on configuration.
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

        # Save initial indices and vector datasets.
        np.save(self.workload_dir / "initial_indices.npy", init_indices)
        np.save(self.workload_dir / "base_vectors.npy", self.base_vectors)
        if self.queries is not None:
            np.save(self.workload_dir / "query_vectors.npy", self.queries)

        self.runbook["parameters"] = {
            "dataset": self.dataset,
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
        logger.info("Workload initialization complete. Initial resident set size: %d", int(np.sum(self.resident_set)))

    def sample_indices(self, size: int, op_type: str) -> np.ndarray:
        """
        Sample indices for a given operation type.

        Parameters:
            size: Number of indices to sample.
            op_type: Operation type ("insert", "delete", or "search").

        Returns:
            Array of sampled indices.
        """
        if op_type == "insert":
            pool = self.all_ids[~self.resident_set]
        elif op_type == "delete":
            pool = self.all_ids[self.resident_set]
        elif op_type == "search":
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
        Compute ground truth for the given query indices using the current resident vectors.

        Parameters:
            q_indices: Indices corresponding to the queries.

        Returns:
            Dictionary with ground truth computation time and nearest neighbor IDs.
        """
        current_ids = np.where(self.resident_set)[0]
        current_vectors = self.base_vectors[current_ids]
        query_vectors = self.queries[q_indices] if self.queries is not None else self.base_vectors[q_indices]
        d = self.base_vectors.shape[1]
        gt_index = faiss.IndexFlatL2(d) if self.metric == "l2" else faiss.IndexFlatIP(d)
        gt_index.add(current_vectors)
        start_time = time.time()
        _, gt_ids = gt_index.search(query_vectors.astype(np.float32), GT_K)
        elapsed = time.time() - start_time
        return {"gt_time": elapsed, "gt_ids": gt_ids}

    # --- Helper Functions for Operation Processing ---

    def process_insert(self, op_index: int) -> Optional[Dict[str, Any]]:
        """Process an insert operation and save the indices."""
        indices = self.sample_indices(self.update_batch_size, "insert")
        if indices.size == 0:
            logger.info("Op %d [INSERT]: No indices available. Skipping generation.", op_index)
            return None
        self.resident_set[indices] = True
        try:
            np.save(self.operations_dir / f"{op_index}.npy", indices)
        except Exception as e:
            logger.error("Failed to save insert op %d: %s", op_index, e)
            return None
        return {"operation": "insert",
                "sample_size": int(indices.size),
                "n_resident": int(np.sum(self.resident_set))}

    def process_delete(self, op_index: int) -> Optional[Dict[str, Any]]:
        """Process a delete operation and save the indices."""
        indices = self.sample_indices(self.update_batch_size, "delete")
        if indices.size == 0:
            logger.info("Op %d [DELETE]: No indices available. Terminating generation.", op_index)
            return None
        self.resident_set[indices] = False
        try:
            np.save(self.operations_dir / f"{op_index}.npy", indices)
        except Exception as e:
            logger.error("Failed to save delete op %d: %s", op_index, e)
            return None
        return {"operation": "delete",
                "sample_size": int(indices.size),
                "n_resident": int(np.sum(self.resident_set))}

    def process_query(self, op_index: int) -> Optional[Dict[str, Any]]:
        """Process a query operation, compute its ground truth, and save the query indices."""
        q_indices = self.sample_indices(self.query_batch_size, "search")
        if q_indices.size == 0:
            logger.info("Op %d [QUERY]: No query indices available. Terminating generation.", op_index)
            return None
        try:
            np.save(self.operations_dir / f"{op_index}.npy", q_indices)
        except Exception as e:
            logger.error("Failed to save query op %d: %s", op_index, e)
            return None
        entry = {"operation": "search",
                 "sample_size": int(q_indices.size),
                 "n_resident": int(np.sum(self.resident_set))}
        gt_info = self.compute_ground_truth_for_query(q_indices)
        entry["gt_time"] = gt_info["gt_time"]
        return entry

    def update_resident_history(self) -> None:
        """Update resident history for visualization purposes."""
        res_ids = self.all_ids[self.resident_set]
        if res_ids.size > 0:
            n_clusters = int(self.assignments.max() + 1)
            counts_arr = np.bincount(self.assignments[res_ids], minlength=n_clusters)
            # Prevent division by zero with a small constant.
            fractions = counts_arr / (np.bincount(self.assignments[res_ids], minlength=n_clusters) + 1e-8)
            self.resident_history.append(fractions)

    def generate_workload(self) -> None:
        """Generate the workload based on the configured parameters."""
        overall_start = time.time()
        self.initialize_workload()
        op_times = {"insert": 0.0, "delete": 0.0, "search": 0.0}
        counts = {"insert": 0, "delete": 0, "search": 0}

        for i in range(self.num_operations):
            op_start = time.time()
            op_type = np.random.choice(
                ["insert", "delete", "search"],
                p=[self.insert_ratio, self.delete_ratio, self.query_ratio]
            )

            if op_type == "insert":
                entry = self.process_insert(i)
                if entry is None:
                    break
                counts["insert"] += 1
            elif op_type == "delete":
                entry = self.process_delete(i)
                if entry is None:
                    break
                counts["delete"] += 1
            elif op_type == "search":
                entry = self.process_query(i)
                if entry is None:
                    break
                counts["search"] += 1
            else:
                raise ValueError(f"Unknown op type: {op_type}")

            self.runbook["operations"][i] = entry
            op_elapsed = time.time() - op_start
            op_times[op_type] += op_elapsed

            self.update_resident_history()

            log_msg = f"Op {i} [{op_type.upper()}]: resident_size={int(np.sum(self.resident_set))}, " \
                      f"sample_size={entry.get('sample_size', 0)}, op_time={op_elapsed:.3f} s"
            if op_type == "search" and "gt_time" in entry:
                log_msg += f", gt_time={entry['gt_time']:.3f} s"
            logger.info(log_msg)

        total_ops = i + 1
        summary = {
            "n_inserts": counts["insert"],
            "n_deletes": counts["delete"],
            "n_queries": counts["search"],
            "n_operations": total_ops,
            "average_op_time": {op: (op_times[op] / counts[op] if counts[op] > 0 else 0) for op in op_times},
            "total_generation_time": time.time() - overall_start
        }
        self.runbook["summary"] = summary
        logger.info("Operation timings: %s", summary["average_op_time"])
        logger.info("Operations: %d (insert: %d, delete: %d, query: %d)",
                    total_ops, counts["insert"], counts["delete"], counts["search"])
        logger.info("Total workload generation time: %.2f s", summary["total_generation_time"])

        # Save the runbook in JSON format.
        try:
            runbook_json_path = self.workload_dir / "runbook.json"
            with runbook_json_path.open("w") as f:
                json.dump(self.runbook, f, indent=4)
            logger.info("Runbook saved to '%s'", runbook_json_path)
        except Exception as e:
            logger.error("Failed to save runbook: %s", e)


def main() -> None:
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
        query_sample_distribution=config.get("query_sample_distribution", DEFAULT_QUERY_CLUSTER_SAMPLE_DISTRIBUTION),
        seed=config.get("seed", DEFAULT_SEED),
        initial_clustering_path=config.get("initial_clustering_path", None),
        base_vectors_file=config.get("base_vectors_file", None),
        queries_file=config.get("queries_file", None)
    )

    logger.info("Starting workload generation...")
    overall_start = time.time()
    generator.generate_workload()
    total_elapsed = time.time() - overall_start
    logger.info("Workload generated in %.2f s", total_elapsed)


if __name__ == "__main__":
    main()