import os
import json
import matplotlib.pyplot as plt

def plot_runbook_results(runbook_files):
    """
    Given a list of runbook result JSON filenames (each corresponding to a particular index/algorithm),
    this function plots a 2x2 grid:
      A) Operation latency (different linestyles for insert (solid), delete (dashed),
         and search (dotted); y-axis is log scale).
      B) Query recall.
      C) Number of resident vectors.
      D) Memory usage of the index.

    Each algorithm is plotted using a unique color.

    Parameters:
      runbook_files (list of str): List of paths to runbook JSON files.
    """
    # Get default color cycle from matplotlib.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Data will be stored per algorithm label.
    algo_data = {}

    for i, rf in enumerate(runbook_files):
        with open(rf, 'r') as f:
            runbook = json.load(f)

        # Determine an algorithm label.
        if "additional" in runbook and "name" in runbook["additional"]:
            label = runbook["additional"]["name"]
        elif "parameters" in runbook and "algo" in runbook["parameters"]:
            label = runbook["parameters"]["algo"]
        else:
            label = os.path.splitext(os.path.basename(rf))[0]

        # Get vector dimension (if available) for approximate memory calculation.
        vector_dim = runbook.get("parameters", {}).get("vector_dimension", None)

        # Prepare containers.
        latency_insert = []
        latency_delete = []
        latency_search = []
        recall_search = []
        resident = []
        mem_usage = []  # in MB

        # Expect operations stored in a dict with keys that are op indices as strings.
        ops = runbook.get("operations", {})
        sorted_keys = sorted(ops.keys(), key=lambda x: int(x))

        for key in sorted_keys:
            op_entry = ops[key]
            try:
                op_index = int(key)
            except Exception:
                continue
            op_type = op_entry.get("operation", "").lower()
            latency = op_entry.get("latency", None)
            n_res = op_entry.get("n_resident", None)
            if n_res is not None:
                resident.append((op_index, n_res))
                if "memory_usage" in op_entry:
                    mem_usage_val = op_entry["memory_usage"]
                elif vector_dim is not None:
                    # Approximate memory usage: resident count * vector_dim * 4 bytes per float32, converted to MB.
                    mem_usage_val = n_res * vector_dim * 4 / (1024 * 1024)
                else:
                    mem_usage_val = None
                if mem_usage_val is not None:
                    mem_usage.append((op_index, mem_usage_val))

            if op_type == "insert" and latency is not None:
                latency_insert.append((op_index, latency))
            elif op_type == "delete" and latency is not None:
                latency_delete.append((op_index, latency))
            elif op_type == "search":
                if latency is not None:
                    latency_search.append((op_index, latency))
                recall = op_entry.get("recall", None)
                if recall is not None:
                    recall_search.append((op_index, recall))

        algo_data[label] = {
            "latency_insert": latency_insert,
            "latency_delete": latency_delete,
            "latency_search": latency_search,
            "recall_search": recall_search,
            "resident": resident,
            "mem_usage": mem_usage,
            "summary": runbook.get("summary", {})
        }

    # Create a 2x2 grid for plots.
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Panel A: Operation Latency (log y-scale).
    ax = axs[0, 0]
    for i, (label, data) in enumerate(algo_data.items()):
        color = colors[i % len(colors)]
        if data["latency_insert"]:
            x_vals, y_vals = zip(*data["latency_insert"])
            ax.plot(x_vals, y_vals, linestyle='-', color=color, label=f"{label} insert")
        if data["latency_delete"]:
            x_vals, y_vals = zip(*data["latency_delete"])
            ax.plot(x_vals, y_vals, linestyle='--', color=color, label=f"{label} delete")
        if data["latency_search"]:
            x_vals, y_vals = zip(*data["latency_search"])
            ax.plot(x_vals, y_vals, linestyle=':', color=color, label=f"{label} search")
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Latency (sec)")
    ax.set_title("Operation Latency")
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Query Recall.
    ax = axs[0, 1]
    for i, (label, data) in enumerate(algo_data.items()):
        if data["recall_search"]:
            color = colors[i % len(colors)]
            x_vals, y_vals = zip(*data["recall_search"])
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Query Recall")
    ax.set_title("Query Recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    # Panel C: Resident Vectors.
    ax = axs[1, 0]
    for i, (label, data) in enumerate(algo_data.items()):
        if data["resident"]:
            color = colors[i % len(colors)]
            x_vals, y_vals = zip(*data["resident"])
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Resident Vectors")
    ax.set_title("Number of Resident Vectors")
    ax.legend(fontsize=8, loc="upper right")

    # Panel D: Memory Usage.
    ax = axs[1, 1]
    for i, (label, data) in enumerate(algo_data.items()):
        if data["mem_usage"]:
            color = colors[i % len(colors)]
            x_vals, y_vals = zip(*data["mem_usage"])
            ax.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Index Memory Usage")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Example usage:
    runbook_files = [
        "test_workload/results/runbook_results.json"
    ]
    plot_runbook_results(runbook_files)