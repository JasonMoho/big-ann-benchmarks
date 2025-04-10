#!/usr/bin/env python3
import os
import json
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


def load_runbook_results(results_dir):
    """
    Scan the provided results_dir for subdirectories that contain
    a 'runbook_results.json' file. Return a dictionary mapping
    algorithm labels to the parsed runbook data.
    """
    algo_data = {}
    # Iterate through every item in the results directory.
    for item in os.listdir(results_dir):
        subdir = os.path.join(results_dir, item)
        if os.path.isdir(subdir):
            runbook_path = os.path.join(subdir, "runbook_results.json")
            if os.path.exists(runbook_path):
                with open(runbook_path, 'r') as f:
                    runbook = json.load(f)
                # Use the subdirectory name as label unless parameters define an index_config_name.
                params = runbook.get("parameters", {})
                label = params.get("index_config_name", item)
                algo_data[label] = runbook
    return algo_data


def extract_data(runbook):
    """
    From a runbook dictionary, extract ordered data for:
      - latencies: insert, delete, search
      - query recall,
      - resident vector counts,
      - approximate memory usage.
    Assumes operations are stored in a dict with keys that can be cast to int.
    """
    vector_dim = runbook.get("parameters", {}).get("vector_dimension", None)

    latency_insert = []
    latency_delete = []
    latency_search = []
    recall_search = []
    resident = []
    mem_usage = []  # MB

    operations = runbook.get("operations", {})
    sorted_keys = sorted(operations.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        op_entry = operations[key]
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
                mem_val = op_entry["memory_usage"]
            elif vector_dim is not None:
                mem_val = n_res * vector_dim * 4 / (1024 * 1024)
            else:
                mem_val = None
            if mem_val is not None:
                mem_usage.append((op_index, mem_val))
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
    return {
        "latency_insert": latency_insert,
        "latency_delete": latency_delete,
        "latency_search": latency_search,
        "recall_search": recall_search,
        "resident": resident,
        "mem_usage": mem_usage
    }


def plot_all_results(results_dir, output_file=None):
    """
    Scan a results directory for subdirectories containing runbook_results.json;
    then produce a 2x2 grid:
      A) Operation latency (insert: solid, delete: dashed, search: dotted; log-scale y-axis).
      B) Query recall (y-axis [0,1]). If no recall data is found, a message is displayed.
      C) Resident vector count.
      D) Index memory usage.

    A single common legend at the top of the figure maps each method to its color,
    while the latency subplot includes a separate legend (dummy lines) describing the linestyles.

    A basic CLI is provided.
    """
    # Load all runbook results.
    runbooks = load_runbook_results(results_dir)
    if not runbooks:
        print(f"No runbook_results.json files found under {results_dir}")
        return

    # For each method, extract plot data.
    method_plot_data = {}
    method_colors = {}  # Assign a color for each method
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, runbook) in enumerate(runbooks.items()):
        method_plot_data[label] = extract_data(runbook)
        method_colors[label] = default_colors[i % len(default_colors)]

    # Create a 2x2 grid.
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Panel A: Operation Latency.
    ax = axs[0, 0]
    # For each method, plot latency using three linestyles.
    for label, data in method_plot_data.items():
        color = method_colors[label]
        if data["latency_insert"]:
            x, y = zip(*data["latency_insert"])
            ax.plot(x, y, linestyle='-', color=color)
        if data["latency_delete"]:
            x, y = zip(*data["latency_delete"])
            ax.plot(x, y, linestyle='--', color=color)
        if data["latency_search"]:
            x, y = zip(*data["latency_search"])
            ax.plot(x, y, linestyle=':', color=color)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Latency (sec)")
    ax.set_title("Operation Latency")
    ax.set_yscale("log")
    # Create a dummy legend solely for linestyle meanings.
    dummy_insert = mlines.Line2D([], [], color='black', linestyle='-', label='Insert')
    dummy_delete = mlines.Line2D([], [], color='black', linestyle='--', label='Delete')
    dummy_search = mlines.Line2D([], [], color='black', linestyle=':', label='Search')
    ax.legend(handles=[dummy_insert, dummy_delete, dummy_search], fontsize=8, loc="upper right")

    # Panel B: Query Recall.
    ax = axs[0, 1]
    any_recall = False
    for label, data in method_plot_data.items():
        if data["recall_search"]:
            any_recall = True
            color = method_colors[label]
            x, y = zip(*data["recall_search"])
            ax.plot(x, y, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Query Recall")
    ax.set_title("Query Recall")
    ax.set_ylim(0, 1)
    if any_recall:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No recall data available", fontsize=14,
                transform=ax.transAxes, ha="center")

    # Panel C: Resident Vectors.
    ax = axs[1, 0]
    for label, data in method_plot_data.items():
        if data["resident"]:
            color = method_colors[label]
            x, y = zip(*data["resident"])
            ax.plot(x, y, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Resident Vectors")
    ax.set_title("Number of Resident Vectors")
    ax.legend(fontsize=8, loc="upper right")

    # Panel D: Memory Usage.
    ax = axs[1, 1]
    for label, data in method_plot_data.items():
        if data["mem_usage"]:
            color = method_colors[label]
            x, y = zip(*data["mem_usage"])
            ax.plot(x, y, marker='o', linestyle='-', color=color, label=label)
    ax.set_xlabel("Operation Index")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Index Memory Usage")
    ax.legend(fontsize=8, loc="upper right")

    # Create a common legend at the very top showing the mapping method -> color.
    legend_handles = []
    for label, color in method_colors.items():
        legend_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=label))
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(method_colors), fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot results for all methods in a provided results directory."
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing subdirectories with runbook_results.json files.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="If provided, save the output plot to this file (e.g., plot.png).")
    args = parser.parse_args()

    plot_all_results(args.results_dir, args.output_file)


if __name__ == "__main__":
    main()