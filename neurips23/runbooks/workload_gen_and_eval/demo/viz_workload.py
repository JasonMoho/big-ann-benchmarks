#!/usr/bin/env python3
"""
Animate the dynamic workload generator operations visually.

This script loads:
  - base_vectors.npy (and checks that vectors are 2D),
  - initial_indices.npy (initial resident set),
  - runbook.json (which records operation details),
  - query_vectors.npy (if available),
  - and the operation files (stored as .npy files in an "operations" folder).

It uses Matplotlib’s animation to update the scatter plot:
  - Non-resident vectors are shown in light gray.
  - Resident vectors are marked in blue.
  - When a query operation occurs, query points are added with full opacity and then gradually fade out.

Usage:
  python animate_workload.py --workload_dir path/to/workload --interval 500
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_data(workload_dir):
    # Load base vectors and check they are 2D.
    base_vectors_path = os.path.join(workload_dir, "base_vectors.npy")
    if not os.path.exists(base_vectors_path):
        raise FileNotFoundError(f"Base vectors file not found: {base_vectors_path}")
    base_vectors = np.load(base_vectors_path)
    if base_vectors.ndim != 2 or base_vectors.shape[1] != 2:
        raise ValueError(f"Error: Base vectors must be 2D. Got shape: {base_vectors.shape}")

    # Load initial resident set indices.
    initial_indices_path = os.path.join(workload_dir, "initial_indices.npy")
    if not os.path.exists(initial_indices_path):
        raise FileNotFoundError(f"Initial indices file not found: {initial_indices_path}")
    initial_indices = np.load(initial_indices_path)

    # Load runbook JSON.
    runbook_path = os.path.join(workload_dir, "runbook.json")
    if not os.path.exists(runbook_path):
        raise FileNotFoundError(f"Runbook file not found: {runbook_path}")
    with open(runbook_path, "r") as f:
        runbook = json.load(f)

    # Optionally load query vectors if available and check they are 2D.
    query_vectors_path = os.path.join(workload_dir, "query_vectors.npy")
    if os.path.exists(query_vectors_path):
        query_vectors = np.load(query_vectors_path)
        if query_vectors.ndim != 2 or query_vectors.shape[1] != 2:
            raise ValueError(f"Error: Query vectors must be 2D. Got shape: {query_vectors.shape}")
    else:
        query_vectors = None

    # List the operation files from the "operations" folder (sorted by numeric filename).
    operations_dir = os.path.join(workload_dir, "operations")
    if not os.path.isdir(operations_dir):
        raise FileNotFoundError(f"Operations directory not found: {operations_dir}")
    op_files = sorted(
        [f for f in os.listdir(operations_dir) if f.endswith(".npy")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    return base_vectors, initial_indices, runbook, query_vectors, operations_dir, op_files


def update(frame, base_vectors, resident_set, query_vectors, operations_dir, op_files,
           scatter_res, scatter_nonres, scatter_query, ax, runbook, active_queries, fade_rate, info_text, sim_title):
    # If there are no more operations, do nothing.
    if frame >= len(op_files):
        return

    # Load current operation file.
    op_path = os.path.join(operations_dir, op_files[frame])
    indices = np.load(op_path)

    # Retrieve the operation type from the runbook.
    op_info = runbook.get("operations", {}).get(str(frame), runbook.get("operations", {}).get(frame, {}))
    op_type = op_info.get("operation", "unknown").lower()

    # Instead of showing op id/type, we set the title to simulation parameters.
    ax.set_title(sim_title, fontweight='bold', fontsize=8)

    # Process the operation.
    if op_type == "insert":
        resident_set[indices] = True
    elif op_type == "delete":
        resident_set[indices] = False
    elif op_type == "query":
        # For a query op, resident set stays unchanged.
        # Use query_vectors if available; otherwise, fall back to base_vectors.
        if query_vectors is not None:
            query_points = query_vectors[indices]
        else:
            query_points = base_vectors[indices]
        # Add each query point to the active queries list with full opacity.
        for point in query_points:
            active_queries.append({'pos': point, 'alpha': 1.0})

    # Fade all active query points.
    for q in active_queries:
        q['alpha'] = max(0, q['alpha'] - fade_rate)
    # Remove queries that have fully faded out.
    active_queries[:] = [q for q in active_queries if q['alpha'] > 0]

    # Update the query scatter.
    if active_queries:
        positions = np.vstack([q['pos'] for q in active_queries])
        colors = np.array([[1.0, 0.0, 0.0, q['alpha']] for q in active_queries])
    else:
        positions = np.empty((0, 2))
        colors = np.empty((0, 4))
    scatter_query.set_offsets(positions)
    scatter_query.set_edgecolors(colors)

    # Update resident and non-resident scatter data.
    scatter_res.set_offsets(base_vectors[resident_set])
    scatter_nonres.set_offsets(base_vectors[~resident_set])

    # Update the simulation info text in the top left with current operation details.
    info_text.set_text(
        f"Op #: {frame}\n"
        f"Type: {op_type.upper()}\n"
        f"Resident Count: {np.sum(resident_set)}\n"
        f"Total Ops: {len(op_files)}"
    )
    info_text.set_fontweight('bold')
    info_text.set_fontsize(10)


def main():
    parser = argparse.ArgumentParser(
        description="Animate the dynamic workload generator visual debugging (only supports 2D vectors)."
    )
    parser.add_argument("--workload_dir", type=str, required=True,
                        help="Path to the workload directory (must contain base_vectors.npy, initial_indices.npy, runbook.json, operations folder, etc.)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for the animation.")
    args = parser.parse_args()

    # Load required data.
    base_vectors, initial_indices, runbook, query_vectors, operations_dir, op_files = load_data(args.workload_dir)
    n_vectors = base_vectors.shape[0]

    # Initialize the resident set based on initial_indices.
    resident_set = np.zeros(n_vectors, dtype=bool)
    resident_set[initial_indices] = True

    # Initialize an empty list for active query points that will fade over time.
    active_queries = []  # Each element is a dict: {'pos': np.array([x, y]), 'alpha': float}

    # Create the Matplotlib figure and initial scatter plots.
    fig, ax = plt.subplots()
    scatter_nonres = ax.scatter(base_vectors[~resident_set, 0], base_vectors[~resident_set, 1],
                                c="lightgray", s=30, label="Non-resident")
    scatter_res = ax.scatter(base_vectors[resident_set, 0], base_vectors[resident_set, 1],
                             c="blue", s=30, label="Resident Vector")
    scatter_query = ax.scatter([], [], marker="x", s=100, linewidths=2, label="Query", c="red")

    ax.set_xlabel("X", fontweight='bold', fontsize=12)
    ax.set_ylabel("Y", fontweight='bold', fontsize=12)
    ax.tick_params(labelsize=10)

    # Create a legend with bold fonts.
    legend = ax.legend(loc="upper right", prop={'weight': 'bold', 'size': 10})

    # Create a text annotation for operation details in the top left.
    info_text = ax.text(0.01, 0.99, "",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="left",
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(facecolor="white", alpha=0.7))

    # Compute a simulation parameters title from the runbook.
    params = runbook.get("parameters", {})
    sim_title = (
        f"Insert/Delete/Query Ratio={params.get('insert_ratio', 'N/A'), params.get('delete_ratio', 'N/A'), params.get('query_ratio', 'N/A')}, "
        f"Update/Query Distribution={params.get('update_sample_distribution', 'N/A')}"
    )

    # Set a fade_rate (amount to reduce alpha per frame, e.g., 0.1).
    fade_rate = 0.1

    # Create the animation.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(runbook["operations"]),
        fargs=(base_vectors, resident_set, query_vectors, operations_dir, op_files,
               scatter_res, scatter_nonres, scatter_query, ax, runbook, active_queries, fade_rate, info_text,
               sim_title),
        repeat=False
    )

    # save animation
    ani.save(os.path.join(args.workload_dir, "workload_animation.mp4"), fps=args.fps, extra_args=['-vcodec', 'libx264'])
    print("Animation saved to {}".format(os.path.join(args.workload_dir, "workload_animation.mp4")))



if __name__ == "__main__":
    main()