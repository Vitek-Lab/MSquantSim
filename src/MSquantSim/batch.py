"""Run the Random Forest evaluations as many small cluster jobs (e.g. a Slurm job array).

The case-study notebooks build the RF training inputs (one subsampled dataset per
iteration and per number of biological replicates) and a held-out validation set,
then loop over them with train_rf. On a cluster the same loop can be split up:

1. export_rf_inputs: save the inputs and validation set built by the notebook cells;
2. run_rf_task: evaluate one slice of iterations for one (source, n) per job;
3. gather_rf_results: assemble the slices into the notebooks' CSV format
   (rows = iterations, columns = biological replicates per group).

Each evaluation is the same train_rf call on the same input as in the notebook loop.

Command line:
    python -m MSquantSim.batch n_tasks INPUTS.pkl BLOCK
    python -m MSquantSim.batch run INPUTS.pkl TASK BLOCK OUT_DIR
    python -m MSquantSim.batch gather INPUTS.pkl BLOCK OUT_DIR source=path.csv [source=path.csv ...]
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

from MSquantSim.downstreamanalysis import train_rf


def export_rf_inputs(path, inputs, validation, label_col):
    """
    Save RF inputs for batch evaluation.

    inputs     : {source: {n: [DataFrame, ...]}}, each frame with proteins and label_col
    validation : DataFrame with the same protein columns and label_col
    """
    with open(path, "wb") as f:
        pickle.dump({"inputs": inputs, "validation": validation, "label_col": label_col}, f)


def rf_tasks(bundle, block):
    """Task list: (source, n, first iteration, last iteration + 1), `block` iterations each."""
    tasks = []
    for source, by_n in bundle["inputs"].items():
        for n, frames in by_n.items():
            for start in range(0, len(frames), block):
                tasks.append((source, n, start, min(start + block, len(frames))))
    return tasks


COLUMNS = ["source", "n", "iteration", "accuracy"]


def run_rf_task(path, task, block, out_dir):
    """
    Evaluate one task and write its accuracies to OUT_DIR/task_<task>.csv. The file is
    rewritten after every iteration and iterations already in it are skipped, so a task
    stopped by the time limit continues where it left off when run again.
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    source, n, start, stop = rf_tasks(bundle, block)[task]
    label = bundle["label_col"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"task_{task:05d}.csv")
    rows = list(pd.read_csv(out).itertuples(index=False, name=None)) if os.path.exists(out) else []
    done = {row[2] for row in rows}
    for s in range(start, stop):
        if s in done:
            continue
        data = bundle["inputs"][source][n][s]
        X, y = data.drop(columns=label), data[label]
        try:
            accuracy = train_rf(X, y, bundle["validation"])
        except ValueError:
            # unusable training data, e.g. a non-finite simulated value
            accuracy = np.nan
        rows.append((source, n, s, accuracy))
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)


def gather_rf_results(path, block, out_dir, file_names):
    """
    Assemble all task outputs into one CSV per source (iterations x n), as the
    notebooks write them. file_names: {source: output csv path}. Raises if any task
    output is missing.
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    tasks = rf_tasks(bundle, block)
    parts, incomplete = [], []
    for i, (_, _, start, stop) in enumerate(tasks):
        out = os.path.join(out_dir, f"task_{i:05d}.csv")
        part = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=COLUMNS)
        if len(part) < stop - start:
            incomplete.append(i)
        parts.append(part)
    if incomplete:
        raise RuntimeError(f"{path}: incomplete tasks {incomplete}")
    parts = pd.concat(parts)
    for source, csv in file_names.items():
        table = parts[parts.source == source].pivot(index="iteration", columns="n", values="accuracy")
        table = table[sorted(table.columns)]
        table.index.name = None
        table.columns.name = None
        table.to_csv(csv)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "n_tasks":
        with open(args[0], "rb") as f:
            print(len(rf_tasks(pickle.load(f), int(args[1]))))
    elif cmd == "run":
        run_rf_task(args[0], int(args[1]), int(args[2]), args[3])
    elif cmd == "gather":
        gather_rf_results(args[0], int(args[1]), args[2], dict(a.split("=", 1) for a in args[3:]))
    else:
        raise SystemExit(__doc__)
