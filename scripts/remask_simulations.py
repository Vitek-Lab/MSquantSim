"""Re-mask already-generated simulations with the fitted MNAR dropout model.

The simulations were originally masked by thresholding at a per-protein LOD (the 2%
quantile of the observed data). This replaces that with the detection curve learned by
fit_mnar: a shared logistic P(detect) = expit(a + b * x) fitted jointly with per-protein
marginals on the real, pre-imputation data.

Only the *_masked.pkl files are rewritten. The unmasked simulations are read as-is and
never regenerated. CRC is skipped: it has no missing values to learn a dropout curve
from, and it has no masked outputs.

Usage:
    python remask_simulations.py --results-root ../data --dataset MEL
    python remask_simulations.py --results-root ../data          # all datasets
"""
import argparse
import os
import pickle
import sys
import time
import zlib

import numpy as np

from MSquantSim.utility import apply_dropout, fit_mnar

from run_simulations import DATASETS, METHODS, PREPARE

# CRC has no missing values, so the detection curve is unidentifiable there and it has
# no masked files to rewrite.
SKIP = {"CRC"}


def remask_dataset(dataset, results_root, data_root, seed, n_outer):
    cfg = DATASETS[dataset]
    group_col = cfg["group_col"]

    print(f"=== {dataset} ===")
    print("Preparing pre-imputation data...")
    _, _, _, raw = PREPARE[dataset](data_root)
    print(f"  raw frame for MNAR fit: {raw.shape} (group column '{group_col}')")
    n_missing = int(raw.drop(columns=group_col).isna().sum().sum())
    print(f"  missing values available to fit: {n_missing}")
    if n_missing == 0:
        print("  no missingness -> cannot fit a dropout curve, skipping")
        return []

    start = time.time()
    fits, (a, b) = fit_mnar(raw, group_col=group_col, n_outer=n_outer)
    print(f"  fit_mnar: a={a:.4f} b={b:.4f} in {time.time() - start:.1f}s "
          f"({len(fits)} groups)")

    out_dir = os.path.join(results_root, cfg["out_dir"])
    written = []
    for method, (prefix, _) in sorted(METHODS.items()):
        for arm in ("healthy", "disease"):
            src = os.path.join(out_dir, f"{prefix}_{cfg['stem']}_{arm}.pkl")
            dst = os.path.join(out_dir,
                               f"{prefix}_{cfg['masked_stem']}_{arm}_masked.pkl")
            with open(src, "rb") as f:
                sims = pickle.load(f)

            # One generator for the whole file, seeded per dataset/method/arm, so a rerun
            # reproduces the same dropout pattern instead of drawing fresh randomness.
            # crc32 rather than hash(): str hashing is salted per process by default.
            tag = f"{seed}/{dataset}/{method}/{arm}".encode()
            rng = np.random.default_rng(zlib.crc32(tag))
            masked = [apply_dropout(df, a, b, group_col=group_col, rng=rng)
                      for df in sims]

            before = sum(int(df.isna().sum().sum()) for df in sims)
            after = sum(int(df.isna().sum().sum()) for df in masked)
            total = sum(df.size for df in masked)
            with open(dst, "wb") as f:
                pickle.dump(masked, f)
            print(f"  {os.path.basename(dst):44s} NaN {before} -> {after} "
                  f"({100 * after / total:.1f}% of cells)")
            written.append(dst)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True,
                        help="directory holding <DS>_data/results trees")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append",
                        help="repeatable; defaults to every dataset except CRC")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n-outer", type=int, default=8,
                        help="fit_mnar outer iterations")
    args = parser.parse_args()

    datasets = args.dataset or [d for d in sorted(DATASETS) if d not in SKIP]
    written = []
    for ds in datasets:
        if ds in SKIP:
            print(f"=== {ds} === skipped (no missingness, no masked outputs)")
            continue
        written += remask_dataset(ds, args.results_root, args.data_root,
                                  args.seed, args.n_outer)

    print(f"\nrewrote {len(written)} masked files")


if __name__ == "__main__":
    sys.exit(main())
