"""Generate simulations for one dataset/method pair, following the case-study workflow.

The preprocessing here mirrors the notebooks in ../notebooks, with the ordering fix
applied to PDAC DIA and PDAC MRM: the train/validation split now happens *before*
KNN imputation, so the imputer is fit on training rows only (as MEL already did).

CRC is intentionally left on its original workflow: it has no missing values, ships as
two pre-split files, and has no LOD, so neither imputation nor masking applies.

Gene-name conversion (MEL, PDAC DIA) is read from the JSON maps produced by
make_column_maps.py so this script does not need protmapper/INDRA at run time.

Usage:
    python run_simulations.py --dataset MEL --method copula --n-sims 100 --n-jobs 32
"""
import argparse
import json
import os
import pickle
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from MSquantSim.simulate import (simulate_using_copula, simulate_using_per_protein,
                                 simulate_using_tvae)
from MSquantSim.utility import mask_simulations_with_lod

# Where each dataset's results go, and how its pickle files are named. Both mirror the
# paths the notebooks read back in, so downstream cells keep working unchanged.
DATASETS = {
    "MEL":      dict(out_dir="MEL_data/results",      stem="MEL",  masked_stem="MEL",
                     group_col="Condition"),
    "CRC":      dict(out_dir="CRC_data/results",       stem="CRC",  masked_stem=None,
                     group_col="Condition"),
    "PDAC_DIA": dict(out_dir="PDAC_data/DIA/results",  stem="PDAC", masked_stem="PDAC",
                     group_col="GROUP"),
    "PDAC_MRM": dict(out_dir="PDAC_data/MRM/results",  stem="PDAC", masked_stem="PDAC_MRM",
                     group_col="GROUP"),
}

METHODS = {
    "copula":   ("COPULA",   simulate_using_copula),
    "tvae":     ("TVAE",     simulate_using_tvae),
    "pprotein": ("PPROTEIN", simulate_using_per_protein),
}

# Seeds as used by each notebook, kept per dataset/method so results stay comparable
# with what the notebooks produce.
RANDOM_STATES = {
    ("MEL", "copula"): 2, ("MEL", "tvae"): 2, ("MEL", "pprotein"): 2,
    ("CRC", "copula"): 2, ("CRC", "tvae"): 2, ("CRC", "pprotein"): 2,
    ("PDAC_DIA", "copula"): 100, ("PDAC_DIA", "tvae"): 2, ("PDAC_DIA", "pprotein"): 2,
    ("PDAC_MRM", "copula"): 100, ("PDAC_MRM", "tvae"): 2, ("PDAC_MRM", "pprotein"): 2,
}


def drop_high_missing(df, threshold=0.5):
    """Drop columns missing in more than `threshold` of runs."""
    missing_percentages = df.isna().sum() / len(df)
    cols_to_keep = missing_percentages[missing_percentages <= threshold].index
    print(f"  columns after missingness filter: {len(cols_to_keep)} "
          f"(removed {len(missing_percentages) - len(cols_to_keep)})")
    return df[cols_to_keep]


def compute_lod(df, quantile=0.02):
    """Limit of detection per protein, used to mask simulated values back to NaN."""
    return df.select_dtypes(include=[np.number]).quantile(quantile, axis=0)


def split_train_validation(df, condition_col, random_state=20, test_size=0.2):
    """Stratified train/validation split. Runs *before* imputation."""
    X = df.drop(columns=[condition_col])
    y = df[condition_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train = pd.concat([y_train, X_train], axis=1)
    validation = pd.concat([y_test, X_test], axis=1)
    print(f"  train: {train.shape}, validation: {validation.shape}")
    return train, validation


def impute_knn(df, n_neighbors=5, weights="uniform"):
    """KNN-impute numeric columns, fit on `df` alone so validation rows never leak in."""
    orig_cols = df.columns.tolist()
    numeric_cols = [c for c in orig_cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in orig_cols if c not in numeric_cols]

    n_missing = int(df[numeric_cols].isna().sum().sum())
    print(f"  missing values before imputation: {n_missing}")
    if n_missing == 0:
        return df

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]),
                           columns=numeric_cols, index=df.index)
    out = pd.concat([df[non_numeric_cols], imputed], axis=1)[orig_cols]
    print(f"  missing values after imputation: {int(out[numeric_cols].isna().sum().sum())}")
    return out


def prepare_mel(data_root):
    """MEL: gene names -> missingness filter -> INDRA filter -> LOD -> split -> impute."""
    df = pd.read_csv(os.path.join(data_root, "mel/MEL_protein_level_data.csv"))
    df = df.loc[:, ~df.columns.str.contains(';', na=False)]
    df.columns = [c.split('|')[1] if '|' in c else c for c in df.columns]
    df = df.drop(columns=["Run", "Channel"])

    with open(os.path.join(data_root, "mel/MEL_column_map.json")) as f:
        colmap = json.load(f)
    df.columns = [colmap.get(c) for c in df.columns]

    df = drop_high_missing(df)

    # Keep only proteins in the largest connected component of the INDRA network
    with open(os.path.join(data_root, "mel/INDRA_network_MEL.pkl"), "rb") as f:
        network = pickle.load(f)
    filtered = network.loc[:, ["source", "target"]].drop_duplicates()
    G = nx.DiGraph()
    for i in range(len(filtered)):
        G.add_edge(filtered.iloc[i]["source"], filtered.iloc[i]["target"])
    max_component = max(nx.weakly_connected_components(G), key=len)
    print(f"  largest INDRA connected component: {len(max_component)} nodes")
    df = df.loc[:, [c for c in df.columns if c == 'Condition' or c in max_component]]
    print(f"  columns after INDRA filter: {df.shape[1]}")

    lod = compute_lod(df)
    train, _ = split_train_validation(df, "Condition")
    train = impute_knn(train)

    healthy = train[train['Condition'] == "u"].drop(columns=['Condition'])
    disease = train[train['Condition'] == "m"].drop(columns=['Condition'])
    return healthy, disease, lod, df


def prepare_crc(data_root):
    """CRC: unchanged from its notebook. Pre-split files, no missingness, no LOD."""
    train = pd.read_csv(os.path.join(data_root, "crc/CRC_training_protein_level_data.csv"))
    healthy = train[train['Condition'] == 'Healthy'].drop(
        columns=['Condition', 'Run', 'BioReplicate'])
    disease = train[train['Condition'] == 'CRC'].drop(
        columns=['Condition', 'Run', 'BioReplicate'])
    print(f"  train: {train.shape} (no imputation or LOD for CRC)")
    return healthy, disease, None, train


def prepare_pdac_dia(data_root):
    """PDAC DIA: gene names -> missingness filter -> LOD -> split -> impute."""
    df = pd.read_csv(os.path.join(data_root, "pdac/PDAC_DIA_protein_level_data.csv"))
    df = df.loc[:, ~df.columns.str.contains(';', na=False)]

    with open(os.path.join(data_root, "pdac/PDAC_DIA_column_map.json")) as f:
        colmap = json.load(f)
    df = df.drop(columns=[c for c in df.columns if colmap.get(c) == "__DROP__"])
    df.columns = [colmap.get(c) for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c is None])

    df = drop_high_missing(df)
    lod = compute_lod(df)

    train, _ = split_train_validation(df, "GROUP")
    train = impute_knn(train)

    healthy = train[train['GROUP'] == 'Normal'].drop(columns=['GROUP', 'originalRUN'])
    disease = train[train['GROUP'] == 'Cancer'].drop(columns=['GROUP', 'originalRUN'])
    return healthy, disease, lod, df.drop(columns=['originalRUN'])


def prepare_pdac_mrm(data_root):
    """PDAC MRM: missingness filter -> LOD -> split -> impute."""
    # True sample classes, benign removed (data/PDAC_data/MRM/relabel_protein_level.py)
    df = pd.read_csv(os.path.join(data_root, "pdac/PDAC_MRM_protein_level_data_v2.csv"))
    df = drop_high_missing(df)
    lod = compute_lod(df)

    # originalRUN is dropped up front here, matching the MRM notebook cell
    train, _ = split_train_validation(df.drop(columns=['originalRUN']), "GROUP")
    train = impute_knn(train)

    healthy = train[train['GROUP'] == 'Normal'].drop(columns=['GROUP'])
    disease = train[train['GROUP'] == 'Cancer'].drop(columns=['GROUP'])
    return healthy, disease, lod, df.drop(columns=['originalRUN'])


PREPARE = {
    "MEL": prepare_mel,
    "CRC": prepare_crc,
    "PDAC_DIA": prepare_pdac_dia,
    "PDAC_MRM": prepare_pdac_mrm,
}


def simulate(method, arm, n_sims, random_state, n_jobs):
    """Run `n_sims` simulations for one condition arm."""
    _, fn = METHODS[method]
    kwargs = dict(num_simulations=n_sims, samples_per_simulation=len(arm),
                  output_dir=None, output_prefix='simulated', return_data=True,
                  random_state=random_state, n_jobs=n_jobs)
    if method == "tvae":
        return fn(arm, None, **kwargs)
    return fn(arm, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-2,
                        help="parallel workers across simulations (-2 leaves one core free)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    method_prefix, _ = METHODS[args.method]
    random_state = RANDOM_STATES[(args.dataset, args.method)]

    print(f"=== {args.dataset} / {args.method} "
          f"(n_sims={args.n_sims}, n_jobs={args.n_jobs}, random_state={random_state}) ===")
    print("Preparing data...")
    healthy, disease, lod, _ = PREPARE[args.dataset](args.data_root)
    print(f"  healthy arm: {healthy.shape}, disease arm: {disease.shape}")

    out_dir = os.path.join(args.out_root, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    for arm_name, arm in [("healthy", healthy), ("disease", disease)]:
        start = time.time()
        sims = simulate(args.method, arm, args.n_sims, random_state, args.n_jobs)
        print(f"  {arm_name}: {len(sims)} simulations in {time.time() - start:.1f}s")

        path = os.path.join(out_dir, f"{method_prefix}_{cfg['stem']}_{arm_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(sims, f)
        print(f"  wrote {path}")

        if lod is not None:
            masked = mask_simulations_with_lod(sims, lod)
            path = os.path.join(
                out_dir, f"{method_prefix}_{cfg['masked_stem']}_{arm_name}_masked.pkl")
            with open(path, "wb") as f:
                pickle.dump(masked, f)
            print(f"  wrote {path}")

    print("done")


if __name__ == "__main__":
    sys.exit(main())
