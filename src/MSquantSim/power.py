"""Empirical power and false discovery proportion of t-tests for differential abundance.

Experiments are drawn from a pool of "null" samples with no condition effect:
either real data with each protein centered within its condition (a semi-synthetic benchmark), or
rows simulated from a model fitted to such data. Each experiment splits 2N pool
rows into two groups, adds a known log2 fold change to a chosen set of proteins
in one group, and calls differential abundance with Welch t-tests and
Benjamini-Hochberg (BH) adjustment, as run_ttest does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from MSquantSim.downstreamanalysis import _bh_adjust


def center_by_group(df, group_col):
    """Proteins with each group's mean subtracted (removes the condition effect)."""
    prots = [c for c in df.columns if c != group_col]
    X = df[prots].astype(float)
    return X - X.groupby(df[group_col]).transform("mean")


def within_group_variance(df, group_col):
    """Per-protein pooled within-group variance (Series indexed by protein)."""
    prots = [c for c in df.columns if c != group_col]
    groups = [g[prots].astype(float) for _, g in df.groupby(group_col)]
    ss = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dof = sum(g.notna().sum() - 1 for g in groups)
    return ss / dof


def welch_pvalues(a, b, min_samples=2):
    """
    Welch t-test p-values for every column of a (group 1) vs b (group 2).

    Matches run_ttest (scipy ttest_ind, equal_var=False): missing values (NaN)
    are dropped per protein, and proteins with fewer than `min_samples`
    observations in either group get NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.sum(~np.isnan(a), axis=0)
    nb = np.sum(~np.isnan(b), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        va = np.nanvar(a, axis=0, ddof=1) / na
        vb = np.nanvar(b, axis=0, ddof=1) / nb
        t = (np.nanmean(b, axis=0) - np.nanmean(a, axis=0)) / np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
        p = 2 * stats.t.sf(np.abs(t), dof)
    p[(na < min_samples) | (nb < min_samples)] = np.nan
    return p


def seed_changed_sets(corr, n_sets, n_seeds, rng, top_m=None, min_corr=None,
                      max_partners=None):
    """
    Changed-protein sets that follow the correlation structure: random seed
    proteins plus their most positively correlated partners.

    corr         : protein x protein correlation matrix (array or DataFrame)
    n_sets       : number of sets to draw (one per experiment)
    n_seeds      : seed proteins per set, drawn without replacement
    top_m        : rule A, each seed adds its top_m most positively correlated
                   proteins not already in the set, so overlaps are filled with
                   the next-strongest partners and the set size stays
                   n_seeds * (1 + top_m)
    min_corr     : rule B, each seed adds every protein with correlation >=
                   min_corr, at most max_partners (the strongest); set sizes vary
    Exactly one of top_m and min_corr is given. Only positive correlations count;
    a seed without positive (finite) partners adds none, so its set is smaller.

    Returns (sets, seeds): a list of sorted index arrays and an
    (n_sets x n_seeds) array of the seeds.
    """
    if (top_m is None) == (min_corr is None):
        raise ValueError("give exactly one of top_m and min_corr")
    corr = np.array(corr, dtype=float)
    p = corr.shape[0]
    corr = np.where(np.isfinite(corr), corr, -np.inf)
    np.fill_diagonal(corr, -np.inf)
    ranked = np.argsort(-corr, axis=1)  # partners of each protein, strongest first

    sets, seeds = [], np.empty((n_sets, n_seeds), dtype=int)
    for i in range(n_sets):
        s = rng.choice(p, n_seeds, replace=False)
        seeds[i] = s
        changed = set(s.tolist())
        for seed in s:
            partners = [j for j in ranked[seed] if corr[seed, j] > 0]
            if top_m is not None:
                partners = [j for j in partners if j not in changed][:top_m]
            else:
                partners = [j for j in partners if corr[seed, j] >= min_corr][:max_partners]
            changed.update(partners)
        sets.append(np.array(sorted(changed), dtype=int))
    return sets, seeds


def clustered_de_sets(corr, k, rng, min_corr=0.5, max_partners=15):
    """
    One truth assignment with exactly k DE proteins placed in correlated modules.

    Repeatedly draws a seed uniformly from the unselected proteins and adds its
    unselected partners with correlation > min_corr (at most max_partners, the
    strongest), until k proteins are selected. The last cluster is trimmed to
    exactly k by dropping its members least correlated with their seed (never
    the seed). A seed without such partners is a singleton cluster. Each cluster
    gets one random sign.

    corr : protein x protein correlation matrix (array or DataFrame)
    Returns (indices, signs, cluster_sizes): k protein indices, their +1 / -1
    effect directions, and the size of each cluster in the order drawn.
    """
    corr = np.array(corr, dtype=float)
    p = corr.shape[0]
    if not 0 < k <= p:
        raise ValueError(f"k must be in 1..{p}")
    corr = np.where(np.isfinite(corr), corr, -np.inf)
    np.fill_diagonal(corr, -np.inf)
    selected = np.zeros(p, dtype=bool)
    indices, signs, sizes = [], [], []
    while len(indices) < k:
        seed = rng.choice(np.flatnonzero(~selected))
        partners = np.flatnonzero(~selected & (corr[seed] > min_corr))
        partners = partners[np.argsort(-corr[seed, partners], kind="stable")][:max_partners]
        members = np.concatenate([[seed], partners])[:k - len(indices)]  # trims the weakest partners
        selected[members] = True
        sign = rng.choice([-1, 1])
        indices.extend(members.tolist())
        signs.extend([sign] * len(members))
        sizes.append(len(members))
    return np.array(indices, dtype=int), np.array(signs, dtype=int), sizes


def random_de_set(p, k, rng):
    """k DE proteins drawn uniformly from p, each with its own random +1 / -1 direction."""
    return rng.choice(p, k, replace=False), rng.choice([-1, 1], size=k)


def run_experiments(pool, n_per_group, n_changed, log2fc, n_experiments, rng,
                    fdr=0.05, always_changed=None, changed_sets=None, changed_signs=None,
                    independent_columns=False):
    """
    Repeated two-group experiments with known differentially abundant proteins.

    pool           : array (rows x proteins) of null samples; each experiment draws
                     2 * n_per_group distinct rows
    n_changed      : number of truly changed proteins per experiment
    log2fc         : log2 fold change added to the changed proteins in group 2
    always_changed : optional column indices that are changed in every
                     experiment (the remaining slots are chosen at random)
    changed_sets   : optional list of index arrays, the changed proteins of each
                     experiment (e.g. from seed_changed_sets); overrides n_changed
                     and always_changed, and rng then only draws the rows. Adds
                     columns n_changed and n_unchanged to the output
    changed_signs  : optional list of +1 / -1 arrays matching changed_sets; each
                     changed protein gets sign * log2fc (default +log2fc)
    independent_columns : if True, each protein draws its own 2 * n_per_group
                     rows, which keeps each protein's values but removes the
                     correlation between proteins

    Returns
    -------
    experiments : DataFrame, one row per experiment, with tp, fp, discoveries,
                  power (tp / n_changed) and fdp (fp / max(discoveries, 1))
    detected    : DataFrame (experiments x always_changed) of 0/1 detection
                  indicators, or None
    """
    pool = np.asarray(pool, dtype=float)
    n_rows, p = pool.shape
    if 2 * n_per_group > n_rows:
        raise ValueError(f"pool has {n_rows} rows; 2 x {n_per_group} needed")
    if changed_sets is not None:
        if len(changed_sets) != n_experiments:
            raise ValueError(f"{len(changed_sets)} changed sets for {n_experiments} experiments")
        always_changed = None
    if changed_signs is not None and (changed_sets is None or len(changed_signs) != len(changed_sets)):
        raise ValueError("changed_signs needs changed_sets of the same length")
    fixed = np.asarray([] if always_changed is None else always_changed, dtype=int)
    if changed_sets is None and len(fixed) > n_changed:
        raise ValueError("more always_changed proteins than n_changed")
    others = np.setdiff1d(np.arange(p), fixed)

    rows, detected = [], []
    for e in range(n_experiments):
        if independent_columns:
            drawn = np.argsort(rng.random((n_rows, p)), axis=0)[:2 * n_per_group]
            X = pool[drawn, np.arange(p)]
        else:
            X = pool[rng.choice(n_rows, 2 * n_per_group, replace=False)]
        a, b = X[:n_per_group], X[n_per_group:].copy()
        if changed_sets is not None:
            changed = np.asarray(changed_sets[e], dtype=int)
        else:
            changed = np.concatenate([fixed, rng.choice(others, n_changed - len(fixed),
                                                        replace=False)])
        if changed_signs is not None:
            b[:, changed] += log2fc * np.asarray(changed_signs[e], dtype=float)
        else:
            b[:, changed] += log2fc
        called = _bh_adjust(welch_pvalues(a, b)) < fdr
        is_changed = np.zeros(p, dtype=bool)
        is_changed[changed] = True
        tp = int(np.sum(called & is_changed))
        fp = int(np.sum(called & ~is_changed))
        rows.append((tp, fp, tp + fp))
        if len(fixed):
            detected.append(called[fixed].astype(int))

    experiments = pd.DataFrame(rows, columns=["tp", "fp", "discoveries"])
    if changed_sets is not None:
        experiments["n_changed"] = [len(s) for s in changed_sets]
        experiments["n_unchanged"] = p - experiments["n_changed"]
        experiments["power"] = experiments["tp"] / experiments["n_changed"]
    else:
        experiments["power"] = experiments["tp"] / n_changed
    experiments["fdp"] = experiments["fp"] / experiments["discoveries"].clip(lower=1)
    detected = pd.DataFrame(detected, columns=fixed) if len(fixed) else None
    return experiments, detected


def summarize_experiments(experiments, target_power=0.8, fdp_threshold=0.1,
                          fp_share_threshold=0.05):
    """
    Mean power, P(power >= target), and the spread of the FDP over experiments.

    When experiments has an n_unchanged column (run_experiments with
    changed_sets), also p_fp_share_gt_threshold: the share of experiments that
    call more than fp_share_threshold of the unchanged proteins significant.
    """
    power, fdp = experiments["power"], experiments["fdp"]
    out = pd.Series({
        "mean_power": power.mean(),
        "p_power_ge_target": (power >= target_power).mean(),
        "fdp_mean": fdp.mean(),
        "fdp_q90": fdp.quantile(0.90),
        "fdp_q95": fdp.quantile(0.95),
        "p_fdp_gt_threshold": (fdp > fdp_threshold).mean(),
    })
    if "n_unchanged" in experiments:
        share = experiments["fp"] / experiments["n_unchanged"]
        out["p_fp_share_gt_threshold"] = (share > fp_share_threshold).mean()
    return out


def summarize_success(experiments, fdr=0.05, target_power=0.8):
    """
    FDP and TPR (realized power) distributions over experiments, and
    P(success) = P(TPR >= target_power and FDP <= fdr).
    """
    tpr, fdp = experiments["power"], experiments["fdp"]
    return pd.Series({
        "mean_fdp": fdp.mean(),
        "fdp_q90": fdp.quantile(0.90),
        "p_fdp_gt_2alpha": (fdp > 2 * fdr).mean(),
        "mean_tpr": tpr.mean(),
        "sd_tpr": tpr.std(ddof=1),
        "p_success": ((tpr >= target_power) & (fdp <= fdr)).mean(),
    })


def analytic_sample_size(variance, log2fc, pi1, fdr=0.05, power=0.8):
    """
    Independence-based sample size per group for a two-group t-test design.

    Follows the MSstats designSampleSize calculation, with the proportion of
    changed proteins pi1 as an input instead of a fixed m0/m1 = 99: the
    BH-adjusted average significance level is
        alpha = power * fdr / (1 + (1 - fdr) * m0/m1),  m0/m1 = (1 - pi1) / pi1,
    and n = 2 * variance * (z_{1 - alpha/2} + z_{power})^2 / log2fc^2.

    variance : protein variance; the standard ("representative protein") choice
               is the median within-group variance across proteins
    Returns (n per group rounded up, alpha).
    """
    m0_m1 = (1 - pi1) / pi1
    alpha = power * fdr / (1 + (1 - fdr) * m0_m1)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n = 2 * variance * z ** 2 / log2fc ** 2
    return int(np.ceil(n)), alpha
