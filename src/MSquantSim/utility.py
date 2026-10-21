import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from numpy.polynomial.hermite_e import hermegauss
 
# Gauss-Hermite nodes for integrating against a standard normal density
_NODES, _W = hermegauss(40)
_W = _W / np.sqrt(2 * np.pi)
 
 
def _nll_protein(params, x_obs, n_miss, a, b):
    """Negative log-likelihood for one protein given the detection curve."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return 1e12
 
    # observed values: density * P(detected)
    ll = np.sum(norm.logpdf(x_obs, mu, sigma) + log_expit(a + b * x_obs))
 
    # missing values: integrate density * P(not detected) over the real line
    if n_miss > 0:
        t = mu + sigma * _NODES
        p_miss = np.dot(_W, 1.0 - expit(a + b * t))
        ll += n_miss * np.log(max(p_miss, 1e-300))
 
    return -ll
 
 
def _fit_marginals(block, a, b, min_obs=3, max_missing=0.9):
    """Fit mu, sigma for every protein in one group, holding (a, b) fixed."""
    rows = {}
    n_runs = len(block)
    for prot in block.columns:
        col = block[prot]
        x = col.dropna().to_numpy(dtype=float)
        n_miss = int(col.isna().sum())
        if len(x) < min_obs or n_miss / n_runs > max_missing:
            continue
        start = [x.mean(), np.log(x.std(ddof=1) + 1e-6)]
        res = minimize(_nll_protein, start, args=(x, n_miss, a, b),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        mu, log_sigma = res.x
        rows[prot] = (mu, np.exp(log_sigma))
    return pd.DataFrame(rows, index=["mu", "sigma"]).T
 
 
def _fit_detection(blocks, fits, a0, b0):
    """Optimize the shared detection curve with all marginals held fixed."""
    # pre-extract so the objective is cheap
    cache = []
    for block, fit in zip(blocks, fits):
        for prot in fit.index:
            col = block[prot]
            cache.append((col.dropna().to_numpy(dtype=float),
                          int(col.isna().sum()),
                          fit.at[prot, "mu"],
                          np.log(fit.at[prot, "sigma"])))
 
    def objective(ab):
        a, b = ab
        if b <= 1e-6:
            return 1e12
        return sum(_nll_protein([mu, ls], x, nm, a, b)
                   for x, nm, mu, ls in cache)
 
    res = minimize(objective, [a0, b0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 500})
    return res.x
 
 
def fit_mnar(df, group_col="Condition", n_outer=8, tol=1e-4, verbose=True):
    """
    df       : runs x proteins DataFrame, plus one grouping column
    returns  : (dict of {group: DataFrame[mu, sigma]}, (a, b))
    """
    groups = list(df[group_col].unique())
    blocks = [df.loc[df[group_col] == g].drop(columns=group_col).astype(float)
              for g in groups]
 
    vals = df.drop(columns=group_col).to_numpy(dtype=float)
    a, b = -np.nanmedian(vals), 1.0  # 50% detection near the middle of the range
 
    for it in range(n_outer):
        fits = [_fit_marginals(bl, a, b) for bl in blocks]
        a_new, b_new = _fit_detection(blocks, fits, a, b)
        shift = max(abs(a_new - a), abs(b_new - b))
        a, b = a_new, b_new
        if verbose:
            print(f"iter {it}: a={a:.4f}  b={b:.4f}  delta={shift:.2e}")
        if shift < tol:
            break
 
    fits = [_fit_marginals(bl, a, b) for bl in blocks]
    return dict(zip(groups, fits)), (a, b)


def calibrate_offsets(df, a, b, target_rate, group_col="Condition", tol=1e-4):
    """
    Intercept shift that makes the expected missing-value rate of complete data
    equal `target_rate` under the detection curve (a, b).

    df          : complete runs x proteins DataFrame, plus grouping column
    a, b        : detection curve from fit_mnar
    target_rate : desired overall fraction of missing values, in [0, 1)
    returns     : {protein: shift}, the same shift for every protein, in the
                  format apply_dropout expects
    """
    prots = [c for c in df.columns if c != group_col]
    X = df[prots].to_numpy(dtype=float)
    if target_rate <= 0:
        return {p: np.inf for p in prots}

    def expected_rate(shift):
        return np.mean(1.0 - expit(a + shift + b * X))

    # expected_rate decreases monotonically in the shift: bisect on a wide bracket
    lo, hi = -100.0, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if expected_rate(mid) > target_rate:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    return {p: shift for p in prots}


def apply_dropout(sim_df, a, b, group_col="Condition", offsets=None, rng=None):
    """
    Impose MNAR missingness on complete simulated data.
 
    sim_df  : simulated runs x proteins, complete, plus grouping column
    a, b    : detection curve from fit_mnar
    offsets : optional per-protein intercept shifts from calibrate_offsets
    """
    rng = np.random.default_rng() if rng is None else rng
    out = sim_df.copy()
    prots = [c for c in out.columns if c != group_col]
 
    X = out[prots].to_numpy(dtype=float)
    shift = np.zeros(len(prots))
    if offsets is not None:
        shift = np.array([offsets.get(p, 0.0) for p in prots], dtype=float)
 
    p_miss = 1.0 - expit(a + shift[None, :] + b * X)
    X[rng.random(X.shape) < p_miss] = np.nan
    out[prots] = X
    return out

def mask_simulations_with_lod(sim_list, lod_series):
    masked_list = []
    for df in sim_list:
        df_masked = df.copy()
        # Only consider numeric protein columns that appear in lod
        prot_cols = df_masked.select_dtypes(include=[np.number]).columns.intersection(lod_series.index)
        if len(prot_cols) > 0:
            mins = lod_series.loc[prot_cols]
            # Broadcast series of mins across columns and mask values below LOD
            df_masked.loc[:, prot_cols] = df_masked[prot_cols].where(df_masked[prot_cols].ge(mins), np.nan)
        masked_list.append(df_masked)
    return masked_list
