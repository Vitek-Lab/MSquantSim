"""Shared figure styling for MSquantSim plots.

The style sheet (``styles/paper.mplstyle``) implements the Journal of Proteome
Research graphics specifications; draw figures at their final size (7.0 in wide
for double column, <= 3.33 in for single column). Each simulator has one fixed
color, marker, and line style so it reads the same in every figure.
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Okabe-Ito hues; colorblind-safe with >= 3:1 contrast on white
SIMULATOR_STYLES = {
    "Experimental": dict(color="#000000", marker="o", linestyle="-"),
    "Per-protein": dict(color="#009E73", marker="^", linestyle=":"),
    "Copula": dict(color="#0072B2", marker="s", linestyle="-"),
    "TVAE": dict(color="#D55E00", marker="D", linestyle="--"),
}
SIMULATOR_ORDER = list(SIMULATOR_STYLES)

DOUBLE_COLUMN = 7.0   # in
SINGLE_COLUMN = 3.33  # in
BAND_ALPHA = 0.18


def use_paper_style():
    """Apply the packaged JPR style sheet to matplotlib's global rcParams."""
    path = resources.files("MSquantSim") / "styles" / "paper.mplstyle"
    plt.style.use(str(path))


def plot_band(ax, x, mean, lower, upper, name, **kwargs):
    """
    Line through ``mean`` with a shaded ``lower``-``upper`` band in the
    simulator's fixed style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, mean, lower, upper : array-like
    name : str
        Key of ``SIMULATOR_STYLES``; also used as the legend label.
    **kwargs
        Passed on to ``ax.plot`` (e.g. ``zorder``).

    Returns
    -------
    matplotlib.lines.Line2D
    """
    style = SIMULATOR_STYLES[name]
    ax.fill_between(x, lower, upper, color=style["color"], alpha=BAND_ALPHA,
                    linewidth=0)
    (line,) = ax.plot(x, mean, label=name, **style, **kwargs)
    return line


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Bold lowercase panel letter at the top-left corner of ``ax``."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="right", va="bottom")


def cluster_order(df: pd.DataFrame) -> list:
    """
    Column order from complete-linkage clustering of the Pearson correlation
    matrix (same procedure as the case-study notebooks).
    """
    X = df.corr().values
    X[np.isnan(X)] = 0
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method="complete")
    ind = sch.fcluster(L, 0.5 * d.max(), "distance")
    return [df.columns[i] for i in np.argsort(ind)]


def ordered_correlations(sources: dict):
    """
    Pearson correlation matrices with a shared protein order.

    Parameters
    ----------
    sources : dict of {str: pd.DataFrame}
        Must contain "Experimental"; its clustering sets the order used for
        every entry. Non-numeric columns are ignored.

    Returns
    -------
    order : list of str
    corrs : dict of {str: np.ndarray}
    """
    order = cluster_order(sources["Experimental"].select_dtypes(include=[np.number]))
    corrs = {name: df[order].corr().to_numpy() for name, df in sources.items()}
    return order, corrs


def corr_heatmap(ax, corr, **kwargs):
    """
    Draw a correlation matrix as a frameless, rasterized heatmap on ``ax``
    (blue = -1, white = 0, red = +1).

    Returns
    -------
    matplotlib.image.AxesImage
    """
    im = ax.imshow(np.asarray(corr), cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", rasterized=True, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im
