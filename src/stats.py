"""Reusable statistics helpers for the Byte Beads analysis.

Small-sample, paired, mostly binary/ordinal data (n=30, one intact class,
no control group). Functions favour paired and non-parametric tests and
always return an effect size alongside the p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_change(pre: np.ndarray, post: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon signed-rank on complete cases.

    Returns mean difference, both p-values, Cohen's dz, and a 95% CI on the
    mean difference (t-based). Complete-case: rows with NaN in either arm drop.
    """
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    mask = ~np.isnan(pre) & ~np.isnan(post)
    a, b = pre[mask], post[mask]
    diff = b - a
    n = len(diff)
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd else np.nan
    t, p_t = stats.ttest_rel(b, a)
    # Wilcoxon is undefined if all diffs are zero
    try:
        _, p_w = stats.wilcoxon(b, a)
    except ValueError:
        p_w = np.nan
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (diff.mean() - tcrit * se, diff.mean() + tcrit * se)
    return {
        "n": n,
        "mean_diff": diff.mean(),
        "ci95": ci,
        "t": t,
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
        "cohens_dz": dz,
    }


def mcnemar(pre: np.ndarray, post: np.ndarray) -> dict:
    """McNemar test for a paired binary item (exact binomial on discordants)."""
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    mask = ~np.isnan(pre) & ~np.isnan(post)
    a, b = pre[mask].astype(int), post[mask].astype(int)
    b01 = int(np.sum((a == 0) & (b == 1)))  # gained
    b10 = int(np.sum((a == 1) & (b == 0)))  # lost
    n_disc = b01 + b10
    # exact two-sided binomial on the smaller discordant count
    p = stats.binomtest(min(b01, b10), n_disc, 0.5).pvalue if n_disc else 1.0
    return {"n": len(a), "gained_0to1": b01, "lost_1to0": b10, "p_exact": p}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation on complete cases (ordinal / small n)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"n": int(mask.sum()), "rho": rho, "p": p}


def kr20(item_matrix: pd.DataFrame) -> float:
    """Kuder-Richardson 20 internal-consistency reliability for binary items.

    item_matrix: rows = students (complete cases only), columns = items (0/1).
    """
    m = item_matrix.dropna().astype(float)
    k = m.shape[1]
    p = m.mean(axis=0)
    q = 1 - p
    total_var = m.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - (p * q).sum() / total_var)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(running, 1.0)
    return adj
