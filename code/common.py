"""Paths, the nine judgments and helpers shared by the numbered scripts."""
import pathlib
import numpy as np, pandas as pd
from scipy import stats

CODE = pathlib.Path(__file__).resolve().parent
DATA, RESULTS, WORK = CODE / "data", CODE / "results", CODE / "work"
FIGURES = CODE.parent / "figures"
JUDGMENTS = ["interesting", "impressive", "obvious", "easy", "familiar", "stable", "certain", "profound", "consistent"]
SETTINGS = ["zeroshot", "calibrated", "finetuned"]


def ratings():
    """-> table (one row per person x answer: subject, trial, source, adult, the 9 judgments), sorted subject ids,
    and per answer: source (bool) and voice (1 = announcer, 0 = child)."""
    R = pd.read_csv(DATA / "ratings.csv"); subs = np.sort(R.subject.unique())
    tr = R.drop_duplicates("trial").set_index("trial").loc[np.arange(1, 81)]
    return R, subs, tr.source.values == 1, tr.adult.values


def folds():
    """-> {repeat: fold (0-4) of trials 1..80}: 10 repeats of 5 folds, 16 answers each, 4 per voice x source."""
    F = pd.read_csv(DATA / "folds.csv")
    return {r: F[F.repeat == r].set_index("trial").fold.loc[np.arange(1, 81)].values for r in range(10)}


def pair_counts(score, rating, same=None):
    """score, rating: (n, C). Correct and total ordered pairs per column, over pairs with different ratings
    (ties in the score count half)."""
    s = np.sign(score[:, None] - score[None, :]); d = rating[:, None] - rating[None, :]
    ok = d > 0
    if same is not None:
        ok = ok & same[:, :, None]
    return (((s > 0) + 0.5 * (s == 0)) * ok).sum((0, 1)), ok.sum((0, 1))


def real_preference(R, subs, src, voice):
    """judgment -> per-person share of within-voice (sourced, plain) answer pairs where the sourced answer was rated
    higher (ties half); and the crowd's share over everyone's pairs."""
    per, crowd = {}, {}
    for j in JUDGMENTS:
        M = R.pivot(index="subject", columns="trial", values=j).loc[subs].values.astype(float); w, t = [], []
        for r in M:
            ww = tt = 0
            for v in (0, 1):
                d = r[src & (voice == v)][:, None] - r[~src & (voice == v)][None, :]
                ww += (d > 0).sum() + 0.5 * (d == 0).sum(); tt += d.size
            w.append(ww); t.append(tt)
        per[j] = pd.Series(np.array(w) / np.array(t), index=subs); crowd[j] = sum(w) / sum(t)
    return per, crowd


def shuffle_test(x, y, rng, n=10000):
    """One-sided Spearman test: rho and the share of n shuffles of x with rho at least as large."""
    rho = stats.spearmanr(x, y).statistic
    null = np.array([stats.spearmanr(rng.permutation(x), y).statistic for _ in range(n)])
    return rho, (1 + np.sum(null >= rho)) / (n + 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o]); out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out
