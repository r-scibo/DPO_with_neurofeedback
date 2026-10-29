"""Riemannian features of every answer's EEG.

Input  work/eeg_epochs.npz: eeg (44 people, 80 answers, 63 channels, 800 samples), µV, 200 Hz, from 1 s before the end
       of each answer to 3 s after it (the end of the answer and the silent break before rating); built from the public
       raw recordings as described in README.md.
Per answer: covariance (Oracle Approximating Shrinkage), re-centred on the person's Riemannian mean covariance, mapped to
the tangent space at the identity; upper triangle, off-diagonal x sqrt(2): 2,016 features.
-> work/features_riem.npy (44, 80, 2016)
python 1_features.py
"""
import numpy as np
from common import WORK


def oas(x):
    """Oracle Approximating Shrinkage covariance of (channels, times)."""
    x = x - x.mean(1, keepdims=True); n, p = x.shape[1], x.shape[0]
    S = x @ x.T / n; mu = np.trace(S) / p; tr2 = np.trace(S @ S)
    rho = min(1.0, ((1 - 2 / p) * tr2 + np.trace(S) ** 2) / ((n + 1 - 2 / p) * (tr2 - np.trace(S) ** 2 / p)))
    return (1 - rho) * S + rho * mu * np.eye(p)


def _fn(C, fn):
    w, V = np.linalg.eigh(C); return (V * fn(w)) @ V.T


def riemann_mean(Cs, it=20):
    M = Cs.mean(0)
    for _ in range(it):
        Mi = _fn(M, lambda w: w ** -0.5); Ms = _fn(M, np.sqrt)
        L = np.mean([_fn(Mi @ C @ Mi, np.log) for C in Cs], 0)
        M = Ms @ _fn(L, np.exp) @ Ms
        if np.linalg.norm(L) < 1e-8: break
    return M


def tangent(X):
    Cs = np.stack([oas(x) for x in X])
    Mi = _fn(riemann_mean(Cs), lambda w: w ** -0.5)
    iu = np.triu_indices(Cs.shape[1]); wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    return np.stack([_fn(Mi @ C @ Mi, np.log)[iu] * wgt for C in Cs])


if __name__ == "__main__":
    X = np.load(WORK / "eeg_epochs.npz")["eeg"]
    F = []
    for s in range(X.shape[0]):
        F.append(tangent(np.asarray(X[s], np.float64)))
        print("person", s + 1, "/", X.shape[0], flush=True)
    np.save(WORK / "features_riem.npy", np.stack(F).astype(np.float32))
