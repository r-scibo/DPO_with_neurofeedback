"""Decode the nine judgments from the Riemannian features, and test the decoder against shuffled ratings.

Ridge regression, solved in its dual (kernel) form on one linear kernel over all 44 x 80 answers; features z-scored per
person (from that person's answers, no labels); ratings z-scored per person on the training answers only. 10 repeats of
5-fold cross-validation over answers (data/folds.csv), so every score comes from a model that never saw that answer.
Three settings:
  zeroshot    trained on the other 43 people only (leave one person out)
  calibrated  trained on this person's training answers only (64 answers)
  finetuned   calibrated ridge fitted to the residual of the person's zero-shot model
Ridge strength by inner cross-validation over the 4 training folds (zero-shot: x 4 groups of training people).
Pairwise accuracy: within a person and test fold, how often the decoder orders two answers the way the person rated them.
Null (--perm 200): the whole pipeline re-run with each person's ratings shuffled, 200 times; p = share of shuffles at least
as accurate. A GPU is strongly advised for the null (torch, CUDA); the predictions alone take about 25 minutes on a laptop CPU.
-> work/predictions.npz (setting -> (10 repeats, 44, 80, 9)); with --perm: results/decoder_accuracy.csv
python 2_decode.py [--perm 200] [--repeats 10]
"""
import sys, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from common import WORK, RESULTS, DATA, JUDGMENTS, SETTINGS, ratings, folds, pair_counts

GRID = 10.0 ** np.arange(-2, 5)
try:                                      # GPU: same algebra in torch, float32 Cholesky on CUDA
    import torch
    DEV = "cuda" if torch.cuda.is_available() else None
except ImportError:
    torch, DEV = None, None


def _t(a):
    return torch.as_tensor(np.asarray(a), device=DEV)


def ridge_cv(K, va_sets, Y):
    """Inner CV: va_sets = list of (inner_train_idx, inner_val_idx) -> best ridge strength (index) per column."""
    if DEV:
        err = torch.zeros((len(GRID), Y.shape[1]), device=DEV, dtype=torch.float32)
        for itr, iva in va_sets:
            itr, iva = _t(itr), _t(iva); Ktt = K[itr][:, itr]; Kvt = K[iva][:, itr]; I = torch.eye(len(itr), device=DEV)
            for g, lam in enumerate(GRID):
                err[g] += ((Kvt @ torch.cholesky_solve(Y[itr], torch.linalg.cholesky(Ktt + float(lam) * I)) - Y[iva]) ** 2).sum(0)
        return err.argmin(0).cpu().numpy()
    err = np.zeros((len(GRID), Y.shape[1]))
    for itr, iva in va_sets:
        Ktt = K[np.ix_(itr, itr)]; Kvt = K[np.ix_(iva, itr)]
        for g, lam in enumerate(GRID):
            err[g] += ((Kvt @ cho_solve(cho_factor(Ktt + lam * np.eye(len(itr))), Y[itr]) - Y[iva]) ** 2).sum(0)
    return err.argmin(0)


def ridge_fit_predict(K, tr, te, Y, best):
    if DEV:
        tr, te = _t(tr), _t(te); Ktt = K[tr][:, tr]; Kxt = K[te][:, tr]; I = torch.eye(len(tr), device=DEV)
        out = torch.zeros((len(te), Y.shape[1]), device=DEV)
        for g in np.unique(best):
            cols = _t(np.flatnonzero(best == g))
            out[:, cols] = Kxt @ torch.cholesky_solve(Y[tr][:, cols], torch.linalg.cholesky(Ktt + float(GRID[g]) * I))
        return out.double().cpu().numpy()
    out = np.zeros((len(te), Y.shape[1])); Ktt = K[np.ix_(tr, tr)]; Kxt = K[np.ix_(te, tr)]
    for g in np.unique(best):
        cols = best == g
        out[:, cols] = Kxt @ cho_solve(cho_factor(Ktt + GRID[g] * np.eye(len(tr))), Y[np.ix_(tr, cols)])
    return out


def evaluate(F, n_perm=200, repeats=range(10), seed=0):
    """All judgments solved together: columns = judgment x (real ratings + n_perm shuffles). The shuffles are always
    drawn, so the random draws (and the predictions) do not depend on n_perm. -> predictions, pairwise accuracy."""
    R, subs, src, voice = ratings(); fold_of = folds()
    groups = pd.read_csv(DATA / "people_groups.csv").set_index("subject").group.loc[subs].values
    S, NT = len(subs), len(JUDGMENTS)
    F = F.astype(np.float64); F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    Ff = F.reshape(S * 80, -1)
    if DEV:
        Fg = torch.as_tensor(Ff, device=DEV, dtype=torch.float32); K = Fg @ Fg.T / Ff.shape[1]; del Fg
    else:
        K = Ff @ Ff.T / Ff.shape[1]
    rng = np.random.default_rng(seed)
    perm = np.stack([np.arange(80)[None].repeat(S, 0)] + [np.stack([rng.permutation(80) for _ in range(S)]) for _ in range(200)], -1)
    perm = perm[..., :1 + n_perm]; C = 1 + n_perm
    Rm = np.concatenate([np.take_along_axis(R.pivot(index="subject", columns="trial", values=j).loc[subs].values.astype(float)[:, :, None]
                                            .repeat(C, 2), perm, 1) for j in JUDGMENTS], 2)          # (S, 80, NT * C)
    idx = lambda i, ans: i * 80 + np.asarray(ans)
    saved = {st: np.zeros((len(repeats), S, 80, NT)) for st in SETTINGS}
    pacc = {st: np.zeros((len(repeats), S, NT * C)) for st in SETTINGS}
    for ri, rep in enumerate(repeats):
        fold = fold_of[rep]; sc = {st: np.zeros((S, 80, NT * C)) for st in SETTINGS + ["zeroshot_grouped"]}; t0 = time.time()
        for f in range(5):
            tr_a = np.flatnonzero(fold != f); te_a = np.flatnonzero(fold == f)
            mu = Rm[:, tr_a].mean(1, keepdims=True); sd = Rm[:, tr_a].std(1, keepdims=True) + 1e-8
            Y = ((Rm - mu) / sd).reshape(S * 80, NT * C)
            Yg = torch.as_tensor(Y, device=DEV, dtype=torch.float32) if DEV else Y
            inner_f = [tr_a[fold[tr_a] == g] for g in range(5) if g != f]

            def zero_shot(train_people, test_people):
                grp = np.array_split(rng.permutation(train_people), 4)
                va = [(np.concatenate([idx(j, np.setdiff1d(tr_a, inner_f[k])) for j in train_people if j not in grp[k]]),
                       np.concatenate([idx(j, inner_f[k]) for j in grp[k]])) for k in range(4)]
                trn = np.concatenate([idx(j, tr_a) for j in train_people])
                tst = np.concatenate([idx(j, np.arange(80)) for j in test_people])
                return ridge_fit_predict(K, trn, tst, Yg, ridge_cv(K, va, Yg)).reshape(len(test_people), 80, -1)

            # zero-shot by groups of people: not used here, but it draws from the same random stream as the published
            # run, so it is kept to reproduce the published predictions exactly
            for g in np.unique(groups):
                mem = np.flatnonzero(groups == g)
                sc["zeroshot_grouped"][np.ix_(mem, te_a)] = zero_shot(np.flatnonzero(groups != g), mem)[:, te_a]
            for i in range(S):
                zs_all = zero_shot(np.delete(np.arange(S), i), [i])[0]
                sc["zeroshot"][i, te_a] = zs_all[te_a]
                own = idx(i, tr_a)
                va_c = [(idx(i, np.setdiff1d(tr_a, inner_f[k])), idx(i, inner_f[k])) for k in range(4)]
                sc["calibrated"][i, te_a] = ridge_fit_predict(K, own, idx(i, te_a), Yg, ridge_cv(K, va_c, Yg))
                if DEV:
                    Yr = Yg.clone(); Yr[_t(own)] -= torch.as_tensor(zs_all[tr_a], device=DEV, dtype=torch.float32)
                else:
                    Yr = Y.copy(); Yr[own] -= zs_all[tr_a]
                sc["finetuned"][i, te_a] = zs_all[te_a] + ridge_fit_predict(K, own, idx(i, te_a), Yr, ridge_cv(K, va_c, Yr))
        for st in SETTINGS:
            saved[st][ri] = sc[st][..., np.arange(NT) * C]
            for i in range(S):
                c = t = 0
                for f in range(5):
                    te_a = np.flatnonzero(fold == f); a, b = pair_counts(sc[st][i, te_a], Rm[i, te_a]); c = c + a; t = t + b
                with np.errstate(invalid="ignore"):
                    pacc[st][ri, i] = c / t                                     # NaN: no untied pair in some test fold
        print(f"repeat {rep}: {time.time() - t0:.0f}s, mean pairwise accuracy (zero-shot / calibrated / fine-tuned) " +
              ", ".join(f"{j} {np.nanmean(pacc['zeroshot'][ri, :, k * C]):.3f} / {np.nanmean(pacc['calibrated'][ri, :, k * C]):.3f} / "
                        f"{np.nanmean(pacc['finetuned'][ri, :, k * C]):.3f}" for k, j in enumerate(JUDGMENTS[:2])), flush=True)
    return saved, {st: v.mean(0) for st, v in pacc.items()}, C


if __name__ == "__main__":
    a = sys.argv[1:]
    n_perm = int(a[a.index("--perm") + 1]) if "--perm" in a else 0
    reps = range(int(a[a.index("--repeats") + 1])) if "--repeats" in a else range(10)
    R, subs, _, _ = ratings()
    saved, pacc, C = evaluate(np.load(WORK / "features_riem.npy"), n_perm=n_perm, repeats=reps)
    np.savez_compressed(WORK / "predictions.npz", **{st: v.astype(np.float32) for st, v in saved.items()},
                        judgments=np.array(JUDGMENTS), subjects=subs, trials=np.arange(1, 81))
    if n_perm:
        rows = []
        for k, j in enumerate(JUDGMENTS):
            for st in SETTINGS:
                acc = pacc[st][:, k * C:(k + 1) * C]; acc = acc[~np.isnan(acc).any(1)]    # people with a full set of pairs
                obs, null = acc[:, 0].mean(), acc[:, 1:].mean(0)
                ci = stats.t.ppf(.975, len(acc) - 1) * acc[:, 0].std(ddof=1) / np.sqrt(len(acc))
                rows.append(dict(judgment=j, setting=st, people=len(acc), pairwise_accuracy=round(100 * obs, 1), ci95=round(100 * ci, 1),
                                 null_95th=round(100 * np.percentile(null, 95), 1), p=(1 + (null >= obs).sum()) / (1 + len(null))))
        RESULTS.mkdir(exist_ok=True); D = pd.DataFrame(rows); D.to_csv(RESULTS / "decoder_accuracy.csv", index=False); print(D.to_string(index=False))
