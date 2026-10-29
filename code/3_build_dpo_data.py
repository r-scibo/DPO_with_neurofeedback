"""Turn decoded (or rated) judgments into DPO preference data: one dataset per person, judgment and setting.

Comparisons. For one person, every pair of an answer with a source and an answer without one, in the same voice (so a
voice preference cannot pass as a source preference) and, for decoded scores, in the same test fold of the same repeat
(scores from different fold models are not comparable). The answer with the higher score wins; ties are dropped.
Decoded: 10 repeats x 5 folds x 2 voices x 4 x 4 = 1,600 comparisons per person. Real ratings: 2 voices x 20 x 20.
Examples. Every study question has both versions written out (data/study_pairs.csv). A comparison (sourced answer i,
plain answer j) gives two DPO examples, for question(i) and question(j): the version of the winning type is chosen, the
other rejected. So the share of examples with the sourced version chosen equals the share of comparisons it won.
1,000 examples per dataset, drawn at random (with replacement if fewer exist); people with < 20 untied comparisons are
left out. Answer -> question: data/trial_text_map.csv.
Datasets (work/train/<name>.jsonl):
  pp_<setting>_<judgment>_s<person>   per person; setting = zeroshot, calibrated, finetuned (EEG) or real (ratings)
  pp_calibrated-shufmap_interesting_s<person>   same labels, answer -> question map replaced by a random one
  pool_<zeroshot|real>_<judgment>     one dataset from everyone's comparisons
  pool_null<00-38>                    coin-flip winners on the pooled zero-shot Interesting comparisons (null)
-> work/train/*.jsonl, work/train/manifest.csv, work/train/jobs.txt (the training jobs for 4_train_dpo.py)
python 3_build_dpo_data.py [--predictions work/predictions.npz]
"""
import sys, json, zlib
import numpy as np, pandas as pd
from common import DATA, WORK, JUDGMENTS, SETTINGS, ratings, folds

N = 1000; MIN_UNTIED = 20; TRAIN = WORK / "train"


def decoded_comparisons(S, fold_of, src, voice):
    """S: (repeats, 80) one person's scores -> list of (sourced answer, plain answer, sourced won); ties dropped."""
    out = []
    if not np.isfinite(S).all():
        return out
    for rep in range(S.shape[0]):
        for g in range(5):
            for v in (0, 1):
                a = np.flatnonzero((fold_of[rep] == g) & src & (voice == v)); b = np.flatnonzero((fold_of[rep] == g) & ~src & (voice == v))
                d = S[rep, a][:, None] - S[rep, b][None, :]
                out += [(a[x], b[y], d[x, y] > 0) for x, y in zip(*np.nonzero(d != 0))]
    return out


def real_comparisons(r, src, voice):
    out = []
    for v in (0, 1):
        a = np.flatnonzero(src & (voice == v)); b = np.flatnonzero(~src & (voice == v))
        d = r[a][:, None] - r[b][None, :]
        out += [(a[x], b[y], d[x, y] > 0) for x, y in zip(*np.nonzero(d != 0))]
    return out


def examples(comps, rowmap, Q):
    ex = []
    for i, j, won in comps:
        for t in (i, j):
            q = Q.loc[rowmap[t]]; good, bad = (q.sourced, q.plain) if won else (q.plain, q.sourced)
            ex.append(dict(prompt=[{"role": "user", "content": q.question}], chosen=[{"role": "assistant", "content": good}],
                           rejected=[{"role": "assistant", "content": bad}], sourced_chosen=bool(won)))
    return ex


def write(name, ex, meta, rows):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    pick = rng.choice(len(ex), N, replace=len(ex) < N)
    with open(TRAIN / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for k in pick:
            f.write(json.dumps({k_: v for k_, v in ex[k].items() if k_ != "sourced_chosen"}, ensure_ascii=False) + "\n")
    rows.append(dict(name=name, **meta, n_examples_available=len(ex),
                     share_sourced_chosen=round(float(np.mean([e["sourced_chosen"] for e in ex])), 4),
                     share_in_file=round(float(np.mean([ex[k]["sourced_chosen"] for k in pick])), 4)))


if __name__ == "__main__":
    a = sys.argv[1:]; pfile = a[a.index("--predictions") + 1] if "--predictions" in a else DATA / "predictions.npz"
    TRAIN.mkdir(parents=True, exist_ok=True)
    R, subs, src, voice = ratings(); fold_of = folds()
    P = np.load(pfile); assert list(P["judgments"]) == JUDGMENTS
    Q = pd.read_csv(DATA / "study_pairs.csv").set_index("row")
    M = pd.read_csv(DATA / "trial_text_map.csv").set_index("trial").row.loc[np.arange(1, 81)].astype(int).values
    qcls = ((Q.voice == "announcer").astype(int) * 2 + (Q.detail == "detailed")).loc[M].values
    assert (qcls == voice * 2 + src).all()                                  # the map always keeps voice and source right
    Mshuf = M.copy(); rng = np.random.default_rng(0)
    for c in range(4):
        idx = np.flatnonzero(voice * 2 + src == c); Mshuf[idx] = M[rng.permutation(idx)]
    rows = []
    for k, tgt in enumerate(JUDGMENTS):
        Rt = R.pivot(index="subject", columns="trial", values=tgt).loc[subs].values.astype(float)
        pooled = {"zeroshot": [], "real": []}
        for i, s in enumerate(subs):
            for st in SETTINGS:
                comps = decoded_comparisons(P[st][:, i, :, k].astype(float), fold_of, src, voice)
                if st == "zeroshot":
                    pooled["zeroshot"] += comps
                if len(comps) >= MIN_UNTIED:
                    meta = dict(kind="pp", setting=st, target=tgt, subject=s, n_untied=len(comps))
                    write(f"pp_{st}_{tgt}_s{s}", examples(comps, M, Q), meta, rows)
                    if st == "calibrated" and tgt == "interesting":
                        write(f"pp_calibrated-shufmap_{tgt}_s{s}", examples(comps, Mshuf, Q), {**meta, "setting": "calibrated-shufmap"}, rows)
            comps = real_comparisons(Rt[i], src, voice); pooled["real"] += comps
            if len(comps) >= MIN_UNTIED:
                write(f"pp_real_{tgt}_s{s}", examples(comps, M, Q), dict(kind="pp", setting="real", target=tgt, subject=s, n_untied=len(comps)), rows)
        for st in ("zeroshot", "real"):
            write(f"pool_{st}_{tgt}", examples(pooled[st], M, Q), dict(kind="pool", setting=st, target=tgt, n_untied=len(pooled[st])), rows)
        if tgt == "interesting":
            for n in range(39):
                flip = np.random.default_rng([n, 0]).random(len(pooled["zeroshot"])) < 0.5
                write(f"pool_null{n:02d}", examples([(i, j, f) for (i, j, _), f in zip(pooled["zeroshot"], flip)], M, Q),
                      dict(kind="pool", setting="coinflip", target="all", null=n, n_untied=len(pooled["zeroshot"])), rows)
    D = pd.DataFrame(rows); D.to_csv(TRAIN / "manifest.csv", index=False)
    seeds = lambda n: (0, 1, 2) if n.startswith(("pool_zeroshot", "pool_real")) else (0,)    # pooled: 3 training seeds
    (TRAIN / "jobs.txt").write_text("\n".join(["base"] + [f"{n}:{s}" for n in D.name for s in seeds(n)]) + "\n")
    print(D.groupby(["kind", "setting"]).agg(datasets=("name", "size"), share_mean=("share_sourced_chosen", "mean"),
                                             share_min=("share_sourced_chosen", "min"), share_max=("share_sourced_chosen", "max")).round(3))
