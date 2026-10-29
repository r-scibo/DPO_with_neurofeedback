"""Does each model follow its person, and does one model trained on everyone follow the crowd?

Per person: for each judgment and setting, Spearman rho across people between the change in the model's citing rate and
the person's real preference (share of their within-voice (sourced, plain) pairs where they rated the sourced answer
higher). One-sided p from 10,000 shuffles of the models across people; Holm across the 9 judgments within each setting.
The test registered in advance: Interesting, calibrated decoder. Real ratings are the positive control (not tested).
Pooled: change in citing rate (mean of 3 training seeds) against 39 models trained on coin-flip labels, one-sided in the
direction of the crowd's real preference.
-> results/perperson.csv, results/pooled.csv, and the numbers quoted in the blog post
python 6_stats.py
"""
import numpy as np, pandas as pd
from common import RESULTS, JUDGMENTS, SETTINGS, ratings, real_preference, shuffle_test, holm

if __name__ == "__main__":
    J = pd.read_csv(RESULTS / "dpo_jobs.csv"); R, subs, src, voice = ratings()
    per, crowd = real_preference(R, subs, src, voice); rng = np.random.default_rng(0)
    rows = []
    for tgt in JUDGMENTS:                                    # this order fixes the random stream of the shuffle tests
        for st in SETTINGS + ["real", "calibrated-shufmap"]:
            g = J[(J.kind == "pp") & (J.setting == st) & (J.target == tgt) & (J.seed == 0)]
            if len(g):
                rp = per[tgt].loc[g.subject.astype(int)].values
                row = dict(judgment=tgt, setting=st, n=len(g), mean_d_source=g.d_source_rate.mean(), mean_d_margin=g.d_margin.mean(),
                           mean_d_accuracy=g.d_accuracy.mean())
                row["rho"], row["p"] = shuffle_test(g.d_source_rate.values, rp, rng)
                row["rho_margin"], row["p_margin"] = shuffle_test(g.d_margin.values, rp, rng)
                rows.append(row)
    PP = pd.DataFrame(rows)
    for st in SETTINGS:
        m = PP.setting == st; PP.loc[m, "p_holm"] = holm(PP.loc[m, "p"].values)
    PP["registered"] = (PP.judgment == "interesting") & (PP.setting == "calibrated")
    PP.to_csv(RESULTS / "perperson.csv", index=False)

    null = J[(J.kind == "pool") & (J.setting == "coinflip")]; rows = []
    for tgt in JUDGMENTS:
        sign = 1 if crowd[tgt] > 0.5 else -1
        for st in ("zeroshot", "real"):
            g = J[(J.kind == "pool") & (J.setting == st) & (J.target == tgt)]
            row = dict(judgment=tgt, setting=st, crowd_real_preference=crowd[tgt], direction="up" if sign > 0 else "down", seeds=len(g),
                       share_in_data=g.share_sourced_chosen.iloc[0], d_source_rate=g.d_source_rate.mean(), d_margin=g.d_margin.mean(),
                       d_accuracy=g.d_accuracy.mean(), null_n=len(null))
            row["p"] = (1 + np.sum(sign * null.d_source_rate.values >= sign * g.d_source_rate.mean())) / (1 + len(null))
            row["p_margin"] = (1 + np.sum(sign * null.d_margin.values >= sign * g.d_margin.mean())) / (1 + len(null))
            rows.append(row)
    PO = pd.DataFrame(rows); z = PO.setting == "zeroshot"; PO.loc[z, "p_holm"] = holm(PO.loc[z, "p"].values)
    PO.to_csv(RESULTS / "pooled.csv", index=False)

    q = PP.set_index(["judgment", "setting"]); o = PO.set_index(["judgment", "setting"]); base = J[J.job == "base"].iloc[0]
    eeg = J[(J.kind == "pp") & J.setting.isin(SETTINGS)]
    cal = J[(J.kind == "pp") & (J.setting == "calibrated") & (J.target == "interesting") & (J.seed == 0)]
    rat = J[(J.kind == "pp") & (J.setting == "real") & (J.target == "interesting") & (J.seed == 0)]
    print(f"Base model: cites in {base.source_rate:.0f} % of answers, {base.accuracy:.0f} % correct; "
          f"largest accuracy loss of any EEG-trained model: {-eeg.d_accuracy.min():.1f} points")
    print(f"EEG preference shares (Interesting, calibrated): {cal.share_sourced_chosen.min():.0%} to {cal.share_sourced_chosen.max():.0%}; "
          f"ratings: {rat.share_sourced_chosen.min():.0%} to {rat.share_sourced_chosen.max():.0%}")
    r = q.loc[("interesting", "calibrated")]
    print(f"Each model follows its person (Interesting): EEG rho = {r.rho:.2f}, p = {r.p:.3f} (registered); "
          f"ratings rho = {q.loc[('interesting', 'real')].rho:.2f}")
    print(f"Change in citing, EEG models {cal.d_source_rate.min():+.0f} to {cal.d_source_rate.max():+.0f} points; "
          f"rating models {rat.d_source_rate.min():+.0f} to {rat.d_source_rate.max():+.0f}")
    print("One model trained on everyone's ratings: " + ", ".join(f"{j} {o.loc[(j, 'real')].d_source_rate:+.1f}"
                                                                 for j in ("impressive", "interesting", "easy", "obvious")))
    e = q.loc[("easy", "zeroshot")]
    print(f"Easy to understand, EEG (zero-shot decoder): rho = {e.rho:.2f}, p = {e.p:.3f}, Holm across 9 judgments p = {e.p_holm:.3f}")
    z = PO[PO.setting == "zeroshot"]
    print(f"One model trained on everyone's EEG (zero-shot): smallest p across the 9 judgments = {z.p.min():.2f}")
