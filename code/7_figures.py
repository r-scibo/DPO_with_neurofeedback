"""The blog post's figures, from results/.
-> ../figures/personal_real.svg, personal_calibrated.svg, tradeoff.svg
python 7_figures.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, FIGURES, ratings, real_preference

INK, INK2, GRID = "#111111", "#6b6b6b", "#e6e6e6"
EEG, REAL = "#2a6fdb", "#9a9a9a"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "font.size": 11.5, "svg.fonttype": "none",
                     "svg.hashsalt": "dpo", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0})


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.05, metadata={"Date": None}); plt.close(fig)


if __name__ == "__main__":
    J = pd.read_csv(RESULTS / "dpo_jobs.csv"); PP = pd.read_csv(RESULTS / "perperson.csv").set_index(["judgment", "setting"])
    PO = pd.read_csv(RESULTS / "pooled.csv").set_index(["judgment", "setting"])
    per, _ = real_preference(*ratings())

    # each person's model vs that person's real preference (Interesting): trained on ratings, trained on EEG
    for st, c, head in (("real", REAL, "Trained on their ratings"), ("calibrated", EEG, "Trained on their EEG")):
        fig, ax = plt.subplots(figsize=(3.7, 3.3))
        g = J[(J.kind == "pp") & (J.setting == st) & (J.target == "interesting") & (J.seed == 0)]
        x = per["interesting"].loc[g.subject.astype(int)].values; y = g.d_source_rate.values
        ax.axhline(0, color=INK2, lw=0.8)
        b = np.polyfit(x, y, 1); xx = np.array([x.min(), x.max()]); ax.plot(xx, np.polyval(b, xx), color=c, lw=1.5, alpha=0.6)
        ax.scatter(x, y, s=28, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        q = PP.loc[("interesting", st)]
        ax.set_title(head, loc="left", fontsize=12, color=INK, pad=22)
        ax.text(0, 1.03, f"ρ = {q.rho:.2f}" + ("" if st == "real" else f", p = {q.p:.3f}"), transform=ax.transAxes, color=INK2, fontsize=10.5)
        ax.set_xlabel("Person's preference for sourced answers"); ax.set_ylabel("Change in citing rate (points)")
        fig.tight_layout(); save(fig, f"personal_{st}.svg")

    # one model per judgment, trained on everyone's ratings
    keep = [("impressive", "Impressive"), ("interesting", "Interesting"), ("easy", "Easy to understand"), ("obvious", "Obvious")]
    v = np.array([PO.loc[(k, "real")].d_source_rate for k, _ in keep])
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    y = np.arange(len(keep))[::-1]
    ax.barh(y, v, height=0.56, color=np.where(v > 0, EEG, "#d9822b"))
    for yy, vv in zip(y, v):
        ax.text(vv + (0.4 if vv > 0 else -0.4), yy, f"{vv:+.1f}".replace("-", "−"), va="center", ha="left" if vv > 0 else "right", color=INK, fontsize=11)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([n for _, n in keep], color=INK)
    ax.grid(axis="y", visible=False); ax.set_xlim(-13, 18)
    ax.set_xlabel("Change in citing rate (points)")
    fig.tight_layout(); save(fig, "tradeoff.svg")
    print("saved to", FIGURES)
