"""Draw fig_budget.pdf from data/confirm_rows.npz (saved simulator output).

usage: python -B scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
STYLE = Path.home() / ".claude/skills/ml-paper-writing/scripts/paper.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
d = np.load(HERE / "data/confirm_rows.npz")
draw = d["draw_id"]
x = np.log2(1 + d["budgets"])
labels = [str(b) for b in d["budgets"]]
COLORS = {"beta11": "#000000", "level": "#0072B2", "level_exact": "#E69F00",
          "level_exact_kernel": "#009E73", "full": "#D55E00", "full_gamma1": "#CC79A7"}
NAMES = {"beta11": "Beta(1,1)", "level": "level model", "level_exact": "+ same item",
         "level_exact_kernel": "+ similar items", "full": "+ prior, $\\gamma$=0.3 (L2c)",
         "full_gamma1": "prior at $\\gamma$=1"}


def per_draw_mean(rows):
    return pd.DataFrame(rows).groupby(draw).mean().to_numpy()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.3))
for name in ("beta11", "full", "full_gamma1"):
    m = per_draw_mean(d[name])
    mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))
    ax1.plot(x, mu, marker="o", color=COLORS[name], lw=1.3, ms=3)
    ax1.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=COLORS[name], alpha=0.15, lw=0)
ax1.axhline(0.25, color="#999999", lw=0.8, ls="--")
ax1.text(5, 0.2515, "constant 0.5", ha="right", va="bottom", fontsize=7, color="#666666")
ax1.text(2.05, 0.2275, "Beta(1,1)", fontsize=7.5, color=COLORS["beta11"], ha="left", va="bottom")
ax1.text(4.1, 0.1955, "L2c", fontsize=7.5, color=COLORS["full"], ha="left", va="bottom")
ax1.text(0.35, 0.2555, "prior at $\\gamma$=1", fontsize=7.5, color=COLORS["full_gamma1"], ha="left", va="bottom")
ax1.set_xticks(x, labels)
ax1.set_xlabel("labels revealed per group")
ax1.set_ylabel("Brier score")
ax1.set_ylim(0.175, 0.262)
ax1.set_title("(a) Brier per budget")

base = per_draw_mean(d["beta11"])
for k, name in enumerate(("level", "level_exact", "level_exact_kernel", "full", "full_gamma1")):
    diff = per_draw_mean(d[name]) - base
    mu, se = diff.mean(0), diff.std(0, ddof=1) / np.sqrt(len(diff))
    off = (k - 2) * 0.07
    ax2.errorbar(x + off, mu, yerr=1.96 * se, color=COLORS[name], marker="o", ms=2.5, lw=1.1,
                 capsize=1.5, label=NAMES[name])
ax2.axhline(0, color=COLORS["beta11"], lw=0.8)
ax2.set_xticks(x, labels)
ax2.set_xlabel("labels revealed per group")
ax2.set_ylabel("Brier minus Beta(1,1)")
ax2.set_title("(b) Gain over Beta(1,1)")
ax2.legend(loc="upper right", fontsize=6.5, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
ax2.set_ylim(-0.0095, 0.0115)
fig.savefig(HERE / "fig_budget.pdf")
print("wrote fig_budget.pdf")

# ---------------------------------------------------------------- Figure 2: test-pool calibration and final ladder
import json  # noqa: E402

lad = json.loads((HERE / "data/ladder_summary.json").read_text())
off = np.array(json.loads((HERE / "data/official_beta_per_budget.json").read_text()))
off_later = json.loads((HERE / "data/official_beta_later_per_budget.json").read_text())
# Simulated Beta(2,2), the prior of the official baseline archive, with and without test weights.
zb = np.load(HERE / "data/baselines_all_conf.npz")
share = np.array(json.loads((HERE / "data/test_pool_stats.json").read_text())["test_pool"]["bin_share"])
bins = np.minimum((np.abs(zb["p"] - 0.5) / 0.1).astype(int), 4)
w = share[bins] / (np.bincount(bins, minlength=5) / len(bins))[bins]
sim_un, sim_own = zb["beta_a2"].mean(0), (w[:, None] * zb["beta_a2"]).sum(0) / w.sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.3))
for run in off:
    ax1.plot(x, run, color="#999999", lw=0.8, marker="o", ms=2, alpha=0.8)
for run in off_later:
    ax1.plot(x, run, color="#999999", lw=0.8, ls=":", marker="o", ms=2, alpha=0.8)
ax1.plot(x, off.mean(0), color="#000000", lw=1.4, marker="o", ms=3)
ax1.plot(x, sim_un, color="#0072B2", lw=1.3, ls="--", marker="s", ms=2.5)
ax1.plot(x, sim_own, color="#D55E00", lw=1.3, marker="s", ms=2.5)
ax1.text(2.6, 0.224, "simulator, unweighted", fontsize=7, color="#0072B2", ha="left")
ax1.text(0.1, 0.126, "simulator, test weights", fontsize=7, color="#D55E00", ha="left")
ax1.text(3.05, 0.177, "official runs\n(mean in black)", fontsize=7, color="#000000", ha="left", va="center")
ax1.set_xticks(x, labels)
ax1.set_xlabel("labels revealed per group")
ax1.set_ylabel("Brier score of Beta baseline")
ax1.set_title("(a) Calibrating the simulator")
ax1.set_ylim(0.12, 0.26)

# (b) Simulated ALC against the share of test groups below 0.5 (scripts/side_of_half.py).
side = json.loads((HERE / "data/side_of_half.json").read_text())
rec = side["simulated"]["recent"]
pis = np.array(rec["pi"][1:], dtype=float)
COL2 = {"L2c": "#0072B2", "L3": "#000000", "L4": "#009E73", "L5": "#D55E00", "L6": "#CC79A7"}
LAB2 = {"L2c": "L2c (narrow)", "L3": "L3 (wide), final", "L4": "L4 (+ date)", "L5": "L5 (+ name)", "L6": "L6 (tuned)"}
lo = side["summary"]["pi_lower95_L4"]
ax2.axvspan(lo, 1.0, color="#DDDDDD", lw=0)
ax2.axvline(rec["train_share_below"], color="#777777", lw=0.8, ls=":")
for n in ("L2c", "L6", "L5", "L4", "L3"):
    # Percent of the Beta(1,1) ALC at the same share, as in Table 1.
    y = 100 * np.array([m for m, s_ in rec["alc_vs_beta"][n][1:]]) / np.array(rec["beta_alc"][1:])
    ax2.plot(pis, y, color=COL2[n], lw=1.6 if n == "L3" else 1.1, label=LAB2[n])
ax2.axhline(0, color="#000000", lw=0.6)
ax2.text(lo + 0.01, 1.4, "official\nbudget-0\nscores", fontsize=6, va="top", ha="left")
ax2.text(rec["train_share_below"] + 0.02, -6.9, "training data", fontsize=6, va="bottom", ha="left")
ax2.set_xlim(0, 1)
ax2.set_ylim(-7.0, 3.1)
ax2.set_xlabel("share of test groups below 0.5")
ax2.set_ylabel("ALC change vs Beta(1,1), %")
ax2.set_title("(b) Test weights, recent subjects")
handles, labels_ = ax2.get_legend_handles_labels()
order = [labels_.index(LAB2[n]) for n in ("L3", "L4", "L5", "L6", "L2c")]
ax2.legend([handles[i] for i in order], [labels_[i] for i in order], loc="upper left", ncol=2, fontsize=5.8,
           handlelength=1.2, borderaxespad=0.2, labelspacing=0.2, columnspacing=0.8)
fig.savefig(HERE / "fig_testpool.pdf")
print("wrote fig_testpool.pdf")
