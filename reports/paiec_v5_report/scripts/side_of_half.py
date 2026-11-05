"""Side of 0.5: official budget-0 scores of archives that shift predictions, and the simulated ranking.

The Beta baseline predicts 0.5 at budget 0, so its feedback gives |p - 0.5| but not the side.
L4 and L5 add release-date and model-name offsets that raise the prediction for recent subjects.
Their official budget-0 Brier score therefore tells whether test groups lie mostly above or below 0.5.
Only per-archive means over group evaluations leave this script. No predictor reads group identifiers.

The simulator part reweights the saved confirmation rows (data/ladder_*_conf.npz) by the test
shares of |p - 0.5| and, within each bin, by a share pi of groups below 0.5.

usage: python -B scripts/side_of_half.py   (writes data/side_of_half.json)
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
ROW = r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)"
W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
ARCHIVES = {"adaptive_baseline.zip": "Beta22", "beta_probe_20261105.zip": "Beta22", "beta_conc1_probe.zip": "Beta22",
            "L1c_level_exact_cpu.zip": "L1c", "L1_level_exact.zip": "L1c", "L2_prior_exact_kernel.zip": "L2", "L3_wide_prior.zip": "L3",
            "L4_wide_date.zip": "L4", "L5_wide_date_name.zip": "L5",
            "semantic_selected_bundled.zip": "V2", "semantic_v3_bundled.zip": "V3"}
SIM = ["beta", "L1c", "L2c", "wide_level", "wide_exact", "wide_kernel", "L3", "L4", "L5", "L5_no_text", "L6"]
PIS = np.round(np.arange(0, 1.0001, 0.05), 2)


def archive_of():
    rec = ROOT / "private/submissions"
    out = {}
    for p in list(rec.glob("*.json")) + list(rec.glob("*/*.json")):
        if p.name.startswith("._") or p.name == "track_state.json":
            continue
        d = json.loads(p.read_text())
        if d.get("submission_id"):
            out[int(d["submission_id"])] = d["archive"]
    for line in (rec / "resubmits.jsonl").read_text().splitlines():
        d = json.loads(line)
        out[int(d["submission_id"])] = d["archive"]
    return out


def official():
    groups, runs = {}, {}
    for sid, arch in sorted(archive_of().items()):
        name = ARCHIVES.get(arch)
        f = ROOT / f"private/feedback/{sid}_scoring_stdout.txt"
        if not name or not f.exists():
            continue
        parts = re.split(r"Label budget:\s*(\d+)", f.read_text())
        b0 = [float(x[3]) for x in re.findall(ROW, parts[2])]
        if not b0:
            continue
        groups.setdefault(name, []).extend(b0)
        runs.setdefault(name, []).append({"id": sid, "b0": statistics.fmean(b0)})
    out = {}
    for name, v in groups.items():
        out[name] = {"runs": runs[name], "group_evaluations": len(v), "b0": statistics.fmean(v),
                     "se": statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")}
    return out


def weights(p, pi, share):
    bins = np.minimum((np.abs(p - 0.5) / 0.1).astype(int), 4)
    ts = np.bincount(bins, minlength=5) / len(bins)
    w = share[bins] / ts[bins]
    if pi is None:
        return w
    low = p < 0.5
    for b in range(5):
        m = bins == b
        lo = (m & low).sum() / max(m.sum(), 1)
        w[m & low] *= pi / max(lo, 1e-9)
        w[m & ~low] *= (1 - pi) / max(1 - lo, 1e-9)
    return w


def ratio_se(w, d, draw):
    df = pd.DataFrame({"W": w, "D": w * d, "k": draw}).groupby("k").sum()
    r = df.D.sum() / df.W.sum()
    k = len(df)
    return float(r), float(math.sqrt(((df.D - r * df.W) ** 2).sum() * k / (k - 1)) / df.W.sum())


def crossing(xs, ys, target):
    """Smallest x where the increasing curve ys reaches target, by linear interpolation; None if never."""
    for i in range(1, len(xs)):
        if ys[i - 1] < target <= ys[i]:
            return float(xs[i - 1] + (target - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1]))
    return 0.0 if ys[0] >= target else None


def simulated(share):
    out = {}
    for pop in ("recent", "all"):
        z = np.load(HERE / f"data/ladder_{pop}_conf.npz")
        p, draw = z["p"], z["draw_id"]
        alc = {m: z[m] @ W6 for m in SIM}
        z3 = np.load(HERE / f"data/baselines_l3_{pop}_conf.npz")
        assert np.allclose(z3["L3"], z["L3"]) and np.allclose(z3["beta"], z["beta"])
        alc["L3_own"] = z3["L3_own"] @ W6
        res = {"train_share_below": float(np.mean(p < 0.5)), "pi": [], "b0": {m: [] for m in ("L3", "L4", "L5")},
               "alc_vs_beta": {m: [] for m in SIM[1:]}, "L5_minus_L3": [], "L4_minus_L3": [], "beta_alc": [],
               "L3_minus_L3_own": []}
        for pi in [None] + list(PIS):
            w = weights(p, pi, share)
            res["pi"].append(pi)
            for m in ("L3", "L4", "L5"):
                res["b0"][m].append(float(np.average(z[m][:, 0], weights=w)))
            for m in SIM[1:]:
                res["alc_vs_beta"][m].append(ratio_se(w, alc[m] - alc["beta"], draw))
            res["beta_alc"].append(float(np.average(alc["beta"], weights=w)))
            res["L3_minus_L3_own"].append(ratio_se(w, alc["L3"] - alc["L3_own"], draw))
            res["L5_minus_L3"].append(ratio_se(w, alc["L5"] - alc["L3"], draw))
            res["L4_minus_L3"].append(ratio_se(w, alc["L4"] - alc["L3"], draw))
        out[pop] = res
    return out


def main():
    share = np.array(json.loads((HERE / "data/test_pool_stats.json").read_text())["test_pool"]["bin_share"])
    off = official()
    sim = simulated(share)
    rec = sim["recent"]
    grid = slice(1, None)                       # skip the entry without side weights
    pis = np.array(rec["pi"][grid], dtype=float)
    gaps = {}
    for m in ("L4", "L5"):
        g_off = off[m]["b0"] - off["L3"]["b0"]
        g_se = math.sqrt(off[m]["se"] ** 2 + off["L3"]["se"] ** 2)
        g_sim = np.array(rec["b0"][m][grid]) - np.array(rec["b0"]["L3"][grid])
        gaps[m] = {"official": g_off, "official_se": g_se, "sim_at_train_side": rec["b0"][m][0] - rec["b0"]["L3"][0],
                   "sim_max": float(g_sim.max()), "pi_match": crossing(pis, g_sim, g_off),
                   "pi_lower95": crossing(pis, g_sim, g_off - 1.645 * g_se)}
    d53 = np.array([v[0] for v in rec["L5_minus_L3"][grid]])
    breakeven = crossing(pis, d53, 0.0)

    def at(pi_value, key):
        i = int(np.argmin(np.abs(pis - pi_value)))
        return rec[key][grid][i]

    pi_lo = gaps["L4"]["pi_lower95"]
    summary = {"breakeven_pi": breakeven, "pi_lower95_L4": pi_lo,
               "L5_minus_L3_at_train_side": rec["L5_minus_L3"][0], "L5_minus_L3_at_pi_lower95": at(pi_lo, "L5_minus_L3"),
               "L5_minus_L3_at_pi_one": at(1.0, "L5_minus_L3"),
               "L3_vs_beta_range": [min(v[0] for v in rec["alc_vs_beta"]["L3"][grid]),
                                    max(v[0] for v in rec["alc_vs_beta"]["L3"][grid])],
               "L5_vs_beta_range": [min(v[0] for v in rec["alc_vs_beta"]["L5"][grid]),
                                    max(v[0] for v in rec["alc_vs_beta"]["L5"][grid])],
               "wide_kernel_minus_L3_range": [min(k[0] - t[0] for k, t in zip(rec["alc_vs_beta"]["wide_kernel"],
                                                                                rec["alc_vs_beta"]["L3"])),
                                              max(k[0] - t[0] for k, t in zip(rec["alc_vs_beta"]["wide_kernel"],
                                                                                rec["alc_vs_beta"]["L3"]))],
               "all_L5_minus_L3_range": [min(v[0] for v in sim["all"]["L5_minus_L3"][grid]),
                                         max(v[0] for v in sim["all"]["L5_minus_L3"][grid])]}
    # Single-archive bounds: where each simulated budget-0 curve meets the official score.
    b0 = {m: np.array(rec["b0"][m][grid]) for m in ("L3", "L4", "L5")}
    summary["pi_from_L3"] = crossing(pis, b0["L3"], off["L3"]["b0"])
    summary["pi_from_L4"] = crossing(pis, b0["L4"], off["L4"]["b0"])
    summary["pi_from_L5"] = crossing(pis, b0["L5"], off["L5"]["b0"])
    summary["pi_from_L4_lower95"] = crossing(pis, b0["L4"], off["L4"]["b0"] - 1.645 * off["L4"]["se"])
    summary["pi_lower95_L4_two_sided"] = crossing(pis, np.array(rec["b0"]["L4"][grid]) - np.array(rec["b0"]["L3"][grid]),
                                                  gaps["L4"]["official"] - 1.96 * gaps["L4"]["official_se"])
    summary["L3_b0_sim_at_pi_lower95"] = float(b0["L3"][int(np.argmin(np.abs(pis - pi_lo)))])
    summary["L6_crosses_beta"] = crossing(pis, np.array([v[0] for v in rec["alc_vs_beta"]["L6"][grid]]), 0.0)
    rel = [100 * v[0] / b for v, b in zip(rec["alc_vs_beta"]["L3"], rec["beta_alc"])]
    hi = [r for r, x in zip(rel[1:], pis) if x >= pi_lo - 1e-9]
    summary["L3_rel_at_train_side"] = rel[0]
    summary["L3_rel_high_pi_range"] = [min(hi), max(hi)]
    own = rec["L3_minus_L3_own"]
    summary["L3_share_loss_at_train_side"] = own[0][0]
    summary["L3_share_loss_at_pi_lower95"] = at(pi_lo, "L3_minus_L3_own")[0]
    summary["L3_vs_beta_at_pi_lower95"] = rec["alc_vs_beta"]["L3"][grid][int(np.argmin(np.abs(pis - pi_lo)))][0]
    summary["plotted_se_max"] = max(v[1] for m in ("L2c", "L3", "L4", "L5", "L6") for v in rec["alc_vs_beta"][m][grid])
    summary["L5_minus_L3_se_max"] = max(v[1] for v in rec["L5_minus_L3"][grid])
    out = {"official_b0": off, "gaps": gaps, "simulated": sim, "summary": summary}
    (HERE / "data/side_of_half.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, v in sorted(off.items()):
        print(f"{name:6} runs {len(v['runs'])} groups {v['group_evaluations']:3d} b0 {v['b0']:.4f} se {v['se']:.4f}")
    print(json.dumps(gaps, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
