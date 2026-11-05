"""Write numbers.tex, the only source of numbers quoted in main.tex.

Reads saved artifacts only (no training, no prediction, no network).
usage: python -B scripts/make_numbers.py
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import datetime
import glob
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus_full"
CV = ROOT / "colab_jobs/results/bake2/work/results"
OUT = {}


def put(name, value, fmt=None):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    OUT[name] = value if fmt is None else format(value, fmt)
    if fmt and fmt.startswith("+"):
        # Typeset the sign as a math minus or plus, keep the magnitude separately.
        OUT[name] = ("$-$" if value < 0 else "$+$") + format(abs(value), fmt[1:])
        OUT[name + "Abs"] = format(abs(value), fmt[1:])


def thousands(n):
    return f"{int(n):,}"


# ------------------------------------------------------------ corpus
log = (CORPUS / "build.log").read_text()
n_all, n_bin = map(int, re.search(r"benchmarks (\d+) binary (\d+)", log).groups())
skipped_fetch = re.findall(r"skip (\w+) FileNotFoundError", log)
stats = json.loads((CORPUS / "stats.json").read_text())
no_binary = [k for k, v in stats.items() if "skipped" in v]
kept = {k: v for k, v in stats.items() if "skipped" not in v}
pairs = pq.read_table(CORPUS / "pairs.parquet", columns=["benchmark", "subject_key", "n"]).to_pandas()
groups = pairs.groupby(["benchmark", "subject_key"]).size()
subjects = pq.read_table(CORPUS / "subjects.parquet").to_pandas()
put("NBenchAll", n_all)
put("NBenchBinary", n_bin)
put("NBenchNotFetched", len(skipped_fetch))
put("NBenchInStats", len(stats))
put("NBenchNoBinary", len(no_binary))
put("NBenchKept", pairs.benchmark.nunique())
put("NItems", thousands(sum(v["items"] for v in kept.values())))
put("NPairs", thousands(len(pairs)))
put("NTrials", thousands(pairs.n.sum()))
put("NSubjectConfigs", thousands(len(subjects)))
put("NSubjectsWithPairs", thousands(pairs.subject_key.nunique()))
put("NGroupsEighty", thousands((groups >= 80).sum()))
put("NBenchGroupsEighty", groups[groups >= 80].reset_index().benchmark.nunique())
put("NBenchCapped", sum(1 for v in kept.values() if v["items"] == 4000))
put("PctName", 100 * (subjects.normalized_name != "").mean(), ".0f")
put("PctRelease", 100 * (subjects.release_date != "").mean(), ".0f")
put("PctEffort", 100 * (subjects.reasoning_effort != "").mean(), ".1f")
OUT["SkippedFetch"] = ", ".join(s.replace("_", r"\_") for s in skipped_fetch)
OUT["SkippedNoBinary"] = ", ".join(s.replace("_", r"\_") for s in no_binary)

# ------------------------------------------------------------ leave-benchmark-out prior study
def cv(tag):
    r = json.loads((CV / f"cv_{tag}.json").read_text())
    grid = {k: statistics.fmean(v) for k, v in r["grid"].items()}
    best_by_gamma = {}
    for key, val in grid.items():
        gamma = float(key.split("|")[0].strip("()").split(",")[2])
        best_by_gamma[gamma] = min(best_by_gamma.get(gamma, 9.0), val)
    return r, grid, best_by_gamma


for tag, name in [("A", "A"), ("B", "B"), ("C", "C"), ("cv1", "Init")]:
    r, grid, bbg = cv(tag)
    best = min(grid.values())
    beta = r["baseline_mean"]["beta"]
    put(f"Lobo{name}", best, ".5f")
    put(f"Lobo{name}Rel", 100 * (best - beta) / beta, "+.2f")
    if tag == "B":
        put("LoboBeta", beta, ".5f")
        put("LoboFolds", len(r["folds"]))
        put("LoboGroups", thousands(sum(f["groups"] for f in r["folds"])))
        put("LoboBenchmarks", sum(f["benchmarks"] for f in r["folds"]))
        for g, word in [(0.3, "Three"), (0.45, "FourFive"), (0.6, "Six"), (0.75, "SevenFive"), (0.9, "Nine")]:
            put(f"LoboBGamma{word}", bbg[g], ".5f")
        pb = r["best"][2]
        put("LoboBBzero", pb[0], ".4f")
    if tag == "cv1":
        put("LoboInitGammaMin", min(bbg), ".1f")

for tag, name in [("bake_e5", "BakeEfive"), ("bake_qwen06", "BakeQwen")]:
    r, grid, bbg = cv(tag)
    best = min(grid.values())
    put(name, best, ".5f")
    put(f"{name}Rel", 100 * (best - r["baseline_mean"]["beta"]) / r["baseline_mean"]["beta"], "+.2f")
    put("BakeBeta", r["baseline_mean"]["beta"], ".5f")

r = json.loads((CV / "cv_item1.json").read_text())
grid = {k: statistics.fmean(v) for k, v in r["grid"].items()}
put("LoboKernel", min(grid.values()), ".5f")
put("LoboKernelBeta", r["baseline_mean"]["beta"], ".5f")

# ------------------------------------------------------------ official-structure simulator, development draws
dev = json.loads((ROOT / "colab_jobs/results/sim_official_1.json").read_text())
base = dev["beta"]["alc"]
put("DevBeta", base, ".5f")
for key, name in [("level", "Level"), ("exact4", "Exact"), ("exact4_k03_15", "Kernel"),
                  ("z03_exact4_k03", "Prior"), ("z03_exact4", "PriorNoKernel"), ("z06_exact4", "PriorSix")]:
    put(f"Dev{name}", dev[key]["alc"] - base, "+.4f")

grid_lines = (HERE / "data/grid_top_400draws.txt").read_text().splitlines()
put("GridN", int(grid_lines[0].split()[0]))
top = re.search(r"ALC ([0-9.]+) vs beta ([-+0-9.]+) \(se ([0-9.]+)\) per budget \[(.*?)\]", grid_lines[2])
grid_alc, grid_diff, grid_se = float(top.group(1)), float(top.group(2)), float(top.group(3))
put("GridBest", grid_alc, ".5f")
put("GridDiff", grid_diff, "+.5f")
put("GridSe", grid_se, ".5f")
put("GridRel", 100 * grid_diff / (grid_alc - grid_diff), "+.1f")
put("GridBeta", grid_alc - grid_diff, ".5f")

# ------------------------------------------------------------ confirmation draws
conf = json.loads((HERE / "data/confirm_summary.json").read_text())
put("ConfDraws", conf["draws"])
put("ConfSeedZero", thousands(conf["seed0"]))
put("ConfSeedLast", thousands(conf["seed0"] + conf["draws"] - 1))
put("ConfGroups", thousands(conf["groups"]))
cv_ = conf["variants"]
put("ConfBeta", cv_["beta11"]["alc"], ".5f")
for key, name in [("beta22", "BetaTwo"), ("level", "Level"), ("level_exact", "Exact"),
                  ("level_exact_kernel", "Kernel"), ("full", "Full"), ("prior_only", "PriorOnly"),
                  ("full_gamma1", "GammaOne")]:
    v = cv_[key]
    put(f"Conf{name}", v["diff_vs_beta11"], "+.5f")
    put(f"Conf{name}Se", v["se_diff_draws"], ".5f")
    put(f"Conf{name}Rel", v["rel_diff_pct"], "+.2f")
    put(f"Conf{name}Alc", v["alc"], ".5f")
put("ConfFullBzero", cv_["full"]["per_budget"][0], ".4f")
put("ConfFullBlast", cv_["full"]["per_budget"][-1], ".4f")
put("ConfBetaBlast", cv_["beta11"]["per_budget"][-1], ".3f")
put("ConfPriorGain", cv_["full"]["diff_vs_beta11"] - cv_["level_exact_kernel"]["diff_vs_beta11"], "+.5f")
put("ConfShareStructure", 100 * cv_["level_exact_kernel"]["diff_vs_beta11"] / cv_["full"]["diff_vs_beta11"], ".0f")
draw_sd = cv_["beta11"]["se_alc_draws"] * math.sqrt(conf["draws"])
pair_sd = cv_["full"]["se_diff_draws"] * math.sqrt(conf["draws"])
put("SimDrawSd", draw_sd, ".3f")
put("SimPairSd", pair_sd, ".4f")
import numpy as np  # noqa: E402
rows = np.load(HERE / "data/confirm_rows.npz")
sizes = pd.Series(rows["draw_id"]).value_counts().value_counts()
put("ConfNine", int(sizes.get(9, 0)))
put("ConfEight", int(sizes.get(8, 0)))
put("ConfSeven", int(sizes.get(7, 0)))

# ------------------------------------------------------------ official formative runs
archive_of = {}
for p in glob.glob(str(ROOT / "private/submissions/*.json")) + glob.glob(str(ROOT / "private/submissions/*/*.json")):
    if Path(p).name.startswith("._") or Path(p).name == "track_state.json":
        continue
    d = json.loads(Path(p).read_text())
    if d.get("submission_id"):
        archive_of[int(d["submission_id"])] = d["archive"]
for line in (ROOT / "private/submissions/resubmits.jsonl").read_text().splitlines():
    d = json.loads(line)
    archive_of[int(d["submission_id"])] = d["archive"]
runs = []
for p in sorted(glob.glob(str(ROOT / "private/feedback/*_scoring_stdout.txt"))):
    sid = int(Path(p).name.split("_")[0])
    text = Path(p).read_text()
    parts = re.split(r"Label budget:\s*(\d+)", text)
    row = r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)"
    summary = re.findall(row, parts[0])
    per_budget = [statistics.fmean(float(x[3]) for x in re.findall(row, parts[i + 1])) for i in range(1, len(parts), 2)]
    eval_pairs = [int(x[2]) for x in summary]
    runs.append({"pairs": eval_pairs, "id": sid, "archive": archive_of.get(sid, "?"), "groups": len(summary),
                 "benchmarks": len({x[1] for x in summary}), "alc": statistics.fmean(float(x[3]) for x in summary),
                 "b0": per_budget[0], "b31": per_budget[-1],
                 "multi": sorted(pd.Series([x[1] for x in summary]).value_counts().tolist(), reverse=True)})
all_groups = set()
all_bench = set()
for p in glob.glob(str(ROOT / "private/feedback/*_scoring_stdout.txt")):
    head = Path(p).read_text().split("Label budget")[0]
    for s, b, *_ in re.findall(r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)", head):
        all_groups.add((s, b))
        all_bench.add(b)
put("OffTables", len(runs))
all_pairs = [n for r in runs for n in r["pairs"]]
put("OffPairsMin", min(all_pairs))
put("OffPairsMax", max(n for n in all_pairs if n <= 100))
put("OffPairsLarge", sum(1 for n in all_pairs if n > 100))
put("OffPairsLargeValue", max(all_pairs))
put("OffTablesNine", sum(1 for r in runs if r["groups"] == 9 and r["multi"][:2] == [2, 2]))
put("OffTablesEight", sum(1 for r in runs if r["groups"] == 8))
assert all(r["benchmarks"] == 7 for r in runs) and \
    sum(1 for r in runs if r["groups"] == 9 and r["multi"][:2] == [2, 2]) + sum(1 for r in runs if r["groups"] == 8) == len(runs)
put("OffGroupEvals", sum(r["groups"] for r in runs))
put("OffDistinctGroups", len(all_groups))
put("OffDistinctBench", len(all_bench))
# The test weights and every calibration number use the three Beta runs that existed when the
# model was chosen. Later Beta runs are reported separately in Table 4 as an outside check.
CALIB_BETA_IDS = {1003760, 1032881, 1032882}
for sid, name in list(archive_of.items()):
    if name in ("beta_probe_20261105.zip", "beta_conc1_probe.zip"):  # baseline files re-uploaded as queue probes
        archive_of[sid] = "adaptive_baseline.zip"
for r in runs:
    r["archive"] = archive_of.get(r["id"], r["archive"])
assert {r["id"] for r in runs if r["archive"] == "adaptive_baseline.zip"} >= CALIB_BETA_IDS
beta_runs = [r["alc"] for r in runs if r["id"] in CALIB_BETA_IDS]
later_beta = [r["alc"] for r in sorted(runs, key=lambda r: r["id"])
              if r["archive"] == "adaptive_baseline.zip" and r["id"] not in CALIB_BETA_IDS]
put("OffBetaLaterN", len(later_beta))
put("OffBetaAllSd", statistics.stdev(beta_runs + later_beta), ".4f")
put("OffBetaLaterRuns", ", ".join(f"{v:.4f}" for v in later_beta))
if later_beta:
    put("OffBetaLaterMean", statistics.fmean(later_beta), ".4f")
OUT["TabBetaLaterRow"] = (rf"Beta(2,2), later runs & {len(later_beta)} & "
                          + ", ".join(f"{v:.4f}" for v in later_beta)
                          + (f" & {statistics.fmean(later_beta):.4f} ({statistics.stdev(later_beta) / math.sqrt(len(later_beta)):.4f})"
                             if len(later_beta) > 1 else " & ") + r" \\") if later_beta else ""
put("OffBetaN", len(beta_runs))
put("OffBetaMean", statistics.fmean(beta_runs), ".4f")
put("OffBetaSd", statistics.stdev(beta_runs), ".4f")
put("OffBetaSe", statistics.stdev(beta_runs) / math.sqrt(len(beta_runs)), ".4f")
put("OffBetaRuns", ", ".join(f"{v:.4f}" for v in beta_runs))
put("OffBetaRange", max(beta_runs) - min(beta_runs), ".4f")
by_archive = {r["archive"]: r for r in sorted(runs, key=lambda r: -r["id"])}  # earliest run per archive
put("OffVtwo", by_archive["semantic_selected_bundled.zip"]["alc"], ".4f")

# Table rows for V2 and L5: finished runs from feedback files, waiting runs from the queue snapshot.
unscored = json.loads((HERE / "data/unscored_runs.json").read_text())
finished_ids = {r["id"] for r in runs}
waiting = [w for w in unscored["waiting"] if w["id"] not in finished_ids]
checked = datetime.datetime.strptime(unscored["checked_utc"], "%Y-%m-%dT%H:%MZ")
OUT["UnscoredAsOf"] = f"{checked.day} {checked:%B} {checked:%H:%M} UTC"
for archive, name in [("semantic_selected_bundled.zip", "Vtwo"), ("semantic_v3_bundled.zip", "Vthree"),
                      ("L5_wide_date_name.zip", "Lfive")]:
    alcs = [r["alc"] for r in sorted(runs, key=lambda r: r["id"]) if r["archive"] == archive]
    n_wait = sum(1 for w in waiting if w["archive"] == archive)
    OUT[f"Tab{name}Runs"] = f"{len(alcs)}" + (f" ({n_wait} queued)" if n_wait else "")
    OUT[f"Tab{name}Alc"] = ", ".join(f"{v:.4f}" for v in alcs) if alcs else "none"
    OUT[f"Tab{name}Mean"] = (f"{statistics.fmean(alcs):.4f} ({statistics.stdev(alcs) / math.sqrt(len(alcs)):.4f})"
                             if len(alcs) > 1 else "")
    put(f"Off{name}N", len(alcs))
    if alcs:
        put(f"Off{name}Mean", statistics.fmean(alcs), ".4f")
        put(f"Off{name}BzeroMean", statistics.fmean(r["b0"] for r in runs if r["archive"] == archive), ".4f")
l5_alcs = [r["alc"] for r in runs if r["archive"] == "L5_wide_date_name.zip"]
if l5_alcs:
    OUT["LfiveOfficialSentence"] = (
        f"Finally, the {len(l5_alcs)} official run{'s' if len(l5_alcs) > 1 else ''} of L5 show that it works on the platform "
        r"but cannot confirm its simulated gain, which needs about \RunsNeededOurs\ runs per archive.")
else:
    OUT["LfiveOfficialSentence"] = (
        r"Finally, the stalled platform queue left L5 without an official run by \UnscoredAsOf.")
put("OffVtwoBzero", by_archive["semantic_selected_bundled.zip"]["b0"], ".4f")
put("OffVthree", by_archive["semantic_v3_bundled.zip"]["alc"], ".4f")
put("OffVthreeBzero", by_archive["semantic_v3_bundled.zip"]["b0"], ".4f")
blast = [r["b31"] for r in runs if r["id"] in CALIB_BETA_IDS]
put("OffBetaBlastMin", min(blast), ".3f")
put("OffBetaBlastMax", max(blast), ".3f")
(HERE / "data/official_runs.json").write_text(json.dumps(runs, indent=1) + "\n")

# Power for an unpaired comparison of run means, two-sided 5% test, 80% power.
z = norm.ppf(0.975) + norm.ppf(0.8)
delta = abs(cv_["full"]["diff_vs_beta11"])
all_beta = beta_runs + later_beta  # every run of the Beta archive, for run-to-run spread
for sd, name in [(0.016, "Low"), (0.024, "High"), (statistics.stdev(all_beta), "Ours")]:
    put(f"RunsNeeded{name}", math.ceil(2 * z * z * sd * sd / (delta * delta)))
put("SpreadRatio", (max(all_beta) - min(all_beta)) / delta, ".0f")
put("OffBetaAllN", len(all_beta))
put("OffBetaAllRuns", ", ".join(f"{v:.4f}" for v in all_beta))
put("OffBetaAllMean", statistics.fmean(all_beta), ".4f")
put("OffBetaAllSe", statistics.stdev(all_beta) / math.sqrt(len(all_beta)), ".4f")
put("OffBetaAllRange", max(all_beta) - min(all_beta), ".4f")
put("RunsNeededPaired", math.ceil(z * z * pair_sd * pair_sd / (delta * delta)))
# Expected best of 20 runs below the mean, Blom approximation.
put("BestOfTwentyShift", -norm.ppf((1 - 0.375) / (20 + 0.25)) * 0.016, ".3f")

put("ConfGammaOneBzero", cv_["full_gamma1"]["per_budget"][0], ".4f")
put("ConfFullSeRel", 100 * cv_["full"]["se_diff_draws"] / cv_["beta11"]["alc"], ".2f")
dmean = {n: pd.DataFrame(rows[n] @ rows["weights"]).groupby(rows["draw_id"]).mean()[0].to_numpy()
         for n in ("full", "level_exact_kernel", "prior_only", "level")}
for a, b, name in [("full", "level_exact_kernel", "PriorStep"), ("prior_only", "level", "PriorLevel")]:
    dd = dmean[a] - dmean[b]
    put(f"Conf{name}", dd.mean(), "+.5f")
    put(f"Conf{name}Se", dd.std(ddof=1) / math.sqrt(len(dd)), ".5f")

# ------------------------------------------------------------ compute and archives
emb_line = re.search(r"\[\s*(\d+)s\] done items (\d+) subjects (\d+)",
                     (ROOT / "colab_jobs/results/main2/run/job.log").read_text())
put("EmbedSeconds", int(emb_line.group(1)))
put("ArchiveFullMB", (ROOT / "dist/v5/L2c_prior_exact_kernel_cpu.zip").stat().st_size / 1e6, ".1f")
put("ArchiveLevelMB", (ROOT / "dist/v5/L1c_level_exact_cpu.zip").stat().st_size / 1e6, ".1f")
put("PriorMB", (ROOT / "colab_jobs/results/exportB/work/results/prior_B.npz").stat().st_size / 1e6, ".1f")
judge = re.findall(r"rated \d+ of (\d+)", (ROOT / "colab_jobs/results/watch_judge2.log").read_text())
put("JudgeItems", thousands(judge[-1]))

# ------------------------------------------------------------ previous release (V3), from the earlier report
old = (ROOT / "reports/paiec_semantic_report/main.tex").read_text()
m = re.search(r"Total training becomes ([0-9,]+) responses, (\d+) groups", old)
put("VthreeResponses", m.group(1))
put("VthreeGroups", m.group(2))
put("TrialRatio", pairs.n.sum() / int(m.group(1).replace(",", "")), ".0f")
m = re.search(r"They contain ([0-9,]+) admitted binary responses", old)
put("VtwoResponses", m.group(1))

# Appendix table of every finished formative run.
DESC = {"adaptive_baseline.zip": r"Beta(2,2) baseline", "hierarchical_prior.zip": r"V1 hierarchical prior",
        "semantic_selected_bundled.zip": "V2", "semantic_v3_bundled.zip": "V3",
        "semantic_v3_uncertainty_band015.zip": r"V3 with uncertainty band", "L5_wide_date_name.zip": "L5"}
rows_tex = []
for r in sorted(runs, key=lambda r: r["id"]):
    rows_tex.append(f"{r['id']} & {DESC.get(r['archive'], r['archive'])} & {r['groups']} & {r['benchmarks']} & "
                    f"{r['alc']:.4f} & {r['b0']:.4f} & {r['b31']:.4f} \\\\")
OUT["OfficialRows"] = "\n".join(rows_tex)



# ====================================================================== revision 2: test pool, ladder, replays, judge
def ratio_se(w, d, draw):
    """Weighted mean of d over groups and its standard error with draws as clusters."""
    df = pd.DataFrame({"W": w, "D": w * d, "k": draw}).groupby("k").sum()
    r = df.D.sum() / df.W.sum()
    k = len(df)
    return float(r), float(math.sqrt(((df.D - r * df.W) ** 2).sum() * k / (k - 1)) / df.W.sum())


SIM_SHARE = np.array([0.115, 0.077, 0.192, 0.308, 0.308])
tps = json.loads((HERE / "data/test_pool_stats.json").read_text())
OWN_SHARE = np.array(tps["test_pool"]["bin_share"])
put("TpGroups", tps["test_pool"]["group_evaluations"])
put("TpPq", tps["test_pool"]["mean_pq"], ".3f")
put("TrainPq", tps["train"]["all"]["mean_pq"], ".3f")
put("TrainP", tps["train"]["all"]["mean_p"], ".3f")
put("TrainAbove", 100 * tps["train"]["all"]["share_above_half"], ".0f")
put("RecentP", tps["train"]["recent_2025_07"]["mean_p"], ".3f")
put("RecentAbove", 100 * tps["train"]["recent_2025_07"]["share_above_half"], ".0f")
put("RecentGroups", tps["train"]["recent_2025_07"]["groups"])
put("RecentBench", tps["train"]["recent_2025_07"]["benchmarks"])
put("TpShareOwn", ", ".join(f"{x:.3f}" for x in OWN_SHARE))
put("TpLaterGroups", tps["test_pool_later"]["group_evaluations"])
put("TpLaterShare", ", ".join(f"{x:.3f}" for x in tps["test_pool_later"]["bin_share"]))
put("TpLaterPq", tps["test_pool_later"]["mean_pq"], ".3f")
put("TpPooledGroups", tps["test_pool_pooled"]["group_evaluations"])
put("TpShareSim", ", ".join(f"{x:.3f}" for x in SIM_SHARE))
put("TpFarShare", 100 * OWN_SHARE[3:].sum(), ".0f")
put("TpExtremeN", int(round(OWN_SHARE[4] * tps["test_pool"]["group_evaluations"])))
dpj = json.loads((ROOT / "colab_jobs/results/date_prior.json").read_text())
put("DateA", dpj["a"], ".3f")
put("DateB", dpj["b"], ".3f")
put("DateAzero", dpj["a0"], ".3f")

LADDER = {}
for tag in ("recent_dev", "recent_fresh", "all_dev", "all_fresh", "recent_conf", "all_conf"):
    f = HERE / f"data/ladder_{tag}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    bins = np.minimum((np.abs(z["p"] - 0.5) / 0.1).astype(int), 4)
    train_share = np.bincount(bins, minlength=5) / len(bins)
    weights = {"un": np.ones(len(bins))}
    for lab, share in (("tw", SIM_SHARE), ("own", OWN_SHARE)):
        w = share[bins] / np.maximum(train_share[bins], 1e-9)
        weights[lab] = w / w.mean()
    names = [n for n in json.loads(str(z["variants"]))]
    res = {"draws": int(len(set(z["draw_id"]))), "groups": int(len(bins)), "bench": int(z["n_benchmarks"])}
    base = z["beta"] @ W6
    for lab, w in weights.items():
        res[f"beta_{lab}"] = ratio_se(w, base, z["draw_id"])[0]
        res[f"beta_pb_{lab}"] = [ratio_se(w, z["beta"][:, j], z["draw_id"])[0] for j in range(6)]
        for n in names:
            if n == "beta":
                continue
            m, se = ratio_se(w, z[n] @ W6 - base, z["draw_id"])
            res[f"{n}_{lab}"], res[f"{n}_{lab}_se"] = m, se
            res[f"{n}_pbdiff_{lab}"] = [ratio_se(w, z[n][:, j] - z["beta"][:, j], z["draw_id"]) for j in range(6)]
    LADDER[tag] = res
(HERE / "data/ladder_summary.json").write_text(json.dumps(LADDER, indent=1) + "\n")

WORD = {"recent_dev": "RD", "recent_fresh": "RF", "all_dev": "AD", "all_fresh": "AF", "recent_conf": "RC", "all_conf": "AC"}
for tag, res in LADDER.items():
    t = WORD[tag]
    put(f"Lad{t}Draws", res["draws"])
    put(f"Lad{t}Groups", thousands(res["groups"]))
    put(f"Lad{t}Bench", res["bench"])
    for lab, L in (("un", "U"), ("tw", "T"), ("own", "O")):
        put(f"Lad{t}{L}Beta", res[f"beta_{lab}"], ".4f")
        for n, N in (("L1c", "Lonec"), ("L2c", "Ltwoc"), ("wide_level", "Wlevel"), ("wide_exact", "Wexact"),
                     ("wide_kernel", "Wkernel"), ("L3", "Lthree"), ("L4", "Lfour"), ("L5", "Lfive"),
                     ("L5_no_text", "Lfivent"), ("L6", "Lsix")):
            put(f"Lad{t}{L}{N}", res[f"{n}_{lab}"], "+.4f")
            put(f"Lad{t}{L}{N}Se", res[f"{n}_{lab}_se"], ".4f")
        put(f"Lad{t}{L}LfiveRel", 100 * res[f"L5_{lab}"] / res[f"beta_{lab}"], "+.1f")

# Paired comparisons between candidates on the same draws.
for tag, a, b, name in (("recent_conf", "L5", "L6", "PairRFLfiveLsix"), ("all_conf", "L5", "L6", "PairAFLfiveLsix"),
                        ("recent_conf", "L5", "L5_no_text", "PairRFText"), ("all_conf", "L5", "L5_no_text", "PairAFText"),
                        ("recent_conf", "L5", "L3", "PairRFDateName"), ("all_conf", "L5", "L3", "PairAFDateName"),
                        ("all_conf", "L5", "L2c", "PairAFLfiveLtwoc"), ("recent_conf", "L5", "L2c", "PairRFLfiveLtwoc")):
    z = np.load(HERE / f"data/ladder_{tag}.npz")
    W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    bins = np.minimum((np.abs(z["p"] - 0.5) / 0.1).astype(int), 4)
    ts = np.bincount(bins, minlength=5) / len(bins)
    for lab, w in (("T", OWN_SHARE[bins] / ts[bins]), ("U", np.ones(len(bins)))):
        m, se = ratio_se(w / w.mean(), z[a] @ W6 - z[b] @ W6, z["draw_id"])
        put(f"{name}{lab}", m, "+.4f")
        put(f"{name}{lab}Se", se, ".4f")

# Numbers reported by runs that the report scripts do not recompute (data/reported_numbers.json).
rep_ = json.loads((HERE / "data/reported_numbers.json").read_text())
put("JudgeSimWeight", rep_["judge_sim_best_weight"]["value"], ".1f")
put("JudgeSimGain", rep_["judge_sim_gain"]["value"], ".5f")
put("ColabUnits", rep_["colab_units_own"]["value"], ".1f")
put("LfiveSeconds", rep_["l5_runtime_seconds_per_draw_m4"]["value"])
so_budgets = [0, 1, 3, 7, 15, 31]
# Encoder cost with the platform's worker restart at every budget (scripts/time_encoder.py on Colab CPU).
enc_ = json.loads((HERE / "data/encode_timing.json").read_text())
draw_s = list(enc_["draw_cold_encode_s"].values())
groups_per_draw = statistics.fmean(enc_["groups"].values())
load_s = statistics.median(enc_["load_s"][1:])
per_group_s = (statistics.fmean(draw_s) + 2 * len(so_budgets) * load_s) / groups_per_draw
put("EncTextSec", enc_["per_text_mean"], ".2f")
put("EncDrawMin", statistics.fmean(draw_s) / 60, ".0f")
put("EncDrawMinLo", min(draw_s) / 60, ".0f")
put("EncDrawMinHi", max(draw_s) / 60, ".0f")
put("EncLoadSec", load_s, ".1f")
put("EncGroupsPerDraw", groups_per_draw, ".0f")
put("EncMaxGroups", int(8 * 3600 / per_group_s // 10 * 10))
put("EncPerBudgetLo", min(r["encoded"] for r in enc_["encoded_per_budget"]))
put("EncPerBudgetHi", max(r["encoded"] for r in enc_["encoded_per_budget"]))
put("EncDraws", len(draw_s))
put("GridDraws", rep_["grid_tuning_draws"]["value"])
lad_rc = LADDER["recent_conf"]
put("LadRCOLfiveSeRel", 100 * lad_rc["L5_own_se"] / lad_rc["beta_own"], ".2f")

# Official Beta per budget, mean of the three runs (for the calibration figure).
official_beta_pb, later_beta_pb = [], []
for p_ in sorted(glob.glob(str(ROOT / "private/feedback/*_scoring_stdout.txt"))):
    sid = int(Path(p_).name.split("_")[0])
    if archive_of.get(sid) != "adaptive_baseline.zip":
        continue
    parts = re.split(r"Label budget:\s*(\d+)", Path(p_).read_text())
    row = r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)"
    (official_beta_pb if sid in CALIB_BETA_IDS else later_beta_pb).append(
        [statistics.fmean(float(x[3]) for x in re.findall(row, parts[i + 1])) for i in range(1, len(parts), 2)])
official_beta_pb = np.array(official_beta_pb)
(HERE / "data/official_beta_per_budget.json").write_text(json.dumps(official_beta_pb.tolist()) + "\n")
(HERE / "data/official_beta_later_per_budget.json").write_text(json.dumps(later_beta_pb) + "\n")

# Extra statistics (judge, name ability, replays).
ex = json.loads((HERE / "data/extra_stats.json").read_text())
put("JudgeRated", thousands(ex["judge"]["rated"]))
put("JudgeBench", ex["judge"]["benchmarks"])
put("JudgeMedian", ex["judge"]["median_within"], "+.3f")
put("JudgeNeg", 100 * ex["judge"]["share_negative"], ".0f")
put("JudgeBetween", ex["judge"]["between"], "+.3f")
put("NameCount", ex["name_ability"]["names_runtime"])
put("NameBench", ex["name_ability"]["benchmarks"])
put("NamePos", 100 * ex["name_ability"]["share_positive"], ".0f")
put("NameMedian", ex["name_ability"]["median"], ".2f")
for name, N in (("v2", "Two"), ("v3", "Three")):
    r_ = ex["replay"][name]
    put(f"Rep{N}Draws", r_["draws"])
    put(f"Rep{N}Groups", r_["groups"])
    for lab, L_ in (("testw", "T"), ("unweighted", "U")):
        put(f"Rep{N}{L_}Run", r_[f"{lab}_run"], ".4f")
        put(f"Rep{N}{L_}Lfive", r_[f"{lab}_L5"], ".4f")
        put(f"Rep{N}{L_}Beta", r_[f"{lab}_beta"], ".4f")
        put(f"Rep{N}{L_}Diff", r_[f"{lab}_L5_minus_run"], "+.4f")
        put(f"Rep{N}{L_}DiffSe", r_[f"{lab}_L5_minus_run_se"], ".4f")
        put(f"Rep{N}{L_}VsBeta", r_[f"{lab}_run_minus_beta"], "+.4f")
        put(f"Rep{N}{L_}VsBetaSe", r_[f"{lab}_run_minus_beta_se"], ".4f")
put("RepCheckDraws", ex["replay"]["L5_check"]["draws"])
put("RepCheckGroups", ex["replay"]["L5_check"]["groups"])
put("RepCheckDiff", ex["replay"]["L5_check"]["mean_abs_diff"], ".4f")
put("RepCheckRuntime", ex["replay"]["L5_check"]["runtime_mean"], ".4f")
put("RepCheckSim", ex["replay"]["L5_check"]["sim_mean"], ".4f")
put("ArchiveFinalMB", (ROOT / "dist/v5/L5_wide_date_name.zip").stat().st_size / 1e6, ".1f")
put("OffBetaPbOne", official_beta_pb.mean(0)[1], ".4f")
put("OffBetaPbLast", official_beta_pb.mean(0)[-1], ".4f")

# Stronger Beta baselines and the no-sharing ablation (scripts/sim_baselines.py, same draws as the ladder).
W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
BOOT = np.random.default_rng(7).multinomial(tps["test_pool"]["group_evaluations"], OWN_SHARE, size=20000)
BOOT = BOOT / BOOT.sum(1, keepdims=True)
for tag, t in (("recent_conf", "RC"), ("all_conf", "AC")):
    zb = np.load(HERE / f"data/baselines_{tag}.npz")
    zl = np.load(HERE / f"data/ladder_{tag}.npz")
    if not (np.allclose(zb["beta"], zl["beta"]) and np.allclose(zb["L5"], zl["L5"])):
        raise SystemExit(f"baseline draws differ from the ladder draws ({tag})")
    bins = np.minimum((np.abs(zb["p"] - 0.5) / 0.1).astype(int), 4)
    ts = np.bincount(bins, minlength=5) / len(bins)
    alc = {n: zb[n] @ W6 for n in json.loads(str(zb["variants"]))}
    alc.update({n: zl[n] @ W6 for n in ("L2c", "L6", "wide_level", "wide_exact")})
    for lab, L in (("un", "U"), ("own", "O")):
        w = np.ones(len(bins)) if lab == "un" else OWN_SHARE[bins] / ts[bins]
        w = w / w.mean()
        for n, N in (("beta_a0.25", "Bqtr"), ("beta_a0.5", "Bhalf"), ("beta_a2", "Btwo"),
                     ("wide_level_own", "Wown"), ("L5_own", "Lfiveown")):
            m, se = ratio_se(w, alc[n] - alc["beta"], zb["draw_id"])
            put(f"Lad{t}{L}{N}", m, "+.4f")
            put(f"Lad{t}{L}{N}Se", se, ".4f")
        for a, b, N in (("L5", "L5_own", "Share"), ("wide_level", "wide_level_own", "LevelShare"),
                        ("L5", "beta_a0.5", "Bhalf"), ("wide_exact", "wide_level", "ExactStep")):
            m, se = ratio_se(w, alc[a] - alc[b], zb["draw_id"])
            put(f"Pair{t}{N}{L}", m, "+.4f")
            put(f"Pair{t}{N}{L}Se", se, ".4f")
    # The official baseline is Beta(2,2): its simulated ALC, for a like-for-like comparison with official runs.
    for lab, L in (("un", "U"), ("own", "O")):
        w = np.ones(len(bins)) if lab == "un" else OWN_SHARE[bins] / ts[bins]
        w = w / w.mean()
        put(f"SimBtwo{t}{L}", float((w * alc["beta_a2"]).sum() / w.sum()), ".4f")
    # Check on Beta runs made after the model choice: their shares alone, and pooled with the first three.
    for key, K in (("test_pool_later", "Later"), ("test_pool_pooled", "Pooled")):
        sh = np.array(tps[key]["bin_share"])
        w = sh[bins] / ts[bins]
        w = w / w.mean()
        for a_, b_, N in (("L5", "beta", "LfiveBeta"), ("L5", "L2c", "LfiveLtwoc"), ("L5", "beta_a2", "LfiveBtwo")):
            m, se = ratio_se(w, alc[a_] - alc[b_], zb["draw_id"])
            put(f"Sh{K}{t}{N}", m, "+.4f")
            put(f"Sh{K}{t}{N}Se", se, ".4f")
            put(f"Sh{K}{t}{N}Abs", abs(m), ".4f")
        put(f"Sh{K}{t}SimBtwo", float((w * alc["beta_a2"]).sum() / w.sum()), ".4f")
    # Sampling error of the 26-group share estimate: resample the group evaluations' bins.
    for a, b, N in (("L5", "beta", "LfiveBeta"), ("L5", "L2c", "LfiveLtwoc"), ("L5", "L6", "LfiveLsix")):
        d = alc[a] - alc[b]
        per_bin = np.array([d[bins == k].mean() for k in range(5)])
        est = BOOT @ per_bin
        lo, hi = np.quantile(est, [0.05, 0.95])
        put(f"Boot{t}{N}Lo", lo, "+.4f")
        put(f"Boot{t}{N}Hi", hi, "+.4f")
        put(f"Boot{t}{N}P", 100 * (est < 0).mean(), ".0f")
        put(f"Bin{t}{N}Zero", per_bin[0], "+.4f")
        put(f"Bin{t}{N}Four", per_bin[4], "+.4f")
put("BootReps", "20,000")

lines =["% Generated by scripts/make_numbers.py. Do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in OUT.items()]
(HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
