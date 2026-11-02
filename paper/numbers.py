#!/usr/bin/env python3
"""Recomputes every number the paper prints from the files in results/ and research/.

    python3 paper/numbers.py          # print each value with its source, then check; exit 1 on any mismatch
    python3 paper/numbers.py --write  # same, and rewrite paper/numbers.tex

It fails on: a value that disagrees with what README.md states, a summary JSON that disagrees with its raw run file,
a stale paper/numbers.tex, a macro the paper uses and this script does not define, a measured-looking number typed
directly into the paper's text, and a word the paper's rules forbid. Standard library only.
"""
import os
import sys

# This file's name shadows the standard library's `numbers`, which `statistics` and `decimal` import.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import json
import math
import random
import re
import statistics
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
TEX = PAPER / "induction-tab.tex"
MACROS = PAPER / "numbers.tex"

WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 8: "Eight", 12: "Twelve"}

registry = {}
mismatches = []


def fmt(x, places):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def sci(x, places=1):
    mantissa, exponent = f"{x:.{places}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def lr_text(lr):
    return {0.003: "3e-3", 0.001: "1e-3"}[lr]


def put(name, value, source, expect=None):
    """Register one macro. `expect` is the value README.md or the brief states; a disagreement fails the run."""
    assert name.isalpha(), f"LaTeX macro names are letters only: {name}"
    assert name not in registry, f"duplicate macro {name}"
    text = str(value)
    registry[name] = (text, source)
    if expect is not None and text != str(expect):
        mismatches.append(f"{name}: computed {text}, stated {expect} ({source})")


def same(label, a, b, tolerance=1e-9):
    equal = a == b if not isinstance(a, float) or not isinstance(b, float) else abs(a - b) <= tolerance
    if not equal:
        mismatches.append(f"{label}: {a} against {b}")


def js(name):
    return json.loads((RESULTS / name).read_text())


def rows(name):
    return [json.loads(line) for line in (RESULTS / f"{name}.jsonl").read_text().splitlines() if line.strip()]


def km_median(times, fired):
    """Kaplan-Meier median, the same rule as tools/analyze_study.py. Infinite when survival never reaches one half."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    n, s = len(times), 1.0
    for rank, i in enumerate(order):
        if fired[i]:
            s *= 1 - 1 / (n - rank)
            if round(s, 5) <= 0.5:
                return times[i]
    return math.inf


def te(runs):
    return [r["t_event"] if r["t_event"] is not None else r["budget"] for r in runs], [r["t_event"] is not None for r in runs]


def median_of(runs):
    return km_median(*te(runs))


def boot_ratio(a, b, resamples=4000, seed=0):
    """Bootstrap interval over seeds of the ratio of Kaplan-Meier medians, as in tools/analyze_third.py."""
    rng = random.Random(seed)
    ta, ea = te(a)
    tb, eb = te(b)
    out = []
    for _ in range(resamples):
        ia = [rng.randrange(len(ta)) for _ in ta]
        ib = [rng.randrange(len(tb)) for _ in tb]
        out.append(km_median([ta[i] for i in ia], [ea[i] for i in ia]) / km_median([tb[i] for i in ib], [eb[i] for i in ib]))
    out.sort()
    return out[int(0.025 * resamples)], out[int(0.975 * resamples)]


def curriculum_of(cfg):
    return ("uniform" if cfg["uniform"] else "bigram") + (" tiled" if cfg.get("tile") else " clone")


def config_numbers():
    tab = json.loads((ROOT / "configs/tab.json").read_text())
    defaults = json.loads((ROOT / "configs/defaults.json").read_text())
    study = rows("study-lr3e-3")[0]["cfg"]
    a = rows("config-a")[0]["cfg"]
    src = "configs/tab.json"
    for name, key in (("nTabV", "V"), ("nTabT", "T"), ("nTabD", "d"), ("nTabH", "H"), ("nTabB", "B"), ("nTabMlp", "mlp")):
        put(name, tab[key], src)
    put("nTabLr", lr_text(tab["lr"]), src)
    put("nTabRange", f"[{tab['rMin']}, {tab['rMax']})", src)
    src = "results/study-lr3e-3.jsonl"
    for name, key in (("nStudyT", "T"), ("nStudyD", "d"), ("nStudyH", "H"), ("nStudyB", "B"), ("nStudyV", "V")):
        put(name, study[key], src)
    put("nRangeShort", f"[{study['rMin']}, {study['rMax']})", src)
    put("nStudyParams", rows("study-lr3e-3")[0]["params"], src)
    put("nLrHi", lr_text(0.003), "tools/gen_study.py")
    put("nLrLo", lr_text(0.001), "tools/gen_study.py")
    src = "configs/defaults.json"
    put("nBetaOne", defaults["b1"], src)
    put("nBetaTwo", defaults["b2"], src)
    put("nEps", sci(defaults["eps"], 0), src)
    put("nWeightDecay", defaults["wd"], src)
    put("nWarmup", defaults["warmup"], src)
    src = "results/config-a.jsonl"
    for name, key in (("nCfgAV", "V"), ("nCfgAT", "T"), ("nCfgAD", "d"), ("nCfgAH", "H"), ("nCfgAB", "B"), ("nCfgAProbeT", "probeT"), ("nCfgAProbeB", "probeB")):
        put(name, a[key], src)
    put("nCfgALr", lr_text(a["lr"]), src)
    put("nCfgAPRep", a["pRep"], src)


def timing_numbers():
    text = (RESULTS / "bench-m4pro.txt").read_text()
    bench = json.loads(text.strip().splitlines()[-1])
    src = "results/bench-m4pro.txt"
    runs = bench["out"]
    seconds = [r["wallEvent"] / 1000 for r in runs]
    per_step = [float(x) for x in re.findall(r"([\d.]+) ms/step", text)]
    join = lambda xs: ", ".join(xs[:-1]) + ", and " + xs[-1]
    put("nBenchEvents", join([str(r["tEvent"]) for r in runs]), src, "500, 475, 600, 425, and 475")
    put("nBenchSeconds", join([fmt(s, 1) for s in seconds]), src, "5.2, 4.9, 6.2, 4.4, and 4.9")
    put("nBenchMedianSeconds", fmt(statistics.median(seconds), 1), src, "4.9")
    put("nBenchMsPerStep", fmt(statistics.median(per_step), 1), src, "10.3")
    put("nBrowserVersion", bench["browser"], src)
    put("nBenchSeeds", len(runs), src)
    put("nTabParams", js("breakdown-m4pro.json")["params"], "results/breakdown-m4pro.json", 435456)

    watched = js("watched-m4pro.json")
    src = "results/watched-m4pro.json"
    put("nWatchedMedian", fmt(watched["medianClockSeconds"], 1), src)
    put("nWatchedLeast", fmt(watched["medianLeastLoadedClockSeconds"], 1), src)
    put("nWatchedRounds", watched["rounds"], src)
    put("nWatchedLoad", fmt(watched["otherLoadFactor"], 1), src)
    clocks = [r["clockSeconds"] for r in watched["runs"]]
    put("nWatchedLow", fmt(min(clocks), 1), src)
    put("nWatchedHigh", fmt(max(clocks), 1), src)
    quoted = [f"{fmt(watched[k], 1)} s" for k in ("medianClockSeconds", "medianLeastLoadedClockSeconds")]
    same("README and the page quote the watched clock", all(q in (ROOT / f).read_text() for q in quoted for f in ("README.md", "index.html")), True)
    same("watched and benchmark reach the event at the same steps", {r["seed"]: r["tEvent"] for r in watched["runs"]}, {r["seed"]: r["tEvent"] for r in runs})

    text = (RESULTS / "breakdown-m4pro.txt").read_text()
    src = "results/breakdown-m4pro.txt"
    step = float(re.search(r"step ([\d.]+) ms", text).group(1))
    put("nStepMs", fmt(step, 2), src, "10.04")
    put("nDispatches", re.search(r"(\d+) dispatches", text).group(1), src)
    put("nOverheadUs", re.search(r"overhead ([\d.]+) us", text).group(1), src)
    put("nOverheadMs", re.search(r"([\d.]+) ms per step", text).group(1), src)
    entries = re.findall(r"^\s+(\w+)\s+(\d+)\s+([\d.]+) ms$", text, re.M)
    times = {name: float(ms) for name, _, ms in entries}
    counts = {name: int(count) for name, count, _ in entries}
    for name, key in (("nPhaseFwd", "fwd"), ("nPhaseLoss", "loss"), ("nPhaseBwd", "bwd"), ("nPhaseOpt", "opt"), ("nGemmMs", "GEMM"), ("nEmbedBwdMs", "EMBED_BWD"), ("nTokenIndexMs", "TOKEN_INDEX"), ("nLnFwdMs", "LN_FWD")):
        put(name, fmt(times[key], 2), src)
    put("nGemmDispatches", counts["GEMM"], src)
    kernels = [k for k in times if k.isupper()]
    put("nKernelSum", fmt(sum(times[k] for k in kernels), 2), src)
    put("nGemmShare", fmt(100 * times["GEMM"] / step, 0), src)
    same("GEMM is the README's 8.44", times["GEMM"], 8.44)

    native = js("native-speed.json")
    src = "results/native-speed.json"
    put("nMpsMs", fmt(native["mpsMsPerStep"], 2), src, "5.50")
    put("nCpuMs", fmt(native["cpuMsPerStep"], 2), src, "13.93")
    put("nCpuThreads", native["threads"], src, 8)
    put("nTorchVersion", native["torch"], src)
    put("nMpsFaster", fmt(float(registry["nBenchMsPerStep"][0]) / native["mpsMsPerStep"], 1), "results/bench-m4pro.txt over results/native-speed.json", "1.9")
    put("nMpsFasterReplay", fmt(step / native["mpsMsPerStep"], 2), "results/breakdown-m4pro.txt over results/native-speed.json")


def parity_numbers():
    metal = js("parity-apple-metal-3.json")
    swift = js("parity-google-swiftshader.json")
    src = "results/parity-apple-metal-3.json"
    cfgs = metal["configs"].values()
    worst_grad = max(max(c["summary"]["grad0"]["worstRelL2"], c["summary"]["gradN"]["worstRelL2"]) for c in cfgs)
    put("nParityConfigs", len(metal["configs"]), src, 5)
    put("nParitySteps", metal["steps"], src, 20)
    put("nParityGrad", sci(worst_grad), src, "1.1e-6")
    exact = f"{worst_grad:.2e}".replace("e-0", "e-")
    same(f"README and the page quote the stored parity figure {exact}", all(exact in (ROOT / f).read_text() for f in ("README.md", "index.html")), True)
    put("nParityParam", sci(max(c["summary"]["pN"]["worstRelL2"] for c in cfgs)), src)
    put("nParityLoss", sci(max(c["lossMaxAbsErr"] for c in cfgs)), src)
    src = "results/parity-google-swiftshader.json"
    put("nSwiftConfigs", len(swift["configs"]), src, 2)
    put("nSwiftGrad", sci(max(max(c["summary"]["grad0"]["worstRelL2"], c["summary"]["gradN"]["worstRelL2"]) for c in swift["configs"].values())), src)


def whole_run(prefix, summary_file, run_files, native_runs, browser_runs, stated_worst, stated_under=None):
    summary = js(summary_file)
    src = f"results/{summary_file}"
    native = {r["seed"]: r for r in native_runs}
    browser = {r["seed"]: r for r in browser_runs}
    seeds = sorted(set(native) & set(browser))
    worst, under, evals = 0.0, 0, 0
    for s in seeds:
        for a, b in zip(native[s]["curve"], browser[s]["curve"]):
            same("eval steps align", a["step"], b["step"])
            d = abs(a["copyAdv"] - b["copyAdv"])
            worst, under, evals = max(worst, d), under + (d < 1e-5), evals + 1
    same(f"{summary_file} evals", evals, summary["evals"])
    same(f"{summary_file} worst copy advantage", worst, summary["worstCopyAdvDifference"])
    same(f"{summary_file} same event", sum(native[s]["t_event"] == browser[s]["t_event"] for s in seeds), summary["sameEventStep"])
    same(f"{summary_file} same anchor", sum(native[s]["t_pv"] == browser[s]["t_pv"] for s in seeds), summary["sameAnchorStep"])
    stops = [c for s in seeds for c in [browser[s]["curve"][-1]] if c["step"] != browser[s]["t_event"]]
    same(f"{summary_file} runs stop at the event", len(stops), 0)
    put(f"n{prefix}Seeds", summary["seeds"], src, 64)
    put(f"n{prefix}SameEvent", summary["sameEventStep"], src, 64)
    put(f"n{prefix}SameAnchor", summary["sameAnchorStep"], src, 64)
    put(f"n{prefix}Evals", summary["evals"], src)
    put(f"n{prefix}WorstCopy", sci(summary["worstCopyAdvDifference"]), src, stated_worst)
    put(f"n{prefix}WorstLoss", sci(summary["worstLossDifference"]), src)
    put(f"n{prefix}Under", under, f"results/{run_files}", stated_under)
    put(f"n{prefix}MedianEvent", int(summary["medianBrowser"]), src)


def trainer_numbers():
    tab_browser = rows("tab-browser")
    whole_run("WholeTab", "tab-native-vs-browser.json", "tab-native.jsonl and tab-browser.jsonl", rows("tab-native"), tab_browser, "6.5e-5", 1328)
    main_cfg = rows("browser-2-2")[0]["cfg"]
    whole_run("WholeStudy", "native-vs-browser.json", "study-lr3e-3.jsonl and browser-2-2.jsonl", [r for r in rows("study-lr3e-3") if r["cfg"] == main_cfg], rows("browser-2-2"), "3.2e-6")
    put("nUnderTol", sci(1e-5, 0), "the threshold of the count in this table")
    put("nEvalEvery", main_cfg.get("evalEvery", 25), "results/browser-2-2.jsonl (default 25)", 25)

    third = js("third-summary.json")
    src = "results/third-summary.json"
    tab = third["tab"]
    at = tab["atEvent"]
    put("nTabEvents", tab["events"], src, 64)
    put("nTabRuns", tab["n"], src, 64)
    put("nTabMedian", tab["medianStep"], src, 475)
    put("nTabEventMin", tab["eventRange"][0], src, 400)
    put("nTabEventMax", tab["eventRange"][1], src, 700)
    put("nTabStalled", len(tab["stalled"]), src, 0)
    put("nTabFirstPass", fmt(at["firstPassMedian"], 2), src, "5.23")
    put("nTabSecondPass", fmt(at["secondPassMedian"], 2), src, "3.01")
    put("nTabAdvantage", fmt(at["advantageMedian"], 2), src, "2.23")
    put("nTabShare", fmt(100 * at["shareFromFirstPassMedian"], 0), src, "30")
    put("nChance", fmt(at["chance"], 2), src, "4.56")
    same("chance is ln V", at["chance"], math.log(96))
    put("nTabBudget", tab_browser[0]["budget"], "results/tab-browser.jsonl", 3000)
    fractions = [r["t_pv"] / r["t_event"] for r in tab_browser]
    put("nTabGap", fmt(statistics.median(fractions), 3), "results/tab-browser.jsonl")

    saved = rows("saved-run")[0]
    src = "results/saved-run.jsonl"
    end = saved["curve"][-1]
    put("nSavedStep", end["step"], src, 3000)
    put("nSavedAdv", fmt(end["copyAdv"], 2), src, "4.59")
    put("nSavedFirst", fmt(end["firstPass"], 2), src, "4.73")
    put("nSavedSecond", fmt(end["secondPass"], 2), src, "0.14")

    negatives = third["negatives"]
    put("nNegTabLayer", fmt(negatives["tab-neg-L1"]["maxCopyAdv"], 3), src, "0.913")
    put("nNegTabNoRep", fmt(negatives["tab-neg-norep"]["maxCopyAdv"], 3), src, "0.029")
    same("tab negatives fire nowhere", negatives["tab-neg-L1"]["events"] + negatives["tab-neg-norep"]["events"], 0)
    one_layer = rows("tab-neg-L1")
    peak = max((c for r in one_layer for c in r["curve"]), key=lambda c: c["copyAdv"])
    put("nNegLayerFirst", fmt(peak["firstPass"], 2), "results/tab-neg-L1.jsonl")
    put("nNegLayerSecond", fmt(peak["secondPass"], 2), "results/tab-neg-L1.jsonl")
    put("nNegBudget", one_layer[0]["budget"], "results/tab-neg-L1.jsonl", 6000)
    put("nNegSeeds", len(one_layer), "results/tab-neg-L1.jsonl", 8)
    study_negatives = rows("study-negatives")
    src = "results/study-negatives.jsonl"
    layer = [r for r in study_negatives if r["cfg"]["L"] == 1]
    norep = [r for r in study_negatives if r["cfg"].get("pRep") == 0]
    same("study negatives fire nowhere", sum(r["t_event"] is not None for r in study_negatives), 0)
    put("nNegStudyLayer", fmt(max(c["copyAdv"] for r in layer for c in r["curve"]), 2), src, "1.20")
    put("nNegStudyNoRep", fmt(max(c["copyAdv"] for r in norep for c in r["curve"]), 3), src, "0.067")


def second_pass_numbers():
    src = "every results/*.jsonl that logs both passes"
    chance = math.log(96)
    logged = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        if path.stem == "fine-2-2-extra":
            continue
        for r in rows(path.stem):
            at_event = next((c for c in r["curve"] if c["step"] == r["t_event"]), None)
            if at_event and "secondPass" in at_event:
                logged.append((path.stem, at_event))
    above = [(name, c) for name, c in logged if c["secondPass"] > chance]
    same("second pass above chance only on Howe's language", {name for name, _ in above}, {"crossed-rule-bigram", "crossed-range-bigram"})
    second = [c["secondPass"] for _, c in above]
    first = [c["firstPass"] for _, c in above]
    prefix = [max(c["prefix"]) for _, c in above]
    put("nLoggedEvents", len(logged), src)
    put("nLangEvents", len(above), src, 3)
    put("nLangSecondLow", fmt(min(second), 2), src, "4.92")
    put("nLangSecondHigh", fmt(max(second), 2), src, "5.34")
    put("nLangFirstLow", fmt(min(first), 2), src)
    put("nLangFirstHigh", fmt(max(first), 2), src)
    put("nLangPrefixLow", fmt(min(prefix), 2), src)
    put("nLangPrefixHigh", fmt(max(prefix), 2), src)
    put("nLangUnlogged", sum(r["t_event"] is not None for r in rows("study-curriculum")), "results/study-curriculum.jsonl", 28)
    same("that file does not log both passes", any("secondPass" in c for r in rows("study-curriculum") for c in r["curve"]), False)


def correction_numbers():
    c = js("contrast-correction.json")
    src = "results/contrast-correction.json"
    study = js("study-summary.json")["contrasts"]
    third = js("third-summary.json")
    counted = sum(x.get("ci") is not None for x in study) + sum(x.get("kmCi") is not None for x in study) + sum(v.get("ci") is not None for part in ("curriculum", "depth") for v in third[part].values()) + 1
    same("intervals read", c["intervalsReadTotal"], counted)
    stored = {(tuple(x["a"]), tuple(x["b"]), x["lr"]): x for x in study if isinstance(x["a"], list)}
    for t in c["tying"]:
        k = stored[(tuple(t["a"]), tuple(t["b"]), t["lr"])]
        same(f"correction tool's fit for {t['a']} at {t['lr']}", round(t["fit"], 3), round(k["timeRatio"], 3))
        same(f"correction tool's median ratio for {t['a']} at {t['lr']}", round(t["km"], 3), round(k["kmRatio"], 3))
    put("nIntervalsRead", c["intervalsReadTotal"], src, 55)
    put("nTyingIntervals", c["tyingIntervals"], src, 20)
    put("nAdjustedLevel", f"0.05/{c['intervalsReadTotal']}", src)
    put("nAdjustedResamples", c["resamples"], src)
    put("nAdjustedBound", fmt(c["lowestAdjustedBound"], 2), src)
    same("tying stays above 1 under the correction", c["lowestAdjustedBound"] > 1, True)
    lat = c["lattice"]
    put("nLatticeContinuousLow", fmt(100 * lat["coversContinuousRange"][0], 1), src)
    put("nLatticeContinuousHigh", fmt(100 * lat["coversContinuousRange"][1], 1), src)
    put("nLatticeGridLow", fmt(100 * lat["coversGridRange"][0], 1), src)
    put("nLatticeGridHigh", fmt(100 * lat["coversGridRange"][1], 0), src)
    put("nLatticeReps", lat["reps"], src)
    put("nLatticeN", lat["n"], src, 64)
    put("nLatticeSd", fmt(lat["logSd"], 3), src)


def howe_numbers():
    spec = (ROOT / "research/UPSTREAM-SPEC.md").read_text()
    table = [tuple(map(int, row)) for row in re.findall(r"^\| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", spec, re.M)]
    events = [row[1] for row in table]
    fractions = [row[2] / row[1] for row in table]
    src = "research/UPSTREAM-SPEC.md, section 5"
    put("nHoweSeeds", len(table), src, 30)
    put("nHoweEventMedian", int(statistics.median(events)), src, 6300)
    put("nHoweEventMin", min(events), src, 5650)
    put("nHoweEventMax", max(events), src, 7400)
    put("nHoweFraction", fmt(statistics.median(fractions), 3), src, "0.844")
    study_py = (ROOT / "tools/analyze_study.py").read_text()
    low, high = map(float, re.search(r"ENV = \(([\d.]+), ([\d.]+)\)", study_py).groups())
    put("nEnvLow", fmt(low, 3), "tools/analyze_study.py", "0.735")
    put("nEnvHigh", fmt(high, 3), "tools/analyze_study.py", "0.901")
    novelty = (ROOT / "research/NOVELTY.md").read_text()
    law = re.search(r"post-hoc across (\d+) valid-anchor runs: anchor at ([\d.]+) of time to emergence.*?t_event in \[([\d.]+), ([\d.]+)\]", novelty)
    src = "research/NOVELTY.md, quoting arXiv 2609.19000"
    put("nLawRuns", law.group(1), src, 80)
    put("nLawMedian", law.group(2), src, "0.843")
    put("nEnvRatioLow", law.group(3), src, "1.11")
    put("nEnvRatioHigh", law.group(4), src, "1.36")
    same("envelope low is 1/1.36", round(1 / float(law.group(4)), 3), low)
    same("envelope high is 1/1.11", round(1 / float(law.group(3)), 3), high)

    harmonic = lambda t: sum(1 / k for k in range(1, t + 1))
    base = {t: (harmonic(t) - 1) / (t - 1) for t in (64, 128)}
    anchor = js("anchor-summary.json")
    for row in anchor:
        same(f"baseline at T={row['T']}", row["baseline"], base[row["T"]])
    src = "(H_T - 1)/(T - 1), checked against results/anchor-summary.json"
    put("nAnchorThreshold", "0.10", "tools/analyze_third.py")
    put("nBaseSixtyFour", fmt(base[64], 4), src, "0.0594")
    put("nBaseOneTwentyEight", fmt(base[128], 4), src, "0.0349")
    put("nMultHigh", fmt(0.10 / base[128], 2), src, "2.86")
    put("nMultLow", fmt(0.10 / base[64], 2), src, "1.68")
    baseline = re.search(r'precursor is already at ([\d.]+) \(.(\d+)', (ROOT / "research/NOVELTY-2.md").read_text())
    put("nHoweBaselineMult", baseline.group(2), "research/NOVELTY-2.md, quoting arXiv 2609.19000", "10")
    put("nHoweBaselineScore", baseline.group(1), "research/NOVELTY-2.md, quoting arXiv 2609.19000", "0.37")
    rho = re.search(r"paper's ([\d.]+) is one of those orders", spec)
    put("nHoweRho", rho.group(1), "research/UPSTREAM-SPEC.md, quoting arXiv 2609.19000", "0.977")


def config_a_numbers():
    src = "results/config-a.jsonl"
    runs = rows("config-a")
    same("config A is one seed", len(runs), 1)
    run = runs[0]
    last, event, anchor = False, None, None
    for c in run["curve"]:
        if anchor is None and c["prev"][0] >= 0.10:
            anchor = c["step"]
        fired = c["copyAdv"] >= 2
        if fired and last and event is None:
            event = c["step"]
        last = fired
    same("config A event from its own curve", event, run["t_event"])
    same("config A anchor from its own curve", anchor, run["t_pv"])
    put("nCfgASeed", run["seed"], src)
    put("nCfgAEvent", run["t_event"], src, 6225)
    put("nCfgAAnchor", run["t_pv"], src)
    put("nCfgAFraction", fmt(run["t_pv"] / run["t_event"], 3), src, "0.847")
    put("nCfgAMinutes", fmt(run["wall_s"] / 60, 0), src, "24")
    put("nCfgAMsPerStep", fmt(1000 * run["wall_s"] / run["steps_run"], 0), src, "232")
    allowed = re.search(r"`config-a`, seeds 1 to (\d+)", (ROOT / "research/PREREG-3.md").read_text())
    put("nCfgASeedsAllowed", allowed.group(1), "research/PREREG-3.md", "5")


def key_of(arm):
    return WORDS[arm[0]] + WORDS[arm[1]]


def study_numbers():
    summary = js("study-summary.json")
    src = "results/study-summary.json"
    files = {
        (0.003, "uniform tiled", False): ["study-lr3e-3", "study-deep", "deep-12-12", "deep-12-4"],
        (0.001, "uniform tiled", False): ["study-lr1e-3"],
    }
    tag = {"uniform tiled": "UniTiled", "bigram tiled": "BiTiled", "bigram clone": "BiClone", "uniform clone": "UniClone"}
    pool = {name: rows(name) for name in ("study-lr3e-3", "study-lr1e-3", "study-deep", "study-curriculum", "deep-12-12", "deep-12-4", "deep-8-1-long")}
    for a in summary["arms"]:
        arm, lr, cur = (a["L"], a["u"]), a["lr"], a["curriculum"]
        if arm == (8, 1):
            runs = pool["deep-8-1-long"]
        elif cur != "uniform tiled":
            runs = [r for r in pool["study-curriculum"] if curriculum_of(r["cfg"]) == cur]
        else:
            names = files[(lr, cur, False)]
            runs = [r for n in names for r in pool[n] if r["cfg"]["L"] == arm[0] and r["cfg"]["unique"] == arm[1] and r["cfg"]["lr"] == lr]
        same(f"arm {arm} {lr} {cur} n", len(runs), a["n"])
        same(f"arm {arm} {lr} {cur} events", sum(r["t_event"] is not None for r in runs), a["events"])
        if a["medianStep"] is not None:
            same(f"arm {arm} {lr} {cur} median", median_of(runs), a["medianStep"])
        fractions = [r["t_pv"] / r["t_event"] for r in runs if r["t_event"] is not None and r["t_pv"] is not None]
        if a["gap"]:
            same(f"arm {arm} {lr} {cur} gap", statistics.median(fractions), a["gap"]["median"])

        label = key_of(arm) + ("Hi" if lr == 0.003 else "Lo") if cur == "uniform tiled" else tag[cur]
        put(f"nN{label}", a["n"], src)
        put(f"nEv{label}", a["events"], src)
        put(f"nBudget{label}", a["budget"], src)
        if a["medianStep"] is not None:
            put(f"nMed{label}", a["medianStep"], src)
        if a["gap"]:
            put(f"nGap{label}", fmt(a["gap"]["median"], 3), src)
            rho = fmt(a["gap"]["rho"], 2)
            put(f"nRho{label}", f"${rho}$", src)

    by = {((a["L"], a["u"]), a["lr"], a["curriculum"]): a for a in summary["arms"]}
    shallow = [a for a in summary["arms"] if a["curriculum"] == "uniform tiled" and a["L"] <= 4]
    medians = [a["gap"]["median"] for a in shallow]
    low, high = min(medians), max(medians)
    put("nShallowArms", len(shallow), src)
    put("nGapShallowLow", fmt(low, 3), src, "0.727")
    put("nGapShallowHigh", fmt(high, 3), src, "0.848")
    outside = sorted((a["L"], a["u"], a["lr"]) for a in shallow if not 0.735 <= a["gap"]["median"] <= 0.901)
    same("arms under the envelope", outside, [(2, 2, 0.001), (2, 2, 0.003), (3, 3, 0.003)])
    put("nArmsOutside", len(outside), src, 3)
    clone = by[((2, 2), 0.003, "bigram clone")]
    put("nCloneCensored", clone["n"] - clone["events"], src, 4)

    uniform = (0.003, "uniform tiled")
    put("nMainMedian", by[((2, 2), 0.003, "uniform tiled")]["medianStep"], src, 575)
    put("nCensorThreeOne", by[((3, 1), 0.003, "uniform tiled")]["n"] - by[((3, 1), 0.003, "uniform tiled")]["events"], src, 2)
    put("nCensorFourOne", by[((4, 1), 0.003, "uniform tiled")]["n"] - by[((4, 1), 0.003, "uniform tiled")]["events"], src, 7)
    put("nAnchorMissArm", by[((3, 3), 0.001, "uniform tiled")]["anchorNoFire"], src, 8)

    stated = {(2, 1, 0.003): ("1.55", "1.52"), (3, 1, 0.003): ("2.31", "2.17"), (4, 1, 0.003): ("3.27", "2.81"), (2, 1, 0.001): ("1.55", "1.55"), (3, 1, 0.001): ("2.09", "2.06"), (4, 1, 0.001): ("4.07", "4.04"), (8, 2, 0.003): ("2.39", "2.46")}
    tying_lows = []
    for c in summary["contrasts"]:
        if not isinstance(c["a"], list):
            continue
        a, b, lr = tuple(c["a"]), tuple(c["b"]), c["lr"]
        label = key_of(a) + "Vs" + key_of(b) + ("Hi" if lr == 0.003 else "Lo")
        expect = stated.get((a[0], a[1], lr), (None, None)) if c["kind"] == "tying" else (None, None)
        put(f"nFit{label}", fmt(c["timeRatio"], 2), src, expect[0])
        put(f"nFitCi{label}", f"{fmt(c['ci'][0], 2)} to {fmt(c['ci'][1], 2)}", src)
        put(f"nKm{label}", fmt(c["kmRatio"], 2), src, expect[1])
        put(f"nKmCi{label}", f"{fmt(c['kmCi'][0], 2)} to {fmt(c['kmCi'][1], 2)}", src)
        put(f"nFlop{label}", fmt(c["flopRatio"], 2), src)
        if c["kind"] == "tying" and a != (8, 1):
            tying_lows += [c["ci"][0], c["kmCi"][0]]
    put("nLowestTyingBound", fmt(min(tying_lows), 2), src)
    same("no tying interval lies below 1", min(tying_lows) > 1, True)

    curriculum = {(c["a"], c["b"]): c for c in summary["contrasts"] if not isinstance(c["a"], list)}
    compound = curriculum[("bigram clone", "uniform tiled")]
    put("nCompound", fmt(compound["timeRatio"], 1), src, "13.1")
    put("nCompoundCi", f"{fmt(compound['ci'][0], 1)} to {fmt(compound['ci'][1], 1)}", src)
    language = curriculum[("bigram tiled", "uniform tiled")]
    put("nLanguageFit", fmt(language["timeRatio"], 2), src)
    put("nLanguageFitCi", f"{fmt(language['ci'][0], 2)} to {fmt(language['ci'][1], 2)}", src)

    earlier = [r for r in pool["study-deep"] if r["cfg"]["L"] == 8 and r["cfg"]["unique"] == 1]
    src2 = "results/study-deep.jsonl"
    put("nEightOneEarlierSeeds", len(earlier), src2, 32)
    put("nEightOneEarlierEvents", sum(r["t_event"] is not None for r in earlier), src2, 0)
    put("nEightOneEarlierBudget", earlier[0]["budget"], src2, 4000)
    untied = by[((8, 8), 0.003, "uniform tiled")]["medianStep"]
    put("nEightOneBound", fmt(earlier[0]["budget"] / untied, 2), "results/study-deep.jsonl over the (8,8) median")

    third = js("third-summary.json")["depth"]
    long_run = third["(8,1) to 16000"]
    src3 = "results/third-summary.json"
    put("nEightOneMaxPrefix", fmt(max(long_run["finalPrefix"]), 3), src3, "0.089")
    put("nEightOneMinPrefix", fmt(min(long_run["finalPrefix"]), 3), src3)
    put("nEightOneMinCopy", fmt(min(long_run["finalCopyAdv"]), 2), src3)
    put("nEightOneMaxCopy", fmt(max(long_run["finalCopyAdv"]), 2), src3)
    last_evals = [r["curve"][-1] for r in pool["deep-8-1-long"]]
    same("(8,1) runs stop at the event", [c["step"] for c in last_evals], [r["t_event"] for r in pool["deep-8-1-long"]])
    put("nEightOneSecondLow", fmt(min(c["secondPass"] for c in last_evals), 2), "results/deep-8-1-long.jsonl")
    put("nEightOneSecondHigh", fmt(max(c["secondPass"] for c in last_evals), 2), "results/deep-8-1-long.jsonl")
    put("nEightOneMinEvent", min(long_run["events"]), src3)
    put("nEightOneMaxEvent", max(long_run["events"]), src3)
    twelve = third["(12,4) against (12,12)"]
    put("nTwelveKm", fmt(twelve["ratio"], 2), src3, "1.74")
    put("nTwelveKmCi", f"{fmt(twelve['ci'][0], 2)} to {fmt(twelve['ci'][1], 2)}", src3, "1.63 to 1.87")

    others = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        if path.stem == "deep-8-1-long":
            continue
        for r in rows(path.stem):
            if r.get("t_event") is not None and "prefix" in r["curve"][0]:
                match = [c for c in r["curve"] if c["step"] == r["t_event"]]
                if match:
                    others.append(max(match[0]["prefix"]))
    put("nOtherPrefixMin", fmt(min(others), 2), "every results/*.jsonl except results/deep-8-1-long.jsonl", "0.27")

    misses = [r for r in pool["study-lr1e-3"] if r["cfg"]["L"] == 3 and r["cfg"]["unique"] == 3 and r["t_event"] is not None and r["t_pv"] is None]
    src4 = "results/study-lr1e-3.jsonl"
    put("nMissCount", len(misses), src4, 8)
    put("nMissLayerZeroMax", fmt(max(max(c["prev"][0] for c in r["curve"]) for r in misses), 3), src4)
    tops = [max(c["prev"][1] for c in r["curve"]) for r in misses]
    put("nMissLayerOneMin", fmt(min(tops), 3), src4)
    put("nMissLayerOneMax", fmt(max(tops), 3), src4)
    gaps_twelve = by[((12, 4), 0.003, "uniform tiled")]
    put("nQuantTwelve", fmt(25 / gaps_twelve["medianStep"], 3), "25 over the (12,4) median, results/study-summary.json")
    tied = by[((3, 1), 0.003, "uniform tiled")]
    same("(3,1) budget", tied["budget"], 6000)
    return uniform


def curriculum_numbers():
    third = js("third-summary.json")["curriculum"]
    src = "results/third-summary.json"
    rule_bi = third["rule, Howe's language, pRep 1, periods [4, 32): clone against tiled"]
    rule_uni = third["rule, uniform, pRep 0.75, periods [4, 60): tiled against clone"]
    range_uni = third["range, uniform tiled, pRep 1: [4, 60) against [4, 32)"]
    range_bi = third["range, Howe's language clone, pRep 0.75: [4, 32) against [4, 60)"]
    put("nRuleBiCloneEv", rule_bi["a"]["events"], src, 2)
    put("nRuleBiCloneN", rule_bi["a"]["n"], src, 16)
    put("nRuleBiTiledEv", rule_bi["b"]["events"], src, 16)
    put("nRuleBiTiledMed", rule_bi["b"]["medianStep"], src, 1025)
    put("nRuleUniTiledEv", rule_uni["a"]["events"], src, 12)
    put("nRuleUniTiledMed", rule_uni["a"]["medianStep"], src, 2275)
    put("nRuleUniCloneEv", rule_uni["b"]["events"], src, 0)
    put("nRangeUniLongEv", range_uni["a"]["events"], src, 16)
    put("nRangeUniLongMed", range_uni["a"]["medianStep"], src, 1025)
    put("nRangeUniRatio", fmt(range_uni["ratio"], 2), src, "1.78")
    put("nRangeUniCi", f"{fmt(range_uni['ci'][0], 2)} to {fmt(range_uni['ci'][1], 2)}", src, "1.61 to 1.87")
    put("nRangeBiShortEv", range_bi["a"]["events"], src, 1)
    put("nRangeBiLongEv", range_bi["b"]["events"], src, 12)
    put("nRangeBiLongMed", range_bi["b"]["medianStep"], src, 7500)

    high = rows("crossed-range-uniform")
    low = rows("crossed-rule-uniform")
    src = "results/crossed-rule-uniform.jsonl over results/crossed-range-uniform.jsonl"
    same("repeat-rate arms differ only in pRep", {**high[0]["cfg"], "pRep": None}, {**low[0]["cfg"], "pRep": None})
    same("high arm median", median_of(high), rule_range_median(third))
    ratio = median_of(low) / median_of(high)
    lo, hi = boot_ratio(low, high)
    put("nRateRatio", fmt(ratio, 2), src)
    put("nRateCi", f"{fmt(lo, 2)} to {fmt(hi, 2)}", src)
    put("nPRepLow", low[0]["cfg"]["pRep"], "results/crossed-rule-uniform.jsonl", 0.75)
    put("nRangeLong", f"[{low[0]['cfg']['rMin']}, {low[0]['cfg']['rMax']})", "results/crossed-rule-uniform.jsonl", "[4, 60)")
    put("nCrossedBudget", low[0]["budget"], "results/crossed-rule-uniform.jsonl", 8000)
    bigram_rate_high = rows("crossed-rule-bigram")
    bigram_rate_low = rows("crossed-range-bigram")
    same("bigram repeat-rate arms differ only in pRep", {**bigram_rate_high[0]["cfg"], "pRep": None}, {**bigram_rate_low[0]["cfg"], "pRep": None})
    put("nRuleBiBound", fmt(low[0]["budget"] / rule_bi["b"]["medianStep"], 2), "8000 over the tiled median, results/third-summary.json")
    put("nRuleUniBound", fmt(low[0]["budget"] / rule_uni["a"]["medianStep"], 2), "8000 over the tiled median, results/third-summary.json")
    put("nCloneWeakBound", fmt(low[0]["budget"] / range_bi["b"]["medianStep"], 2), "8000 over the clone median, results/third-summary.json")

    how = js("third-summary.json")["tabOnHowe"]
    put("nTabHoweN", how["n"], "results/third-summary.json", 8)
    put("nTabHoweEv", how["events"], "results/third-summary.json", 0)
    put("nTabHoweBudget", rows("tab-howe-curriculum")[0]["budget"], "results/tab-howe-curriculum.jsonl", 8000)


def rule_range_median(third):
    return third["range, uniform tiled, pRep 1: [4, 60) against [4, 32)"]["a"]["medianStep"]


def forecast_numbers():
    both = js("third-summary.json")["spacing"]
    planned, extended, spacing = both["asPlanned"], both["extended"], both["allSeeds"]
    src = "results/third-summary.json"
    stopped = planned["stoppedBeforeTheCoarseEvent"]
    put("nSpacingPlanned", planned["seeds"], src, 61)
    put("nSpacingPlannedSame", planned["sameEventAsStudy"], src, 61)
    same("the plan's readable seeds also match on the anchor", planned["sameAnchorAsStudy"], 61)
    put("nSpacingStopped", len(stopped), src, 3)
    put("nSpacingStoppedSeeds", f"{stopped[0]}, {stopped[1]} and {stopped[2]}", src, "17, 23 and 51")
    put("nPlanStopPast", both["planEventsPast"], "research/PREREG-3.md", 75)
    put("nExtraStopPast", extended["stopPast"], "results/fine-2-2-extra.jsonl", 400)
    put("nSpacingSeeds", spacing["seeds"], src, 64)
    put("nSpacingSameEvent", spacing["sameEventAsStudy"], src, 64)
    put("nSpacingSameAnchor", spacing["sameAnchorAsStudy"], src, 64)
    put("nGapFine", fmt(spacing["gapEvery5"], 3), src, "0.748")
    put("nGapCoarse", fmt(spacing["gapEvery25"], 3), src, "0.727")
    same("the three added seeds do not move the fine median", fmt(planned["gapEvery5"], 3), registry["nGapFine"][0])
    same("the three added seeds do not move the coarse median", fmt(planned["gapEvery25"], 3), registry["nGapCoarse"][0])
    put("nGapDiff", fmt(spacing["pairedMedianDifference"], 3), src)
    put("nShareFine", fmt(100 * spacing["inEnvelopeEvery5"], 0), src)
    put("nShareCoarse", fmt(100 * spacing["inEnvelopeEvery25"], 0), src)
    put("nEventFine", int(spacing["eventEvery5"]), src)
    put("nEventCoarse", int(spacing["eventEvery25"]), src)
    fine = rows("fine-2-2")
    put("nEvalFine", fine[0]["cfg"]["evalEvery"], "results/fine-2-2.jsonl", 5)
    put("nFineRuns", len(fine), "results/fine-2-2.jsonl", 64)

    src = "results/anchor-summary.json"
    for row in js("anchor-summary.json"):
        tag = {64: "SixtyFour", 128: "OneTwentyEight"}[row["T"]] + WORDS[row["u"]]
        put(f"nAncFine{tag}", fmt(row["A, 0.10, every 5"]["median"], 3), src)
        put(f"nAncCoarse{tag}", fmt(row["A, 0.10, every 25"]["median"], 3), src)
        put(f"nAncRescaled{tag}", fmt(row["B, 2.86 baseline, every 5"]["median"], 3), src)
        put(f"nAncEvents{tag}", row["events"], src)
        put(f"nAncRuns{tag}", row["seeds"], src)
    stated = {row["T"]: row for row in js("anchor-summary.json") if row["u"] == 1}
    same("the rescaled anchor's miss", fmt(stated[64]["B, 2.86 baseline, every 5"]["median"], 3), "0.903")
    ci = stated[64]["B, 2.86 baseline, every 5"]["ci"]
    put("nAncMissCi", f"{fmt(ci[0], 3)} to {fmt(ci[1], 3)}", src)
    same("T=128 stalls, 2 blocks", sum(r["t_event"] is None for r in rows("p2-T128-L2-u2")), 5)
    same("T=128 stalls, 1 block", sum(r["t_event"] is None for r in rows("p2-T128-L2-u1")), 2)
    put("nStallTwo", sum(r["t_event"] is None for r in rows("p2-T128-L2-u2")), "results/p2-T128-L2-u2.jsonl", 5)
    put("nStallOne", sum(r["t_event"] is None for r in rows("p2-T128-L2-u1")), "results/p2-T128-L2-u1.jsonl", 2)
    put("nPlanTwoBudget", rows("p2-T128-L2-u1")[0]["budget"], "results/p2-T128-L2-u1.jsonl", 6000)
    put("nPlanTwoT", rows("p2-T128-L2-u1")[0]["cfg"]["T"], "results/p2-T128-L2-u1.jsonl", 128)
    seeds = [r["seed"] for r in rows("p2-T128-L2-u1")]
    put("nPlanTwoSeedLow", min(seeds), "results/p2-T128-L2-u1.jsonl", 101)
    put("nPlanTwoSeedHigh", max(seeds), "results/p2-T128-L2-u1.jsonl", 164)


def heads_numbers():
    heads = js("heads-looped-2-1.json")
    src = "results/heads-looped-2-1.json"
    runs = heads["rows"]
    put("nHeadSeeds", len(runs), src, 8)
    put("nHeadSteps", heads["steps"], src)
    put("nHeadBothMin", min(r["both"] for r in runs), src, 2)
    put("nHeadBothMax", max(r["both"] for r in runs), src, 3)
    put("nHeadPrevMin", fmt(min(max(r["prev"][0]) for r in runs), 2), src, "0.42")
    put("nHeadPrevMax", fmt(max(max(r["prev"][0]) for r in runs), 2), src, "0.47")
    put("nHeadPrefixMin", fmt(min(min(r["prefix"][1]) for r in runs), 2), src, "0.56")
    put("nHeadPrefixMax", fmt(max(max(r["prefix"][1]) for r in runs), 2), src, "0.66")


def corrections_and_dates():
    readme = (ROOT / "README.md").read_text()
    overlap = re.search(r"top-8 overlap of (\d+) to (\d+)% against (\d+) to (\d+)%", readme)
    put("nOverlapRange", f"{overlap.group(1)} to {overlap.group(2)}", "README.md, quoting arXiv 2609.01343", "56 to 66")
    put("nOverlapControl", f"{overlap.group(3)} to {overlap.group(4)}", "README.md, quoting arXiv 2609.01343", "28 to 34")
    sooner = re.search(r"sooner on average \((\d+)k training steps against (\d+)k\)", (ROOT / "research/NOVELTY-2.md").read_text())
    put("nSoonerLooped", f"{sooner.group(1)}k", "research/NOVELTY-2.md, quoting arXiv 2610.24135", "45k")
    put("nSoonerUntied", f"{sooner.group(2)}k", "research/NOVELTY-2.md, quoting arXiv 2610.24135", "55k")
    old = re.search(r"time ratio 1\.55, ([\d.]+), ([\d.]+) at depth 2, 3, 4", (ROOT / "research/NOVELTY-2.md").read_text())
    put("nOldFitThree", old.group(1), "research/NOVELTY-2.md, written before the fix", "2.20")
    put("nOldFitFour", old.group(2), "research/NOVELTY-2.md, written before the fix", "2.83")
    novelty = (ROOT / "research/NOVELTY-2.md").read_text()
    put("nSearchCalls", re.search(r"Coverage is (\d+) search calls", novelty).group(1), "research/NOVELTY-2.md", 13)
    put("nBootResamples", re.search(r"^BOOT = (\d+)$", (ROOT / "tools/analyze_study.py").read_text(), re.M).group(1), "tools/analyze_study.py", 2000)
    put("nPlanOneShare", re.search(r"about (\d+)% of the 3e-3 arms", (ROOT / "research/PREREG.md").read_text()).group(1), "research/PREREG.md", 70)
    for name, path in (("nDatePlanOne", "research/PREREG.md"), ("nDatePlanTwo", "research/PREREG-2.md"), ("nDatePlanThree", "research/PREREG-3.md")):
        date = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", path], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        put(name, date, f"git log, {path}")


ALLOWED_PHRASES = ("depth 12", "12 layers", "12-layer", "32-layer", "32-bit", "95\\%", "section 4.1")


def bare_numbers(body):
    """Numbers typed into the prose. Only macros may carry a decimal, a count of two or more digits, or scientific notation."""
    text = re.sub(r"(?<!\\)%.*", "", body)
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.S)
    text = re.sub(r"\\(label|ref|cite|url|input|href|includegraphics|texttt)(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\arm\{\d+\}\{\d+\}", "", text)
    for phrase in ALLOWED_PHRASES:
        text = text.replace(phrase, "")
    return re.findall(r"\d+\.\d+|\b\d{2,}\b|\b\d+e-\d+\b", text)


def scan_rules(paths):
    banned = [
        (r"\u2014|\u2013|---", "a long dash"),
        (r"/Users/|~/", "a local path"),
        (r"\b(delve|crucially|novel)\b", "a forbidden word"),
        (r"\b(he|him|his|she|her|hers)\b", "a gendered pronoun"),
        (r"\bthe first (to|we|browser|demo|paper|study|work|trainer)\b", "a priority claim"),
    ]
    problems = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for pattern, what in banned:
                if re.search(pattern, line, re.I):
                    problems.append(f"{path.name}:{number}: {what}: {line.strip()[:100]}")
            if re.search(r"replicat", line, re.I) and "one-draw" not in line:
                problems.append(f"{path.name}:{number}: replication without the one-draw qualifier: {line.strip()[:100]}")
    return problems


def check_paper():
    if not TEX.exists():
        return ["paper/induction-tab.tex does not exist yet"]
    tex = TEX.read_text()
    body = tex[tex.index("\\begin{document}"):]
    problems = []
    for name in sorted(set(re.findall(r"\\(n[A-Z][A-Za-z]*)", tex))):
        if name not in registry:
            problems.append(f"paper uses \\{name}, which this script does not define")
    problems += [f"bare number in the paper's text: {n}" for n in bare_numbers(body)]
    generated = sorted(PAPER.glob("fig-*.tex"))
    written = [TEX, PAPER / "README.md", PAPER / "SUBMIT.md", PAPER / "figures.py", Path(__file__).resolve()]
    problems += scan_rules([p for p in written + generated if p.exists() and p != Path(__file__).resolve()])
    for path in (TEX, PAPER / "README.md", PAPER / "SUBMIT.md"):
        if path.exists() and re.search(r"(?<![\w-])--(?![\w-])", path.read_text()):
            problems.append(f"{path.name}: two hyphens render as an en dash in LaTeX")
    return problems


def main():
    config_numbers()
    timing_numbers()
    parity_numbers()
    trainer_numbers()
    howe_numbers()
    config_a_numbers()
    study_numbers()
    curriculum_numbers()
    forecast_numbers()
    heads_numbers()
    corrections_and_dates()
    second_pass_numbers()
    correction_numbers()

    width = max(len(n) for n in registry)
    for name, (value, source) in sorted(registry.items()):
        print(f"{name:<{width}}  {value:<28}  {source}")

    lines = ["% Generated by paper/numbers.py from results/ and research/. Do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, (value, _) in sorted(registry.items())]
    generated = "\n".join(lines) + "\n"
    if "--write" in sys.argv:
        MACROS.write_text(generated)
    elif not MACROS.exists() or MACROS.read_text() != generated:
        mismatches.append("paper/numbers.tex is stale, run with --write")
    problems = mismatches + check_paper()
    print(f"\n{len(registry)} values.")
    if problems:
        print("MISMATCH", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)
    print("Every value agrees with its file, and the paper's text carries no bare number.")


if __name__ == "__main__":
    main()
