"""Generate publication-quality figures for the paper from the saved CSV / JSONL data.

Run: python paper/make_figures.py
Outputs to paper/figures/
"""
from __future__ import annotations

import csv
import json
import os
import statistics as stat
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 140,
})


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"WARNING: missing {path}", file=sys.stderr)
        return []
    return list(csv.DictReader(path.open()))


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


# =============================================================================
# Figure 1 — EXP-1 baseline: minority survival by condition
# =============================================================================

def fig_exp1_minority_survival() -> None:
    rows = _load_csv(ROOT / "runs/20260525-075354_sweep_exp1/sweep_results.csv")
    if not rows: return

    conditions = ["control", "likes", "majority", "leaderboard", "downvote"]
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["condition"]].append(float(r["minority_survival_rate"]))

    means = [_mean_std(buckets[c])[0] for c in conditions]
    stds = [_mean_std(buckets[c])[1] for c in conditions]
    ns = [len(buckets[c]) for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#555555", "#4C78A8", "#E45756", "#F2A541", "#8F6BB3"]
    bars = ax.bar(conditions, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.6)
    # control reference line
    ax.axhline(means[0], color="#444", linestyle="--", alpha=0.4, linewidth=1,
               label=f"control mean ({means[0]:.2f})")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + s + 0.015,
                f"{m:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Minority-opinion survival rate")
    ax.set_xlabel("Feedback condition")
    ax.set_title("EXP-1 — Minority-opinion survival under social-reward UX\n(mean ± SD; n=4 model×seed combinations per condition)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out = FIGDIR / "fig1_exp1_minority_survival.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 2 — EXP-2 dose-response (K vs broad-side share, n=4 per K)
# =============================================================================

def fig_exp2_dose_response() -> None:
    rows = _load_csv(ROOT / "runs/20260524-015245_astroturf_exp2/astroturf_broadside_aggregated.csv")
    if not rows: return

    Ks = [int(r["K"]) for r in rows]
    fb_mean = [float(r["final_broad_share_mean"]) for r in rows]
    fb_std = [float(r["final_broad_share_std"]) for r in rows]
    fs_mean = [float(r["final_strict_share_mean"]) for r in rows]
    fs_std = [float(r["final_strict_share_std"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(Ks, fb_mean, yerr=fb_std, marker="o", linewidth=2,
                color="#4C78A8", capsize=4, label="Broad-side stance share (3 buckets)")
    ax.errorbar(Ks, fs_mean, yerr=fs_std, marker="s", linewidth=2,
                color="#E45756", capsize=4, label="Exact pushed-stance share")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, linewidth=1, label="Majority threshold (0.50)")
    ax.axvspan(15, 22, color="#F2A541", alpha=0.10, label="Highest tested coordination (K=20)")
    ax.set_xlabel("K — number of coordinated AI accounts injected\n(honest population = 30 agents)")
    ax.set_ylabel("Final honest-agent share adopting pushed opinion")
    ax.set_title("EXP-2 — Honest-agent stance share by coordinated-account count\n(mean ± SD; n=4 model×seed combinations per K)")
    ax.set_xticks(Ks)
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    plt.tight_layout()
    out = FIGDIR / "fig2_exp2_dose_response.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 3 — EXP-2 information-ecology metrics by K (4-panel)
# =============================================================================

def fig_exp2_ecology() -> None:
    rows = _load_csv(ROOT / "runs/20260524-015245_astroturf_exp2/astroturf_results.csv")
    if not rows: return

    # Need to load the per-trial scalar.json too — find them
    K_to_scalars = defaultdict(list)
    trial_root = ROOT / "runs/20260524-015245_astroturf_exp2"
    for r in rows:
        if "_error" in r: continue
        model = r["model"].replace("/", "__").replace(":", "_")
        K = int(r["K"])
        seed = r["seed"]
        path = trial_root / model / f"seed{seed}" / f"k{K:03d}" / "likes" / "metrics" / "scalar.json"
        if path.exists():
            with path.open() as f:
                sc = json.load(f)
            K_to_scalars[K].append(sc)

    Ks_sorted = sorted(K_to_scalars.keys())
    if not Ks_sorted: return

    def metric_stats(key: str) -> tuple[list[float], list[float]]:
        means, stds = [], []
        for K in Ks_sorted:
            vals = [float(s.get(key, 0.0)) for s in K_to_scalars[K]]
            m, sd = _mean_std(vals)
            means.append(m); stds.append(sd)
        return means, stds

    pairwise_m, pairwise_s = metric_stats("final_mean_pairwise_sim")
    minor_m, minor_s = metric_stats("minority_survival_rate")
    pers_m, pers_s = metric_stats("final_persona_retention")
    coll_m, coll_s = metric_stats("persona_collapse_score")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    panels = [
        (axes[0, 0], "Pairwise post similarity", pairwise_m, pairwise_s, "linguistic homogeneity"),
        (axes[0, 1], "Minority-opinion survival", minor_m, minor_s, "minority view persistence"),
        (axes[1, 0], "Persona retention", pers_m, pers_s, "in-persona writing"),
        (axes[1, 1], "Composite collapse score", coll_m, coll_s, "higher = more collapse"),
    ]
    for ax, title, m, s, sub in panels:
        ax.errorbar(Ks_sorted, m, yerr=s, marker="o", linewidth=2, color="#4C78A8", capsize=3)
        ax.set_title(f"{title}\n({sub})", fontsize=10)
        ax.set_xlabel("K coordinated agents")
        ax.set_xticks(Ks_sorted)
        ax.grid(alpha=0.3)
    fig.suptitle("EXP-2 — Information-ecology metrics by coordinated-account count",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIGDIR / "fig3_exp2_ecology.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 4 — EXP-6 intervention semantic-endorsement bar
# =============================================================================

def fig_exp6_interventions() -> None:
    rows = _load_csv(ROOT / "runs/20260524-225945_misinfo_exp6/misinfo_semantic_aggregated.csv")
    if not rows: return

    interventions = []
    delta_mean, delta_std = [], []
    for r in rows:
        interventions.append(r["intervention"])
        delta_mean.append(float(r["delta_similarity_mean"]))
        delta_std.append(float(r["delta_similarity_std"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#7A7A7A", "#4C78A8", "#72A0C1", "#9E9E9E"]
    bars = ax.bar(interventions, delta_mean, yerr=delta_std, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    for bar, m in zip(bars, delta_mean):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + (0.005 if m >= 0 else -0.005),
                f"{m:+.3f}",
                ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=9)
    ax.set_ylabel("Δ semantic similarity to seeded claim (final − initial)\npost-hoc thematic metric; not stance-aware")
    ax.set_xlabel("Intervention regime")
    ax.set_title("EXP-6 — Post-hoc semantic theme drift by intervention\n(mean ± SD; n=4 model×seed combinations per intervention)")
    ax.grid(axis="y", alpha=0.3)
    # Annotation about rebuttal caveat
    ax.annotate("Similarity can increase when agents\nrefute the same thematic content",
                xy=(3, delta_mean[3]),
                xytext=(1.9, delta_mean[3] + 0.075),
                arrowprops=dict(arrowstyle="->", alpha=0.5, color="gray"),
                fontsize=8, color="gray", ha="left")
    plt.tight_layout()
    out = FIGDIR / "fig4_exp6_interventions.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 5 — EXP-3 visibility multiplier dose-response
# =============================================================================

def fig_exp3_influence() -> None:
    rows = _load_csv(ROOT / "runs/20260525-221419_influence_exp3/influence_results.csv")
    if not rows: return

    mults = sorted({int(r["multiplier"]) for r in rows})
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["multiplier"])].append(r)

    broad_m, broad_s = [], []
    strict_m, strict_s = [], []
    for m in mults:
        broad_vals = [float(r["final_broad_share"]) for r in buckets[m]]
        strict_vals = [float(r["final_strict_share"]) for r in buckets[m]]
        bm, bsd = _mean_std(broad_vals); broad_m.append(bm); broad_s.append(bsd)
        sm, ssd = _mean_std(strict_vals); strict_m.append(sm); strict_s.append(ssd)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(mults, broad_m, yerr=broad_s, marker="o", linewidth=2,
                color="#4C78A8", capsize=4, label="Broad-side share (3 buckets)")
    ax.errorbar(mults, strict_m, yerr=strict_s, marker="s", linewidth=2,
                color="#E45756", capsize=4, label="Exact pushed-stance share")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, linewidth=1, label="Majority threshold")
    ax.set_xlabel("Visibility multiplier of single influencer agent")
    ax.set_ylabel("Final honest-agent share adopting pushed opinion")
    ax.set_title("EXP-3 — Single-account visibility amplification\nNo monotonic dose-response (mean ± SD; n=4 per multiplier)")
    ax.set_xticks(mults)
    ax.set_xticklabels([f"{m}×" for m in mults])
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "fig5_exp3_influence.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 6 — Cross-model heterogeneity (EXP-6 misinfo Δ per model × intervention)
# =============================================================================

def fig_cross_model() -> None:
    rows = _load_csv(ROOT / "runs/20260524-225945_misinfo_exp6/misinfo_semantic_results.csv")
    if not rows: return

    # Aggregate per (model, intervention)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["model"], r["intervention"])].append(float(r["delta_mean_similarity"]))

    interventions = ["none", "factcheck_label", "deamplify", "rebuttal"]
    models = sorted({r["model"] for r in rows})

    x = np.arange(len(interventions))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    color_map = {"qwen3.5:2b": "#4C78A8", "llama3.2:3b": "#F58518"}
    for i, model in enumerate(models):
        means, stds = [], []
        for intv in interventions:
            vals = buckets.get((model, intv), [])
            m, sd = _mean_std(vals)
            means.append(m); stds.append(sd)
        offset = (i - (len(models) - 1)/2) * width
        ax.bar(x + offset, means, width=width*0.95, yerr=stds, capsize=4,
               color=color_map.get(model, f"C{i}"), edgecolor="black", linewidth=0.6,
               label=model)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.set_ylabel("Δ semantic similarity to seeded claim\n(post-hoc thematic metric; not stance-aware)")
    ax.set_xlabel("Intervention")
    ax.set_title("EXP-6 — Semantic theme drift differs across two tested models\n"
                 "(n=2 seeds per model and intervention)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.95, fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "fig6_cross_model.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 7 — Calibration anchor (A1) summary
# =============================================================================

def fig_anchor() -> None:
    rows = _load_csv(ROOT / "runs/20260525-213406_anchor_a1/anchor_aggregated.csv")
    if not rows: return

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    HUMAN_LO, HUMAN_HI = 0.10, 0.20
    ax.axhspan(HUMAN_LO, HUMAN_HI, color="#54A24B", alpha=0.18,
               label=f"Derived human reference band ({HUMAN_LO:.0%}–{HUMAN_HI:.0%})\n"
                     "(constructed from Salganik et al. 2006; Muchnik et al. 2013)")
    color_map = {"qwen3.5:2b": "#4C78A8", "llama3.2:3b": "#F58518"}
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        xs = [float(r["claimed_majority"]) for r in mrows]
        ys = [float(r["conform_rate_mean"]) for r in mrows]
        es = [float(r["conform_rate_std"]) for r in mrows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=2,
                    color=color_map.get(model), label=model)
    ax.set_xlabel("Claimed majority strength shown to agents")
    ax.set_ylabel("Conform rate (minority → majority shift)")
    ax.set_title("A1 — One-shot bandwagon conformity\n"
                 "Pooled agent rate 0.039 versus derived 0.10–0.20 human reference band")
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(-0.02, 0.32)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "fig7_anchor_calibration.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    fig_exp1_minority_survival()
    fig_exp2_dose_response()
    fig_exp2_ecology()
    fig_exp6_interventions()
    fig_exp3_influence()
    fig_cross_model()
    fig_anchor()
    print(f"\nAll figures saved to {FIGDIR}/")
