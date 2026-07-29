"""Generate publication-quality figures for the paper from the saved CSV / JSONL data.

Run: python paper/make_figures.py
Outputs to paper/figures/
"""
from __future__ import annotations

import csv
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
    blocks = defaultdict(dict)
    for row in rows:
        blocks[(row["model"], row["seed"])][row["condition"]] = float(
            row["minority_survival_rate"]
        )

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    model_colors = {"qwen3.5:2b": "#4C78A8", "llama3.2:3b": "#F58518"}
    shown = set()
    for (model, _), values in sorted(blocks.items()):
        y = [values[condition] for condition in conditions]
        ax.plot(x, y, color=model_colors[model], alpha=0.32, linewidth=1)
        label = model if model not in shown else None
        ax.scatter(x, y, color=model_colors[model], s=30, alpha=0.8, label=label)
        shown.add(model)
    means = [
        np.mean([values[condition] for values in blocks.values()])
        for condition in conditions
    ]
    ax.plot(x, means, color="black", marker="D", linewidth=2, label="block mean")
    ax.set_ylabel("Minority-opinion survival rate")
    ax.set_xlabel("Feedback condition")
    ax.set_title("EXP-1 — Minority-opinion survival by model×seed block")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9, ncol=3, fontsize=8)
    plt.tight_layout()
    out = FIGDIR / "fig1_exp1_minority_survival.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 2 — EXP-2 dose-response (K vs broad-side share, n=4 per K)
# =============================================================================

def fig_exp2_dose_response() -> None:
    rows = _load_csv(ROOT / "runs/20260524-015245_astroturf_exp2/astroturf_broadside_results.csv")
    if not rows: return

    Ks = sorted({int(row["K"]) for row in rows})
    blocks = defaultdict(dict)
    for row in rows:
        blocks[(row["model"], row["seed"])][int(row["K"])] = row

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = [
        (axes[0], "delta_strict_share", "Exact pushed stance (preregistered)"),
        (axes[1], "delta_broad_share", "Broad side (post-hoc)"),
    ]
    for ax, metric, title in panels:
        for values in blocks.values():
            y = [float(values[K][metric]) for K in Ks]
            ax.plot(Ks, y, color="#7A7A7A", alpha=0.45, marker="o", linewidth=1)
        mean = [
            np.mean([float(values[K][metric]) for values in blocks.values()])
            for K in Ks
        ]
        ax.plot(Ks, mean, color="#B22222", marker="D", linewidth=2.2,
                label="block mean")
        ax.axhline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("K coordinated accounts")
        ax.set_xticks(Ks)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Change in honest-agent stance share")
    axes[1].legend(framealpha=0.95, fontsize=8)
    fig.suptitle("EXP-2 — Coordinated-account dose response with all four blocks")
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

    Ks_sorted = sorted({int(row["K"]) for row in rows})
    blocks = defaultdict(dict)
    for row in rows:
        blocks[(row["model"], row["seed"])][int(row["K"])] = row

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    panels = [
        (axes[0, 0], "baseline_final_mean_pairwise_sim", "Pairwise post similarity", "linguistic homogeneity"),
        (axes[0, 1], "baseline_minority_survival_rate", "Minority-opinion survival", "minority view persistence"),
        (axes[1, 0], "baseline_delta_majority_fraction", "Change in modal-stance share", "majority concentration"),
        (axes[1, 1], "baseline_persona_collapse_score", "Composite collapse score", "higher = more collapse"),
    ]
    for ax, metric, title, sub in panels:
        for values in blocks.values():
            y = [float(values[K][metric]) for K in Ks_sorted]
            ax.plot(Ks_sorted, y, color="#777777", alpha=0.42, marker="o", linewidth=1)
        mean = [
            np.mean([float(values[K][metric]) for values in blocks.values()])
            for K in Ks_sorted
        ]
        ax.plot(Ks_sorted, mean, marker="D", linewidth=2.2, color="#4C78A8")
        ax.set_title(f"{title}\n({sub})", fontsize=10)
        ax.set_xlabel("K coordinated agents")
        ax.set_xticks(Ks_sorted)
        ax.grid(alpha=0.3)
    fig.suptitle("EXP-2 — Information-ecology metrics with all four blocks",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIGDIR / "fig3_exp2_ecology.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figure 4 — EXP-6 intervention thematic-similarity change
# =============================================================================

def fig_exp6_interventions() -> None:
    rows = _load_csv(ROOT / "runs/20260524-225945_misinfo_exp6/misinfo_semantic_results.csv")
    if not rows: return

    interventions = ["none", "factcheck_label", "deamplify", "rebuttal"]
    blocks = defaultdict(dict)
    for row in rows:
        blocks[(row["model"], row["seed"])][row["intervention"]] = float(
            row["delta_mean_similarity"]
        )

    x = np.arange(len(interventions))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for values in blocks.values():
        y = [values[intervention] for intervention in interventions]
        ax.plot(x, y, color="#777777", alpha=0.45, marker="o", linewidth=1)
    means = [
        np.mean([values[intervention] for values in blocks.values()])
        for intervention in interventions
    ]
    ax.plot(x, means, color="#4C78A8", marker="D", linewidth=2.2,
            label="block mean")
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_ylabel("Δ semantic similarity to seeded claim (final − initial)\npost-hoc thematic metric; not stance-aware")
    ax.set_xlabel("Intervention regime")
    ax.set_title("EXP-6 — Thematic-similarity change with all four blocks")
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.95, fontsize=8)
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

    blocks = defaultdict(dict)
    for row in rows:
        blocks[(row["model"], row["seed"])][int(row["multiplier"])] = row

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [
        (axes[0], "delta_strict_share", "Change in exact pushed-stance share"),
        (axes[1], "baseline_minority_survival_rate", "Minority-view survival"),
    ]
    for ax, metric, title in panels:
        for values in blocks.values():
            y = [float(values[m][metric]) for m in mults]
            ax.plot(mults, y, color="#777777", alpha=0.45, marker="o", linewidth=1)
        mean = [
            np.mean([float(values[m][metric]) for values in blocks.values()])
            for m in mults
        ]
        ax.plot(mults, mean, color="#4C78A8", marker="D", linewidth=2.2,
                label="block mean")
        ax.set_title(title)
        ax.set_xlabel("Visibility multiplier")
        ax.set_xticks(mults)
        ax.set_xticklabels([f"{m}×" for m in mults])
        ax.grid(alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].legend(framealpha=0.95, fontsize=8)
    fig.suptitle("EXP-3 — One-account rank amplification with all four blocks")
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
        means = []
        for intv in interventions:
            vals = buckets.get((model, intv), [])
            means.append(float(np.mean(vals)))
        offset = (i - (len(models) - 1)/2) * width
        for j, intv in enumerate(interventions):
            vals = buckets.get((model, intv), [])
            jitter = np.linspace(-0.045, 0.045, len(vals))
            ax.scatter(
                np.full(len(vals), x[j] + offset) + jitter,
                vals,
                color=color_map.get(model, f"C{i}"),
                alpha=0.65,
                s=34,
            )
        ax.plot(
            x + offset,
            means,
            color=color_map.get(model, f"C{i}"),
            marker="D",
            linewidth=1.8,
            label=model,
        )
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.set_ylabel("Δ semantic similarity to seeded claim\n(post-hoc thematic metric; not stance-aware)")
    ax.set_xlabel("Intervention")
    ax.set_title("EXP-6 — Thematic-similarity drift by model and seed")
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
    rows = _load_csv(ROOT / "runs/20260525-213406_anchor_a1/anchor_results.csv")
    if not rows: return

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    color_map = {"qwen3.5:2b": "#4C78A8", "llama3.2:3b": "#F58518"}
    rng = np.random.default_rng(20260729)
    for index, model in enumerate(sorted({r["model"] for r in rows})):
        mrows = [r for r in rows if r["model"] == model]
        ys = [float(row["conform_rate"]) for row in mrows]
        xs = index + rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(xs, ys, color=color_map[model], alpha=0.55, s=28)
        shifted = sum(int(row["n_shifted"]) for row in mrows)
        total = sum(int(row["n_minority"]) for row in mrows)
        rate = shifted / total
        ax.scatter(index, rate, color="black", marker="D", s=55, zorder=4)
        ax.text(index, max(ys + [rate]) + 0.012, f"{shifted}/{total}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(sorted({r["model"] for r in rows}))
    ax.set_xlabel("Generation model")
    ax.set_ylabel("Conform rate (minority → majority shift)")
    ax.set_title("A1 — One-shot majority-cue response by model\n"
                 "trial rates with pooled model means")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.02, 0.30)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGDIR / "fig7_anchor_calibration.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


# =============================================================================
# Figures 8–9 — Preregistered matched-exposure confirmation
# =============================================================================

def _confirmatory_headlines() -> list[tuple[str, str, list[dict]]]:
    panels = [
        ("v1_core", "Core families (4B–8B)"),
        ("v1_size_extension", "Larger variants (9B–14B)"),
    ]
    return [
        (run, title, _load_csv(
            ROOT / "runs_confirmatory" / run / "confirmatory_analysis.csv"
        ))
        for run, title in panels
    ]


def fig_confirmatory_overall() -> None:
    panels = _confirmatory_headlines()
    outcomes = [
        ("pdi_distributed_minus_single", "delta_alignment",
         "Distributed minus single source\npushed-direction change"),
        ("engagement_likes_minus_control", "opposite_side_survival",
         "Peer-ranked feed minus topic-only\nopposite-side survival"),
        ("engagement_likes_minus_control", "final_honest_pairwise_similarity",
         "Peer-ranked feed minus topic-only\nfinal-post TF-IDF similarity"),
    ]
    colors = ["#4C78A8", "#E45756"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3))
    for ax, (contrast, metric, title) in zip(axes, outcomes):
        selected = []
        for _, panel_title, rows in panels:
            row = next(
                item for item in rows
                if item["contrast"] == contrast
                and item["metric"] == metric
                and item["stratum_type"] == "overall"
            )
            selected.append((panel_title, row))
        y = np.arange(len(selected))[::-1]
        for index, ((label, row), ypos) in enumerate(zip(selected, y)):
            mean = float(row["mean_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            ax.errorbar(
                mean, ypos, xerr=[[mean - low], [high - mean]],
                fmt="o", color=colors[index], capsize=4, markersize=7,
                linewidth=2, label=label,
            )
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _ in selected], fontsize=13)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Paired block difference", fontsize=13)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Preregistered confirmatory effects with 95% block-bootstrap intervals",
        y=1.02, fontsize=16,
    )
    plt.tight_layout()
    out = FIGDIR / "fig8_confirmatory_overall.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig_confirmatory_models() -> None:
    panels = [
        (run, title) for run, title, _ in _confirmatory_headlines()
    ]
    outcomes = [
        ("pdi_distributed_minus_single", "delta_alignment",
         "Distributed minus single\npushed-direction change"),
        ("engagement_likes_minus_control", "opposite_side_survival",
         "Peer-ranked feed minus topic-only\nopposite-side survival"),
        ("engagement_likes_minus_control", "final_honest_pairwise_similarity",
         "Peer-ranked feed minus topic-only\nfinal-post TF-IDF similarity"),
    ]
    colors = ["#4C78A8", "#E45756", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 10.5))
    for row_index, (run, panel_title) in enumerate(panels):
        rows = _load_csv(
            ROOT / "runs_confirmatory" / run / "confirmatory_analysis.csv"
        )
        for col_index, (contrast, metric, title) in enumerate(outcomes):
            ax = axes[row_index, col_index]
            model_rows = [
                row for row in rows
                if row["contrast"] == contrast
                and row["metric"] == metric
                and row["stratum_type"] == "model"
            ]
            overall = next(
                row for row in rows
                if row["contrast"] == contrast
                and row["metric"] == metric
                and row["stratum_type"] == "overall"
            )
            labels = [row["stratum"].replace("ministral-3", "ministral3")
                      for row in model_rows]
            means = np.asarray([float(row["mean_difference"]) for row in model_rows])
            low = np.asarray([float(row["ci95_low"]) for row in model_rows])
            high = np.asarray([float(row["ci95_high"]) for row in model_rows])
            y = np.arange(len(model_rows))
            ax.errorbar(
                means, y, xerr=np.vstack((means - low, high - means)),
                fmt="o", color=colors[col_index], capsize=3, markersize=5,
            )
            overall_mean = float(overall["mean_difference"])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(overall_mean, color=colors[col_index], linewidth=1.4,
                       linestyle="--", alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(
                labels if col_index == 0 else [], fontsize=12
            )
            ax.tick_params(axis="x", labelsize=11)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title(title if row_index == 0 else "")
            if col_index == 0:
                ax.set_ylabel(panel_title, fontsize=13)
            ax.set_xlabel(
                f"Paired difference\npooled {overall_mean:+.3f}", fontsize=12
            )
    fig.suptitle(
        "Preregistered confirmatory contrasts by model variant\n"
        "points are model means; bars are 95% block-bootstrap intervals",
        y=0.995, fontsize=16,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.2, w_pad=1.6)
    out = FIGDIR / "fig9_confirmatory_models.png"
    plt.savefig(out, dpi=180)
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
    fig_confirmatory_overall()
    fig_confirmatory_models()
    print(f"\nAll figures saved to {FIGDIR}/")
