"""Small-sample robustness analysis for the production experiments.

The run-level analysis unit is a model-by-seed block. Post-level observations
are not treated as independent replicates, and the fixed model families are
not treated as a random sample of models.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT_MD = ROOT / "paper" / "robustness_results.md"
OUT_CSV = ROOT / "paper" / "robustness_results.csv"
RNG = np.random.default_rng(20260729)

EXP1 = RUNS / "20260525-075354_sweep_exp1" / "sweep_results.csv"
EXP2 = RUNS / "20260524-015245_astroturf_exp2" / "astroturf_results.csv"
EXP2_BROAD = (
    RUNS / "20260524-015245_astroturf_exp2" / "astroturf_broadside_results.csv"
)
EXP3 = RUNS / "20260525-221419_influence_exp3" / "influence_results.csv"
EXP6 = RUNS / "20260524-225945_misinfo_exp6" / "misinfo_semantic_results.csv"
A1 = RUNS / "20260525-213406_anchor_a1" / "anchor_results.csv"


def paired_exact_p(differences: np.ndarray) -> float:
    """Two-sided exact sign-flip randomization p-value for paired differences."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.mean())
    randomized = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        randomized.append(abs(np.mean(differences * np.asarray(signs))))
    return float(np.mean(np.asarray(randomized) >= observed - 1e-12))


def bootstrap_ci(differences: np.ndarray, draws: int = 50_000) -> tuple[float, float]:
    """Descriptive percentile interval resampling model-by-seed blocks."""
    differences = np.asarray(differences, dtype=float)
    samples = RNG.choice(differences, size=(draws, len(differences)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def paired_result(
    experiment: str,
    contrast: str,
    metric: str,
    differences: np.ndarray,
) -> dict[str, object]:
    differences = np.asarray(differences, dtype=float)
    low, high = bootstrap_ci(differences)
    return {
        "experiment": experiment,
        "contrast": contrast,
        "metric": metric,
        "n_blocks": len(differences),
        "mean_difference": differences.mean(),
        "ci_95_low": low,
        "ci_95_high": high,
        "exact_p": paired_exact_p(differences),
        "positive_blocks": int((differences > 0).sum()),
        "negative_blocks": int((differences < 0).sum()),
        "zero_blocks": int((differences == 0).sum()),
        "block_differences": "; ".join(f"{value:.6f}" for value in differences),
    }


def paired_differences(
    frame: pd.DataFrame,
    treatment_col: str,
    treatment: object,
    control: object,
    metric: str,
) -> np.ndarray:
    pivot = frame.pivot(index=["model", "seed"], columns=treatment_col, values=metric)
    return (pivot[treatment] - pivot[control]).to_numpy()


def within_block_slopes(
    frame: pd.DataFrame,
    dose_col: str,
    metric: str,
    transform=lambda values: values,
) -> np.ndarray:
    slopes = []
    for _, block in frame.groupby(["model", "seed"]):
        x = transform(block[dose_col].to_numpy(dtype=float))
        y = block[metric].to_numpy(dtype=float)
        slopes.append(np.polyfit(x, y, 1)[0])
    return np.asarray(slopes)


def model_means(
    frame: pd.DataFrame,
    treatment_col: str,
    treatment: object,
    control: object,
    metric: str,
) -> str:
    pivot = frame.pivot(index=["model", "seed"], columns=treatment_col, values=metric)
    diff = (pivot[treatment] - pivot[control]).groupby(level="model").mean()
    return "; ".join(f"{model}: {value:+.4f}" for model, value in diff.items())


def main() -> None:
    results: list[dict[str, object]] = []

    exp1 = pd.read_csv(EXP1)
    for condition in ("likes", "majority", "leaderboard", "downvote"):
        for metric in (
            "minority_survival_rate",
            "final_majority_fraction",
            "final_mean_pairwise_sim",
            "final_persona_retention",
        ):
            diff = paired_differences(exp1, "condition", condition, "control", metric)
            row = paired_result("EXP-1", f"{condition} - control", metric, diff)
            row["model_mean_differences"] = model_means(
                exp1, "condition", condition, "control", metric
            )
            results.append(row)

    exp2 = pd.read_csv(EXP2)
    for metric in (
        "baseline_final_mean_pairwise_sim",
        "baseline_minority_survival_rate",
        "baseline_delta_majority_fraction",
        "baseline_persona_collapse_score",
    ):
        diff = paired_differences(exp2, "K", 20, 0, metric)
        row = paired_result("EXP-2", "K=20 - K=0", metric, diff)
        row["model_mean_differences"] = model_means(exp2, "K", 20, 0, metric)
        results.append(row)

    exp2_broad = pd.read_csv(EXP2_BROAD)
    for metric in ("delta_strict_share", "delta_broad_share"):
        diff = paired_differences(exp2_broad, "K", 20, 0, metric)
        row = paired_result("EXP-2", "K=20 - K=0", metric, diff)
        row["model_mean_differences"] = model_means(
            exp2_broad, "K", 20, 0, metric
        )
        results.append(row)
        slopes = within_block_slopes(exp2_broad, "K", metric)
        slope_row = paired_result("EXP-2", "linear slope over K", metric, slopes)
        slope_row["model_mean_differences"] = "slope units per coordinated account"
        results.append(slope_row)

    strict_pivot = exp2_broad.pivot(
        index=["model", "seed"], columns="K", values="strict_flipped"
    ).astype(bool)
    strict_k0 = strict_pivot[0]
    strict_k20 = strict_pivot[20]
    gained_flip = int((strict_k20 & ~strict_k0).sum())
    lost_flip = int((~strict_k20 & strict_k0).sum())
    discordant = gained_flip + lost_flip
    mcnemar_p = (
        binomtest(gained_flip, discordant, 0.5, alternative="two-sided").pvalue
        if discordant
        else 1.0
    )

    exp3 = pd.read_csv(EXP3)
    for metric in (
        "delta_strict_share",
        "delta_broad_share",
        "baseline_final_mean_pairwise_sim",
        "baseline_minority_survival_rate",
    ):
        diff = paired_differences(exp3, "multiplier", 20, 1, metric)
        row = paired_result("EXP-3", "20x - 1x", metric, diff)
        row["model_mean_differences"] = model_means(
            exp3, "multiplier", 20, 1, metric
        )
        results.append(row)
        slopes = within_block_slopes(exp3, "multiplier", metric, np.log2)
        slope_row = paired_result(
            "EXP-3", "linear slope over log2(multiplier)", metric, slopes
        )
        slope_row["model_mean_differences"] = "slope units per doubling"
        results.append(slope_row)

    exp6 = pd.read_csv(EXP6)
    for intervention in ("factcheck_label", "deamplify", "rebuttal"):
        for metric in ("delta_mean_similarity", "delta_n_thematically_aligned"):
            diff = paired_differences(
                exp6, "intervention", intervention, "none", metric
            )
            row = paired_result("EXP-6", f"{intervention} - none", metric, diff)
            row["model_mean_differences"] = model_means(
                exp6, "intervention", intervention, "none", metric
            )
            results.append(row)

    a1 = pd.read_csv(A1)
    a1_summary = []
    for model, group in [("pooled", a1), *a1.groupby("model")]:
        shifted = int(group["n_shifted"].sum())
        total = int(group["n_minority"].sum())
        a1_summary.append(
            (
                model,
                shifted,
                total,
                shifted / total,
                group["conform_rate"].mean(),
                group["conform_rate"].min(),
                group["conform_rate"].max(),
            )
        )

    output = pd.DataFrame(results)
    output.to_csv(OUT_CSV, index=False)

    lines = [
        "# Robustness analysis",
        "",
        "The model-by-seed block is the run-level analysis unit. The two model "
        "families are fixed, and each has two random seeds. With four paired blocks, "
        "a two-sided exact sign-flip test cannot produce p < 0.125, even when all "
        "four differences have the same sign. Bootstrap intervals are descriptive "
        "and unstable at this sample size; neither summary estimates uncertainty "
        "over the population of models or topics.",
        "",
        "## Paired contrasts",
        "",
        "| Experiment | Contrast | Metric | Mean difference | 95% block-bootstrap CI | Exact p | Signs (+/-/0) | Model means |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| {experiment} | {contrast} | {metric} | {mean_difference:+.4f} | "
            "[{ci_95_low:+.4f}, {ci_95_high:+.4f}] | {exact_p:.3f} | "
            "{positive_blocks}/{negative_blocks}/{zero_blocks} | "
            "{model_mean_differences} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## EXP-2 strict flips",
            "",
            f"K=20 produced {int(strict_k20.sum())}/4 strict flips; K=0 produced "
            f"{int(strict_k0.sum())}/4. Paired exact McNemar p = {mcnemar_p:.3f}. "
            "Both K=20 flips occurred under seed 7.",
            "",
            "## A1 one-shot bandwagon shifts",
            "",
            "| Model | Shifted | Exposed minority agents | Pooled rate | Mean trial rate | Trial range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, shifted, total, rate, trial_mean, trial_low, trial_high in a1_summary:
        lines.append(
            f"| {model} | {shifted} | {total} | {rate:.3f} | "
            f"{trial_mean:.3f} | [{trial_low:.3f}, {trial_high:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- These analyses estimate consistency across four model-by-seed blocks, "
            "not uncertainty across topics, model families, or human populations.",
            "- EXP-2 and EXP-3 are not a matched causal comparison: EXP-2 changes "
            "population size and coordinated-account count, while EXP-3 changes "
            "the rank and forced inclusion of one account; feed depth also differs.",
            "- EXP-6 cosine similarity measures thematic proximity, not endorsement. "
            "It cannot establish misinformation adoption without stance-aware labels.",
            "- A1's literature-derived human range is not a preregistered or "
            "like-for-like human benchmark.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
