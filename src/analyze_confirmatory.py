"""Preregistered block-level analysis for the confirmatory extension."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "delta_alignment",
    "final_pushed_side_share",
    "toward_push_share",
    "opposite_side_survival",
    "majority_capture",
    "final_honest_pairwise_similarity",
)
CONTRASTS = {
    "pdi_distributed_minus_single": ("distributed_sources", "single_source"),
    "engagement_likes_minus_control": ("likes", "control"),
    "single_minus_organic": ("single_source", "likes"),
    "distributed_minus_organic": ("distributed_sources", "likes"),
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    numeric = (set(METRICS) - {"majority_capture"}) | {
        "parse_error_rate", "vote_parse_error_rate", "elapsed_seconds",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
        row["seed"] = int(row["seed"])
        if isinstance(row.get("majority_capture"), str):
            row["majority_capture"] = row["majority_capture"].lower() == "true"
    return rows


def complete_blocks(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    blocks: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["topic_id"], int(row["seed"]))
        blocks.setdefault(key, {})[row["condition"]] = row
    return {
        key: value for key, value in blocks.items()
        if {"control", "likes", "single_source", "distributed_sources"} <= set(value)
    }


def sign_test_p(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = int((nonzero > 0).sum())
    tail = min(positive, n - positive)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2 * probability)


def randomization_p(diffs: np.ndarray, draws: int = 1_000_000) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    if len(nonzero) <= 20:
        values = [
            abs(float(np.mean(nonzero * np.asarray(signs))))
            for signs in itertools.product((-1, 1), repeat=len(nonzero))
        ]
        return sum(value >= observed - 1e-12 for value in values) / len(values)
    rng = np.random.default_rng(20260729)
    extreme = 0
    batch = 10_000
    for _ in range(draws // batch):
        signs = rng.choice((-1.0, 1.0), size=(batch, len(nonzero)))
        permuted = np.abs((signs * nonzero).mean(axis=1))
        extreme += int((permuted >= observed - 1e-12).sum())
    return (extreme + 1) / (draws + 1)


def bootstrap_ci(diffs: np.ndarray, draws: int = 20_000) -> tuple[float, float]:
    if len(diffs) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(20260729)
    indices = rng.integers(0, len(diffs), size=(draws, len(diffs)))
    means = diffs[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(blocks: dict[tuple[str, str, int], dict[str, dict[str, Any]]]
              ) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strata: list[tuple[str, str | None]] = [("overall", None)]
    strata += [("model", value) for value in sorted({key[0] for key in blocks})]
    strata += [("topic", value) for value in sorted({key[1] for key in blocks})]
    for contrast_name, (left, right) in CONTRASTS.items():
        for metric in METRICS:
            for stratum_type, stratum_value in strata:
                selected = []
                for key, conditions in blocks.items():
                    if stratum_type == "model" and key[0] != stratum_value:
                        continue
                    if stratum_type == "topic" and key[1] != stratum_value:
                        continue
                    selected.append(float(conditions[left][metric]) - float(conditions[right][metric]))
                if not selected:
                    continue
                diffs = np.asarray(selected, dtype=float)
                low, high = bootstrap_ci(diffs)
                output.append({
                    "contrast": contrast_name,
                    "metric": metric,
                    "stratum_type": stratum_type,
                    "stratum": stratum_value or "all",
                    "n_blocks": len(diffs),
                    "mean_difference": float(diffs.mean()),
                    "median_difference": float(np.median(diffs)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive": int((diffs > 0).sum()),
                    "negative": int((diffs < 0).sum()),
                    "ties": int((diffs == 0).sum()),
                    "sign_test_p": sign_test_p(diffs),
                    "randomization_p": randomization_p(diffs),
                })
    return output


def write_outputs(root: Path, blocks: dict, summary: list[dict[str, Any]]) -> None:
    with (root / "confirmatory_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    primary = [
        row for row in summary
        if row["stratum_type"] == "overall"
        and (
            (row["contrast"] == "pdi_distributed_minus_single"
             and row["metric"] == "delta_alignment")
            or (row["contrast"] == "engagement_likes_minus_control"
                and row["metric"] in {
                    "opposite_side_survival",
                    "final_honest_pairwise_similarity",
                })
        )
    ]
    lines = [
        "# Confirmatory analysis",
        "",
        f"- Complete paired blocks: **{len(blocks)}**",
        f"- Models represented: {len({key[0] for key in blocks})}",
        f"- Topics represented: {len({key[1] for key in blocks})}",
        f"- Seeds represented: {len({key[2] for key in blocks})}",
        "",
        "## Preregistered headline contrasts",
        "",
        "| Contrast | Outcome | n | Mean difference | 95% bootstrap CI | Sign p | Randomization p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['n_blocks']} | "
            f"{row['mean_difference']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['sign_test_p']:.4g} | {row['randomization_p']:.4g} |"
        )
    lines += [
        "",
        "All rows, including model and topic strata, are in "
        "`confirmatory_analysis.csv`. Individual posts are never treated as "
        "independent inferential observations.",
    ]
    (root / "confirmatory_analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = _read_rows(args.root / "results.csv")
    blocks = complete_blocks(rows)
    if not blocks:
        raise SystemExit("No complete four-condition blocks")
    summary = summarize(blocks)
    write_outputs(args.root, blocks, summary)
    print(json.dumps({
        "complete_blocks": len(blocks),
        "models": sorted({key[0] for key in blocks}),
        "topics": sorted({key[1] for key in blocks}),
        "seeds": sorted({key[2] for key in blocks}),
    }, indent=2))


if __name__ == "__main__":
    main()
