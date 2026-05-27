"""Sweep across (model × seed × condition). Aggregates with mean ± std."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .environment import CONDITIONS


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_model_dirname(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def run_sweep(cfg: dict[str, Any], models: list[str], seeds: list[int],
              conditions: list[str] | None = None, tag: str | None = None) -> Path:
    """Run the full grid of (model × seed × condition).

    Output layout:
      runs/<ts>_sweep[_tag]/
        config.json
        sweep_results.csv      # one row per (model, seed, condition)
        aggregated.csv         # one row per (model, condition) with mean/std across seeds
        cross_model.csv        # one row per condition, mean across (models × seeds)
        report.md              # aggregated report
        plots/                 # cross-condition error-bar plots
        <model_dir>/<seed>/<condition>/{logs,metrics}/
    """
    from .run_experiment import run_one_condition

    conds = conditions or cfg.get("conditions", list(CONDITIONS))
    invalid = [c for c in conds if c not in CONDITIONS]
    if invalid:
        raise ValueError(f"Unknown conditions: {invalid}. Valid: {CONDITIONS}")

    run_label = _ts() + "_sweep" + (f"_{tag}" if tag else "")
    out_root = Path(cfg.get("output_dir", "runs")) / run_label
    out_root.mkdir(parents=True, exist_ok=True)

    snapshot = {**cfg, "sweep_models": models, "sweep_seeds": seeds,
                "sweep_conditions": conds}
    _write_json(out_root / "config.json", snapshot)

    total_runs = len(models) * len(seeds) * len(conds)
    print(f"[sweep] {len(models)} models × {len(seeds)} seeds × {len(conds)} "
          f"conditions = {total_runs} condition-runs → {out_root}")
    t_sweep_start = time.time()

    all_rows: list[dict[str, Any]] = []
    done = 0
    for model in models:
        model_dir_name = _safe_model_dirname(model)
        for seed in seeds:
            # Use a model+seed-scoped sub-runs dir so each combo gets its own
            # condition tree under <out_root>/<model>/<seed>/<condition>/...
            scoped_root = out_root / model_dir_name / f"seed{seed}"
            scoped_root.mkdir(parents=True, exist_ok=True)
            scoped_cfg = {**cfg, "model": model, "seed": int(seed)}
            for cond in conds:
                done += 1
                print(f"\n[sweep {done}/{total_runs}] model={model} seed={seed} "
                      f"condition={cond}")
                t0 = time.time()
                try:
                    # run_one_condition writes into scoped_root/<cond>/...
                    scalar = run_one_condition(scoped_cfg, cond, scoped_root,
                                               log_to_stdout=False)
                    print(f"   done in {time.time()-t0:.1f}s | "
                          f"collapse={scalar.get('persona_collapse_score', 0):.3f} "
                          f"maj={scalar.get('final_majority_fraction', 0):.2f} "
                          f"persona={scalar.get('final_persona_retention', 0):.3f}")
                except Exception as e:  # noqa: BLE001
                    print(f"   FAILED: {e}")
                    scalar = {"condition": cond, "_error": str(e)}
                row = {"model": model, "seed": int(seed), **scalar}
                all_rows.append(row)
                # checkpoint after every combo so partial sweeps are useful
                _write_csv(out_root / "sweep_results.csv", all_rows)

    print(f"\n[sweep] {total_runs} runs done in {time.time()-t_sweep_start:.1f}s")

    aggregated = _aggregate_by_model_condition(all_rows)
    cross_model = _aggregate_by_condition(all_rows)
    _write_csv(out_root / "aggregated.csv", aggregated)
    _write_csv(out_root / "cross_model.csv", cross_model)
    _write_json(out_root / "all_rows.json", all_rows)

    _make_sweep_plots(out_root, aggregated, cross_model, conds, models)

    from .report import write_sweep_report
    write_sweep_report(out_root, snapshot, all_rows, aggregated, cross_model)

    print(f"[sweep] complete. See {out_root}")
    return out_root


# ---- aggregation helpers ----------------------------------------------------

_METRIC_KEYS = (
    "persona_collapse_score",
    "final_majority_fraction",
    "final_persona_retention",
    "final_mean_pairwise_sim",
    "minority_survival_rate",
    "delta_confidence",
    "delta_majority_fraction",
)


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if v is not None and not _is_nan(v)],
                     dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # noqa: PLR0124  (NaN check)
    except Exception:  # noqa: BLE001
        return False


def _aggregate_by_model_condition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if "_error" in r:
            continue
        key = (r["model"], r["condition"])
        buckets.setdefault(key, []).append(r)
    for (model, condition), bucket in sorted(buckets.items()):
        row: dict[str, Any] = {"model": model, "condition": condition,
                               "n_seeds": len(bucket)}
        for k in _METRIC_KEYS:
            vals = [b.get(k, 0.0) for b in bucket]
            mean, std, _ = _agg(vals)
            row[f"{k}_mean"] = round(mean, 4)
            row[f"{k}_std"] = round(std, 4)
        out.append(row)
    return out


def _aggregate_by_condition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool across all (model × seed) for each condition — the cross-model view."""
    out: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if "_error" in r:
            continue
        buckets.setdefault(r["condition"], []).append(r)
    # Preserve canonical order
    for condition in CONDITIONS:
        bucket = buckets.get(condition, [])
        if not bucket:
            continue
        row: dict[str, Any] = {"condition": condition, "n_runs": len(bucket)}
        for k in _METRIC_KEYS:
            vals = [b.get(k, 0.0) for b in bucket]
            mean, std, _ = _agg(vals)
            row[f"{k}_mean"] = round(mean, 4)
            row[f"{k}_std"] = round(std, 4)
        out.append(row)
    return out


# ---- plotting ---------------------------------------------------------------

def _make_sweep_plots(out_root: Path, aggregated: list[dict[str, Any]],
                      cross_model: list[dict[str, Any]],
                      conditions: list[str], models: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated:
        return

    metrics_to_plot = [
        ("persona_collapse_score", "Persona collapse score (higher = more collapse)"),
        ("final_majority_fraction", "Final majority fraction"),
        ("final_persona_retention", "Persona retention (TF-IDF cosine)"),
        ("minority_survival_rate", "Minority view survival rate"),
        ("final_mean_pairwise_sim", "Final mean pairwise post similarity"),
    ]

    # Grouped bar with error bars: x=condition, group=model
    x = np.arange(len(conditions))
    width = max(0.12, 0.8 / max(1, len(models)))

    for key, ylabel in metrics_to_plot:
        plt.figure(figsize=(9, 4.8))
        for i, model in enumerate(models):
            means = []
            stds = []
            for cond in conditions:
                row = next((r for r in aggregated
                           if r["model"] == model and r["condition"] == cond), None)
                means.append(row[f"{key}_mean"] if row else 0.0)
                stds.append(row[f"{key}_std"] if row else 0.0)
            offset = (i - (len(models) - 1) / 2) * width
            plt.bar(x + offset, means, width=width, yerr=stds, capsize=3,
                    label=model)
        plt.xticks(x, conditions, rotation=15)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by condition × model")
        plt.legend(fontsize=8)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"by_model_{key}.png", dpi=140)
        plt.close()

    # Cross-model pooled view — one bar per condition with error bars across all
    # (model × seed) runs
    if cross_model:
        for key, ylabel in metrics_to_plot:
            plt.figure(figsize=(7.5, 4.5))
            conds = [r["condition"] for r in cross_model]
            means = [r[f"{key}_mean"] for r in cross_model]
            stds = [r[f"{key}_std"] for r in cross_model]
            plt.bar(conds, means, yerr=stds, capsize=4)
            plt.ylabel(ylabel)
            plt.title(f"{ylabel}: pooled across models & seeds")
            plt.xticks(rotation=15)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"pooled_{key}.png", dpi=140)
            plt.close()
