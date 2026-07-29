"""Post-hoc analysis: re-derive both strict and broad flip criteria from a
completed astroturf sweep's raw posts.jsonl logs.

Strict: honest agents holding EXACTLY the pushed stance bucket (1 of 7).
Broad:  honest agents on the same SIDE as the pushed stance (3 of 7 buckets).

The exact-bucket metric is preregistered. The broader side-level aggregation
was computed after inspecting the strict result and must be reported as
post-hoc.

Usage:
    python -m src.analyze_astroturf runs/<sweep-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORT_SIDE = {"strongly support", "support", "lean support"}
OPPOSE_SIDE = {"strongly oppose", "oppose", "lean oppose"}


def _write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: p.write_text(""); return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen: keys.append(k); seen.add(k)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)


def _push_side_set(pushed_opinion: str) -> set[str]:
    return SUPPORT_SIDE if pushed_opinion in SUPPORT_SIDE else \
           OPPOSE_SIDE if pushed_opinion in OPPOSE_SIDE else {pushed_opinion}


def analyze_trial(trial_dir: Path, pushed_opinion: str) -> dict[str, Any] | None:
    posts_path = trial_dir / "likes" / "logs" / "posts.jsonl"
    agents_path = trial_dir / "likes" / "logs" / "agents_initial.json"
    if not posts_path.exists() or not agents_path.exists():
        return None

    agents = json.loads(agents_path.read_text())
    coord_ids = {a["agent_id"] for a in agents if a.get("coordinated_group")}
    posts = [json.loads(line) for line in posts_path.open()]

    by_round: dict[int, list[dict[str, Any]]] = {}
    for p in posts:
        if p["agent_id"] in coord_ids: continue
        by_round.setdefault(p["round"], []).append(p)

    rounds = sorted(by_round)
    if not rounds: return None

    push_set = _push_side_set(pushed_opinion)

    strict_traj: list[float] = []
    broad_traj: list[float] = []
    rounds_to_strict_majority: int | None = None
    rounds_to_broad_majority: int | None = None

    for r in rounds:
        ops = [p["opinion"] for p in by_round[r]]
        n = max(1, len(ops))
        strict = sum(1 for o in ops if o == pushed_opinion) / n
        broad = sum(1 for o in ops if o in push_set) / n
        strict_traj.append(strict)
        broad_traj.append(broad)
        if rounds_to_strict_majority is None and strict > 0.5: rounds_to_strict_majority = r
        if rounds_to_broad_majority is None and broad > 0.5: rounds_to_broad_majority = r

    return {
        "n_honest": len(set(p["agent_id"] for p in posts) - coord_ids),
        "n_coordinated": len(coord_ids),
        # strict
        "initial_strict_share": strict_traj[0],
        "final_strict_share": strict_traj[-1],
        "delta_strict_share": strict_traj[-1] - strict_traj[0],
        "strict_flipped": rounds_to_strict_majority is not None,
        "rounds_to_strict_majority": rounds_to_strict_majority if rounds_to_strict_majority is not None else -1,
        # broad
        "initial_broad_share": broad_traj[0],
        "final_broad_share": broad_traj[-1],
        "delta_broad_share": broad_traj[-1] - broad_traj[0],
        "broad_flipped": rounds_to_broad_majority is not None,
        "rounds_to_broad_majority": rounds_to_broad_majority if rounds_to_broad_majority is not None else -1,
        # full trajectories for plotting
        "strict_trajectory": strict_traj,
        "broad_trajectory": broad_traj,
    }


def main(sweep_dir: Path) -> None:
    config = json.loads((sweep_dir / "config.json").read_text())
    pushed = config.get("astroturf_pushed_opinion", "strongly support")

    # Walk every model/seed/k trial directory
    out_rows: list[dict[str, Any]] = []
    for model_dir in sorted(sweep_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("plots", "logs"): continue
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir(): continue
            for k_dir in sorted(seed_dir.iterdir()):
                if not k_dir.is_dir() or not k_dir.name.startswith("k"): continue
                K = int(k_dir.name[1:])
                model = model_dir.name.replace("__", "/").replace("_", ":", 1)
                seed = int(seed_dir.name.replace("seed", ""))
                res = analyze_trial(k_dir, pushed)
                if res is None: continue
                row = {"model": model, "seed": seed, "K": K, **res}
                out_rows.append(row)

    if not out_rows:
        print("No trials found in", sweep_dir)
        return

    _write_csv(sweep_dir / "astroturf_broadside_results.csv", out_rows)

    # Aggregate by K across (model, seed)
    import numpy as np
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in out_rows:
        buckets.setdefault(r["K"], []).append(r)

    agg = []
    for K in sorted(buckets):
        bucket = buckets[K]
        row = {"K": K, "n_runs": len(bucket)}
        for key in ("initial_strict_share", "final_strict_share", "delta_strict_share",
                    "initial_broad_share", "final_broad_share", "delta_broad_share"):
            vals = [b[key] for b in bucket]
            row[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{key}_std"] = round(float(np.std(vals)), 4)
        row["strict_flip_rate"] = round(
            sum(1 for b in bucket if b["strict_flipped"]) / len(bucket), 3
        )
        row["broad_flip_rate"] = round(
            sum(1 for b in bucket if b["broad_flipped"]) / len(bucket), 3
        )
        agg.append(row)

    _write_csv(sweep_dir / "astroturf_broadside_aggregated.csv", agg)

    # Console summary
    print(f"\n=== Astroturf sweep: dual-criterion analysis ({sweep_dir.name}) ===")
    print(f"{'K':>4} {'n':>3} {'strict_share→':>14} {'flip%':>6} {'broad_share→':>13} {'flip%':>6} {'broad_Δ_vs_control':>20}")
    base = next((r for r in agg if r["K"] == 0), None)
    for r in agg:
        s = f"{r['initial_strict_share_mean']:.2f}→{r['final_strict_share_mean']:.2f}"
        b = f"{r['initial_broad_share_mean']:.2f}→{r['final_broad_share_mean']:.2f}"
        d_b = (r['delta_broad_share_mean']
               - (base['delta_broad_share_mean'] if base else 0))
        print(f"{r['K']:>4} {r['n_runs']:>3} {s:>14} {r['strict_flip_rate']*100:>5.0f}% "
              f"{b:>13} {r['broad_flip_rate']*100:>5.0f}% {d_b:>+20.3f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.analyze_astroturf <sweep-dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
