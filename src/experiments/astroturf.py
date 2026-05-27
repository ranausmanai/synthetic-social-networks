"""EXP-2 Astroturfing dose-response.

How few coordinated AI agents does it take to manufacture platform consensus?

Sweep K coordinated agents ∈ k_grid across a population of `num_baseline` honest
agents. All coordinated agents push the same opinion and the same scripted
angle. Measures:
  - final share of agents holding the pushed opinion (overall + among honest-only)
  - rounds-to-majority-flip (first round where pushed_opinion > 50% of honest agents)
  - confidence inflation of honest agents over time
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents import Agent, build_agents, inject_coordinated_agents
from ..environment import CONDITIONS


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def _write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _analyze_run(posts_log: list[dict[str, Any]], pushed_opinion: str,
                 coordinated_ids: set[str]) -> dict[str, Any]:
    """Compute astroturf-specific scalars from a posts log."""
    by_round: dict[int, list[dict[str, Any]]] = {}
    for p in posts_log:
        by_round.setdefault(p["round"], []).append(p)

    rounds = sorted(by_round)
    if not rounds:
        return {}

    honest_pushed_share_by_round: list[float] = []
    overall_pushed_share_by_round: list[float] = []
    rounds_to_flip: int | None = None

    for r in rounds:
        posts = by_round[r]
        honest = [p for p in posts if p["agent_id"] not in coordinated_ids]
        overall_share = sum(1 for p in posts if p["opinion"] == pushed_opinion) / len(posts)
        honest_share = (
            sum(1 for p in honest if p["opinion"] == pushed_opinion) / len(honest)
            if honest else 0.0
        )
        overall_pushed_share_by_round.append(overall_share)
        honest_pushed_share_by_round.append(honest_share)
        if rounds_to_flip is None and honest_share > 0.5:
            rounds_to_flip = r

    final_round = rounds[-1]
    final_posts = by_round[final_round]
    final_honest = [p for p in final_posts if p["agent_id"] not in coordinated_ids]
    initial_round = rounds[0]
    initial_honest = [p for p in by_round[initial_round]
                      if p["agent_id"] not in coordinated_ids]

    return {
        "final_overall_pushed_share": overall_pushed_share_by_round[-1],
        "final_honest_pushed_share": honest_pushed_share_by_round[-1],
        "initial_honest_pushed_share": honest_pushed_share_by_round[0],
        "delta_honest_pushed_share": (
            honest_pushed_share_by_round[-1] - honest_pushed_share_by_round[0]
        ),
        "rounds_to_majority_flip": rounds_to_flip if rounds_to_flip is not None else -1,
        "majority_flipped": rounds_to_flip is not None,
        "honest_pushed_share_trajectory": honest_pushed_share_by_round,
        "n_honest": len(final_honest),
        "n_coordinated": len(coordinated_ids),
        "mean_confidence_honest_final": (
            sum(float(p["confidence"]) for p in final_honest) / max(1, len(final_honest))
        ),
        "mean_confidence_honest_initial": (
            sum(float(p["confidence"]) for p in initial_honest) / max(1, len(initial_honest))
        ),
    }


def run_astroturf_sweep(cfg: dict[str, Any], out_root: Path) -> Path:
    """Sweep K coordinated agents over k_grid, models × seeds."""
    from ..run_experiment import run_one_condition

    k_grid = cfg.get("astroturf_k_grid", [0, 1, 3, 5, 10, 20])
    models = cfg.get("models", [cfg.get("model")])
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    num_baseline = int(cfg.get("astroturf_num_baseline", 50))
    pushed_opinion = cfg.get("astroturf_pushed_opinion", "strongly support")
    angle = cfg.get(
        "astroturf_angle",
        "AI-generated content labels should be mandatory and automatic — most people quietly agree.",
    )
    # Sub-condition: we always use 'likes' visibility for astroturf since that's
    # the realistic platform setting. (Could sweep later.)
    inner_condition = cfg.get("astroturf_inner_condition", "likes")
    if inner_condition not in CONDITIONS:
        raise ValueError(f"inner_condition must be one of {CONDITIONS}")

    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "config.json",
                {**cfg, "exp": "astroturf",
                 "k_grid": list(k_grid),
                 "num_baseline": num_baseline,
                 "pushed_opinion": pushed_opinion,
                 "inner_condition": inner_condition})

    all_results: list[dict[str, Any]] = []
    total = len(models) * len(seeds) * len(k_grid)
    done = 0
    t_sweep = time.time()
    for model in models:
        for seed in seeds:
            for k in k_grid:
                done += 1
                run_label = f"{_safe(model)}/seed{seed}/k{k:03d}"
                run_dir = out_root / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                # Build baseline agents, then inject K coordinated agents
                agents = build_agents(num_baseline, int(seed))
                agents = inject_coordinated_agents(
                    agents, n=int(k),
                    pushed_opinion=pushed_opinion,
                    angle=angle,
                )
                coordinated_ids = {a.agent_id for a in agents
                                   if a.coordinated_group is not None}

                scoped_cfg = {**cfg, "model": model, "seed": int(seed),
                              "num_agents": len(agents)}
                print(f"\n[astroturf {done}/{total}] model={model} seed={seed} K={k} "
                      f"(honest={num_baseline}, coordinated={k})")
                t0 = time.time()
                try:
                    scalar = run_one_condition(
                        scoped_cfg, inner_condition, run_dir,
                        log_to_stdout=False, agents_override=agents,
                    )
                    posts_path = run_dir / inner_condition / "logs" / "posts.jsonl"
                    posts = [json.loads(line) for line in posts_path.open()]
                    astro = _analyze_run(posts, pushed_opinion, coordinated_ids)
                    row = {
                        "model": model, "seed": int(seed), "K": int(k),
                        "inner_condition": inner_condition,
                        **astro,
                        **{f"baseline_{k_}": v for k_, v in scalar.items()
                           if k_ != "condition"},
                    }
                    print(f"   done in {time.time()-t0:.1f}s | "
                          f"honest_pushed_share={astro['final_honest_pushed_share']:.3f} "
                          f"flip_round={astro['rounds_to_majority_flip']}")
                except Exception as e:  # noqa: BLE001
                    print(f"   FAILED: {e}")
                    row = {"model": model, "seed": int(seed), "K": int(k),
                           "_error": str(e)}
                all_results.append(row)
                _write_csv(out_root / "astroturf_results.csv", all_results)

    print(f"\n[astroturf] {total} runs in {time.time()-t_sweep:.1f}s")

    # Aggregate: per K mean ± std across (model, seed)
    aggregated = _aggregate_by_k(all_results)
    _write_csv(out_root / "astroturf_aggregated.csv", aggregated)
    _write_json(out_root / "astroturf_all.json", all_results)

    _make_plots(out_root, aggregated, k_grid)
    _write_report(out_root, cfg, aggregated, all_results)
    return out_root


def _safe(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _aggregate_by_k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if "_error" in r:
            continue
        buckets.setdefault(int(r["K"]), []).append(r)
    keys = ("final_honest_pushed_share", "final_overall_pushed_share",
            "delta_honest_pushed_share", "rounds_to_majority_flip",
            "majority_flipped", "mean_confidence_honest_final")
    out: list[dict[str, Any]] = []
    for k in sorted(buckets):
        bucket = buckets[k]
        row: dict[str, Any] = {"K": k, "n_runs": len(bucket)}
        for key in keys:
            vals = [b[key] for b in bucket if b.get(key) not in (None, "")]
            if not vals:
                row[f"{key}_mean"] = 0.0
                row[f"{key}_std"] = 0.0
                continue
            # rounds_to_majority_flip uses -1 for "didn't flip" — keep but report flip-rate separately
            if key == "rounds_to_majority_flip":
                flipped = [v for v in vals if v >= 0]
                row["flip_rate"] = round(len(flipped) / len(vals), 3)
                row["mean_flip_round_when_flipped"] = (
                    round(sum(flipped) / len(flipped), 2) if flipped else None
                )
                continue
            arr = np.asarray([float(v) for v in vals])
            row[f"{key}_mean"] = round(float(arr.mean()), 4)
            row[f"{key}_std"] = round(float(arr.std(ddof=0)), 4)
        out.append(row)
    return out


def _make_plots(out_root: Path, aggregated: list[dict[str, Any]],
                k_grid: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not aggregated:
        return
    ks = [r["K"] for r in aggregated]

    plt.figure(figsize=(8, 4.5))
    means = [r["final_honest_pushed_share_mean"] for r in aggregated]
    stds = [r["final_honest_pushed_share_std"] for r in aggregated]
    plt.errorbar(ks, means, yerr=stds, marker="o", capsize=4)
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="majority threshold")
    plt.xlabel("K (# coordinated agents injected)")
    plt.ylabel("Honest agents adopting pushed opinion (final)")
    plt.title("Astroturfing dose-response: K vs honest-agent capture")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "dose_response.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    flip_rates = [r.get("flip_rate", 0.0) for r in aggregated]
    plt.bar(ks, flip_rates)
    plt.xlabel("K (# coordinated agents)")
    plt.ylabel("Fraction of (model, seed) runs where honest-majority flipped")
    plt.title("Flip rate by K")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "flip_rate.png", dpi=140)
    plt.close()


def _write_report(out_root: Path, cfg: dict[str, Any],
                  aggregated: list[dict[str, Any]],
                  all_results: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# EXP-2 — Astroturfing dose-response\n")
    lines.append("**Question:** How few coordinated AI agents does it take to "
                 "manufacture platform consensus?\n")
    lines.append(f"- Honest baseline agents: {cfg.get('astroturf_num_baseline', 50)}")
    lines.append(f"- Pushed opinion: `{cfg.get('astroturf_pushed_opinion')}`")
    lines.append(f"- Inner condition: `{cfg.get('astroturf_inner_condition', 'likes')}`")
    lines.append(f"- Rounds: {cfg.get('num_rounds')}")
    lines.append(f"- Models: {cfg.get('models')}")
    lines.append(f"- Seeds: {cfg.get('seeds')}\n")
    lines.append("## Results: K → honest-agent capture\n")
    lines.append("| K | n | Honest pushed share (final) | Δ vs initial | "
                 "Flip rate | Mean flip round |")
    lines.append("|---|---|---|---|---|---|")
    for r in aggregated:
        flip = r.get("flip_rate", 0.0)
        mfr = r.get("mean_flip_round_when_flipped")
        lines.append(
            f"| {r['K']} | {r['n_runs']} | "
            f"{r['final_honest_pushed_share_mean']:.3f} ± "
            f"{r['final_honest_pushed_share_std']:.3f} | "
            f"{r['delta_honest_pushed_share_mean']:+.3f} | "
            f"{flip:.2f} | {mfr if mfr is not None else '—'} |"
        )
    lines.append("\n## Plots\n")
    lines.append("![dose response](plots/dose_response.png)\n")
    lines.append("![flip rate](plots/flip_rate.png)\n")
    lines.append("## Interpretation framework\n")
    lines.append(
        "Map K to platform-realistic ratios. K/(honest+K) is the AI-account "
        "share. For a 1000-account platform this translates as: K=10 → 1%, "
        "K=30 → ~3%, K=100 → ~9%. A finding like *'K=10 (≈1% of accounts) "
        "is sufficient to flip honest-majority opinion in ≥50% of runs'* "
        "lands directly in Meta's Coordinated Inauthentic Behavior framework "
        "and is the kind of single number that integrity-team PMs cite to "
        "leadership.\n"
    )
    (out_root / "report.md").write_text("\n".join(lines))
