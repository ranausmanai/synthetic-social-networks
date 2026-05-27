"""EXP-3 — Verified-badge / influence-amplification attack.

Tests whether ONE boosted agent (with N× visibility multiplier) can shift
honest-platform opinion toward its stance — distinct from EXP-2 which uses
many coordinated accounts. This is the "blue-check capture" threat model.

Setup:
  - num_baseline honest agents (default 30)
  - +1 'influencer' agent at agent_id 'inf' with a fixed stance to push
  - Sweep visibility_multiplier ∈ {1, 3, 5, 10, 20}
    - 1 = control (no boost — just one more honest agent)
    - 20 = posts appear in every honest feed every round and ride the top-K
  - For each (model, seed, multiplier), measure honest-side share trajectory
    for both strict (exact stance) and broad (support-side) criteria.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents import Agent, INITIAL_STANCES, PERSONAS, build_agents


# Broad-side aggregation for the "real" flip metric
SUPPORT_SIDE = {"strongly support", "support", "lean support"}
OPPOSE_SIDE = {"strongly oppose", "oppose", "lean oppose"}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def _write_csv(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text(""); return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen: keys.append(k); seen.add(k)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow(r)


def _make_influencer(pushed_opinion: str) -> Agent:
    """A persona that *looks* high-status — the verified-badge attack persona."""
    return Agent(
        agent_id="inf",
        persona="prominent verified industry commentator",
        style="confident, references credentials, claims insider knowledge",
        group="influencer",
        initial_opinion=pushed_opinion,
        current_opinion=pushed_opinion,
        confidence=0.88,
        posting_style="confident, references credentials, claims insider knowledge",
    )


def _analyze_run(posts_log: list[dict[str, Any]], influencer_id: str,
                 pushed_opinion: str) -> dict[str, Any]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for p in posts_log:
        by_round.setdefault(p["round"], []).append(p)
    rounds = sorted(by_round)

    # Honest only (exclude influencer)
    is_support_side = pushed_opinion in SUPPORT_SIDE
    push_side_set = SUPPORT_SIDE if is_support_side else OPPOSE_SIDE

    strict_traj: list[float] = []
    broad_traj: list[float] = []
    rounds_to_strict: int | None = None
    rounds_to_broad: int | None = None
    for r in rounds:
        honest = [p for p in by_round[r] if p["agent_id"] != influencer_id]
        n = max(1, len(honest))
        strict = sum(1 for p in honest if p["opinion"] == pushed_opinion) / n
        broad = sum(1 for p in honest if p["opinion"] in push_side_set) / n
        strict_traj.append(strict); broad_traj.append(broad)
        if rounds_to_strict is None and strict > 0.5: rounds_to_strict = r
        if rounds_to_broad is None and broad > 0.5: rounds_to_broad = r

    return {
        "final_strict_share": strict_traj[-1] if strict_traj else 0.0,
        "initial_strict_share": strict_traj[0] if strict_traj else 0.0,
        "delta_strict_share": (strict_traj[-1] - strict_traj[0]) if strict_traj else 0.0,
        "final_broad_share": broad_traj[-1] if broad_traj else 0.0,
        "initial_broad_share": broad_traj[0] if broad_traj else 0.0,
        "delta_broad_share": (broad_traj[-1] - broad_traj[0]) if broad_traj else 0.0,
        "rounds_to_strict_majority": rounds_to_strict if rounds_to_strict is not None else -1,
        "rounds_to_broad_majority": rounds_to_broad if rounds_to_broad is not None else -1,
        "strict_flipped": rounds_to_strict is not None,
        "broad_flipped": rounds_to_broad is not None,
        "strict_trajectory": strict_traj,
        "broad_trajectory": broad_traj,
    }


def _safe(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def run_influence_sweep(cfg: dict[str, Any], out_root: Path) -> Path:
    from ..run_experiment import run_one_condition

    models = cfg.get("models", [cfg.get("model")])
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    multipliers = cfg.get("influence_multipliers", [1, 3, 5, 10, 20])
    num_baseline = int(cfg.get("influence_num_baseline", 30))
    pushed_opinion = cfg.get("influence_pushed_opinion", "strongly support")
    inner_condition = cfg.get("influence_inner_condition", "likes")

    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "config.json",
                {**cfg, "exp": "influence",
                 "multipliers": list(multipliers),
                 "num_baseline": num_baseline,
                 "pushed_opinion": pushed_opinion,
                 "inner_condition": inner_condition})

    all_rows: list[dict[str, Any]] = []
    total = len(models) * len(seeds) * len(multipliers)
    done = 0
    t_start = time.time()

    for model in models:
        for seed in seeds:
            for mult in multipliers:
                done += 1
                run_label = f"{_safe(model)}/seed{seed}/mult{int(mult):03d}"
                run_dir = out_root / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                # Build baseline + add one influencer with the multiplier
                agents = build_agents(num_baseline, int(seed))
                influencer = _make_influencer(pushed_opinion)
                influencer.visibility_multiplier = int(mult)
                agents = agents + [influencer]

                # Wire multipliers into the env via cfg
                scoped_cfg = {
                    **cfg, "model": model, "seed": int(seed),
                    "num_agents": len(agents),
                    "visibility_multipliers": {
                        a.agent_id: a.visibility_multiplier for a in agents
                        if a.visibility_multiplier > 1
                    },
                }
                print(f"\n[influence {done}/{total}] model={model} seed={seed} "
                      f"mult={mult}×")
                t0 = time.time()
                try:
                    scalar = run_one_condition(
                        scoped_cfg, inner_condition, run_dir,
                        log_to_stdout=False, agents_override=agents,
                    )
                    posts_path = run_dir / inner_condition / "logs" / "posts.jsonl"
                    posts = [json.loads(line) for line in posts_path.open()]
                    res = _analyze_run(posts, "inf", pushed_opinion)
                    row = {
                        "model": model, "seed": int(seed),
                        "multiplier": int(mult),
                        **res,
                        **{f"baseline_{k}": v for k, v in scalar.items()
                           if k != "condition"},
                    }
                    print(f"   done in {time.time()-t0:.1f}s | "
                          f"broad_share={res['final_broad_share']:.3f} "
                          f"strict={res['final_strict_share']:.3f} "
                          f"broad_flipped={res['broad_flipped']}")
                except Exception as e:  # noqa: BLE001
                    print(f"   FAILED: {e}")
                    row = {"model": model, "seed": int(seed),
                           "multiplier": int(mult), "_error": str(e)}
                all_rows.append(row)
                _write_csv(out_root / "influence_results.csv", all_rows)

    print(f"\n[influence] {total} runs in {time.time()-t_start:.0f}s")
    aggregated = _aggregate_by_multiplier(all_rows)
    _write_csv(out_root / "influence_aggregated.csv", aggregated)
    _make_plots(out_root, aggregated)
    _write_report(out_root, cfg, aggregated, all_rows)
    return out_root


def _aggregate_by_multiplier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if "_error" in r: continue
        buckets.setdefault(int(r["multiplier"]), []).append(r)
    keys = ("final_strict_share", "final_broad_share",
            "delta_strict_share", "delta_broad_share",
            "rounds_to_broad_majority", "broad_flipped")
    out: list[dict[str, Any]] = []
    for m in sorted(buckets):
        bucket = buckets[m]
        row: dict[str, Any] = {"multiplier": m, "n_runs": len(bucket)}
        for key in keys:
            vals = [b[key] for b in bucket if b.get(key) is not None]
            if not vals:
                row[f"{key}_mean"] = 0.0; row[f"{key}_std"] = 0.0; continue
            if key == "rounds_to_broad_majority":
                flipped = [v for v in vals if v >= 0]
                row["broad_flip_rate"] = round(len(flipped) / len(vals), 3)
                row["mean_broad_flip_round"] = (
                    round(sum(flipped) / len(flipped), 2) if flipped else None
                )
                continue
            arr = np.asarray([float(v) for v in vals])
            row[f"{key}_mean"] = round(float(arr.mean()), 4)
            row[f"{key}_std"] = round(float(arr.std(ddof=0)), 4)
        out.append(row)
    return out


def _make_plots(out_root: Path, aggregated: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not aggregated: return
    mults = [r["multiplier"] for r in aggregated]

    plt.figure(figsize=(8, 4.5))
    s = [r["final_broad_share_mean"] for r in aggregated]
    e = [r["final_broad_share_std"] for r in aggregated]
    plt.errorbar(mults, s, yerr=e, marker="o", capsize=4, label="broad-side (3 buckets)")
    s2 = [r["final_strict_share_mean"] for r in aggregated]
    e2 = [r["final_strict_share_std"] for r in aggregated]
    plt.errorbar(mults, s2, yerr=e2, marker="s", capsize=4, label="strict (1 bucket)")
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="majority")
    plt.xlabel("Visibility multiplier of single influencer agent")
    plt.ylabel("Final honest-agent share adopting pushed opinion")
    plt.title("Influence-amplification dose response (1 boosted agent vs honest pop.)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "influence_dose_response.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    flip = [r.get("broad_flip_rate", 0.0) for r in aggregated]
    plt.bar([str(m) + "×" for m in mults], flip)
    plt.ylabel("Fraction of runs where pushed opinion reached majority (broad-side)")
    plt.title("Influencer-flip rate by multiplier")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "influence_flip_rate.png", dpi=140)
    plt.close()


def _write_report(out_root: Path, cfg: dict[str, Any],
                  aggregated: list[dict[str, Any]],
                  all_rows: list[dict[str, Any]]) -> None:
    lines = ["# EXP-3 — Influence amplification (verified-badge attack)\n"]
    lines.append("**Question:** Can ONE boosted AI account — the verified-badge "
                 "attack vector — flip platform opinion faster than dozens of "
                 "coordinated accounts (EXP-2)?\n")
    lines.append(f"- Honest baseline agents: {cfg.get('influence_num_baseline', 30)}")
    lines.append(f"- Pushed opinion: `{cfg.get('influence_pushed_opinion', 'strongly support')}`")
    lines.append(f"- Inner condition: `{cfg.get('influence_inner_condition', 'likes')}`")
    lines.append(f"- Rounds: {cfg.get('num_rounds')}")
    lines.append(f"- Models: {cfg.get('models')}")
    lines.append(f"- Seeds: {cfg.get('seeds')}\n")
    lines.append("## Results — single influencer's reach\n")
    lines.append("| Multiplier | n | Final broad-side share | Final strict share | "
                 "Broad flip rate | Mean flip round |")
    lines.append("|---|---|---|---|---|---|")
    for r in aggregated:
        bfr = r.get("broad_flip_rate", 0.0)
        mbr = r.get("mean_broad_flip_round")
        lines.append(
            f"| {r['multiplier']}× | {r['n_runs']} | "
            f"{r['final_broad_share_mean']:.3f} ± {r['final_broad_share_std']:.3f} | "
            f"{r['final_strict_share_mean']:.3f} ± {r['final_strict_share_std']:.3f} | "
            f"{bfr:.2f} | {mbr if mbr is not None else '—'} |"
        )
    lines.append("\n![dose response](plots/influence_dose_response.png)\n")
    lines.append("![flip rate](plots/influence_flip_rate.png)\n")
    lines.append("## Industry framing\n")
    lines.append(
        "This experiment tests the **'one verified account with amplified reach'** "
        "threat vector — distinct from the coordinated-account model in EXP-2. "
        "Multiplier values map to platform-realistic reach asymmetries: a 10× "
        "boost roughly corresponds to verified-account engagement on Twitter/X "
        "circa 2021–22 per public studies; 20× corresponds to top-tier "
        "blue-check or paid-promotion levels.\n\n"
        "If even one boosted AI account meaningfully shifts honest opinion, "
        "the implication for platform Trust & Safety teams is: verified-badge "
        "UX needs stronger identity verification before granting reach "
        "multipliers, not just user-pays subscription gating.\n"
    )
    (out_root / "report.md").write_text("\n".join(lines))
