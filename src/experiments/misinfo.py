"""EXP-6 Misinformation cascade speed under different interventions.

One agent is seeded with a designated false claim and posts it every round
with high confidence. We measure how quickly the false claim spreads through
the rest of the population under four interventions:

    - none: baseline (no fact-check, no demotion)
    - factcheck_label: environment appends a "[disputed: <reason>]" label
                       whenever a seeded-claim post is shown to others
    - deamplify: posts containing the claim get their visibility multiplier
                 dropped (won't appear in the top-K feeds)
    - rebuttal: every round, an environment-injected rebuttal post appears
                in everyone's feed

Note: The intervention mechanics require Environment to inspect post text against
claim keywords. The minimal wiring lives in environment.py; the
`misinfo_intervention` config flag controls behavior at feedback-rendering time.

We track:
    - rounds-to-belief-majority (claim_endorser_share > 0.5)
    - final share of agents endorsing the claim
    - cumulative confidence inflation of endorsers
"""
from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents import Agent, build_agents, make_misinformation_seed
from ..environment import CONDITIONS


INTERVENTIONS = ("none", "factcheck_label", "deamplify", "rebuttal")


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


def _post_endorses_claim(text: str, claim_keywords: list[str]) -> bool:
    """Heuristic: a post 'endorses' the claim if it contains the claim's key
    nouns/phrases positively. Tuned per-claim via the keyword list.
    """
    if not text:
        return False
    t = text.lower()
    # negation guard
    if any(neg in t for neg in [" not ", "n't ", " false", " disputed", "no truth",
                                 "debunked", "no evidence", " hoax"]):
        return False
    return all(re.search(rf"\b{re.escape(k.lower())}\b", t) for k in claim_keywords)


def _analyze_run(posts_log: list[dict[str, Any]],
                 seeder_id: str,
                 claim_keywords: list[str]) -> dict[str, Any]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for p in posts_log:
        by_round.setdefault(p["round"], []).append(p)
    rounds = sorted(by_round)

    endorser_share_by_round: list[float] = []
    rounds_to_majority: int | None = None
    for r in rounds:
        posts = by_round[r]
        non_seeder = [p for p in posts if p["agent_id"] != seeder_id]
        if not non_seeder:
            endorser_share_by_round.append(0.0); continue
        share = sum(1 for p in non_seeder
                    if _post_endorses_claim(p["text"], claim_keywords)) / len(non_seeder)
        endorser_share_by_round.append(share)
        if rounds_to_majority is None and share > 0.5:
            rounds_to_majority = r

    final = endorser_share_by_round[-1] if endorser_share_by_round else 0.0
    initial = endorser_share_by_round[0] if endorser_share_by_round else 0.0

    return {
        "final_endorser_share": final,
        "initial_endorser_share": initial,
        "delta_endorser_share": final - initial,
        "rounds_to_majority_belief": rounds_to_majority if rounds_to_majority is not None else -1,
        "majority_believed": rounds_to_majority is not None,
        "endorser_share_trajectory": endorser_share_by_round,
    }


def run_misinfo_sweep(cfg: dict[str, Any], out_root: Path) -> Path:
    """Sweep interventions × (models × seeds) for a fixed seeded claim."""
    from ..run_experiment import run_one_condition

    models = cfg.get("models", [cfg.get("model")])
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    interventions = cfg.get("misinfo_interventions", list(INTERVENTIONS))
    invalid = [i for i in interventions if i not in INTERVENTIONS]
    if invalid:
        raise ValueError(f"Unknown interventions {invalid}; valid: {INTERVENTIONS}")

    num_agents = int(cfg.get("num_agents", 30))
    seeder_id = cfg.get("misinfo_seeder_id", "a00")
    false_claim = cfg.get(
        "misinfo_claim",
        "It has been confirmed that 90% of AI-labeling laws were drafted by "
        "AI labs themselves to suppress small competitors.",
    )
    claim_keywords = cfg.get("misinfo_claim_keywords", ["90%", "ai labs", "suppress"])
    inner_condition = cfg.get("misinfo_inner_condition", "likes")

    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "config.json",
                {**cfg, "exp": "misinfo",
                 "interventions": list(interventions),
                 "false_claim": false_claim,
                 "claim_keywords": claim_keywords,
                 "seeder_id": seeder_id,
                 "inner_condition": inner_condition})

    all_results: list[dict[str, Any]] = []
    total = len(models) * len(seeds) * len(interventions)
    done = 0
    t_sweep = time.time()
    for model in models:
        for seed in seeds:
            for intervention in interventions:
                done += 1
                run_label = f"{_safe(model)}/seed{seed}/{intervention}"
                run_dir = out_root / run_label
                run_dir.mkdir(parents=True, exist_ok=True)

                agents = build_agents(num_agents, int(seed))
                make_misinformation_seed(agents, seeder_id, false_claim)

                # Intervention encoded in cfg flags read by the env
                scoped_cfg = {
                    **cfg, "model": model, "seed": int(seed),
                    "num_agents": len(agents),
                    "misinfo_intervention": intervention,
                    "misinfo_claim_keywords": claim_keywords,
                    "misinfo_false_claim": false_claim,
                }
                print(f"\n[misinfo {done}/{total}] model={model} seed={seed} "
                      f"intervention={intervention}")
                t0 = time.time()
                try:
                    scalar = run_one_condition(
                        scoped_cfg, inner_condition, run_dir,
                        log_to_stdout=False, agents_override=agents,
                    )
                    posts_path = run_dir / inner_condition / "logs" / "posts.jsonl"
                    posts = [json.loads(line) for line in posts_path.open()]
                    res = _analyze_run(posts, seeder_id, claim_keywords)
                    row = {
                        "model": model, "seed": int(seed),
                        "intervention": intervention,
                        **res,
                        **{f"baseline_{k}": v for k, v in scalar.items()
                           if k != "condition"},
                    }
                    print(f"   done in {time.time()-t0:.1f}s | "
                          f"final_endorser_share={res['final_endorser_share']:.3f} "
                          f"flip_round={res['rounds_to_majority_belief']}")
                except Exception as e:  # noqa: BLE001
                    print(f"   FAILED: {e}")
                    row = {"model": model, "seed": int(seed),
                           "intervention": intervention, "_error": str(e)}
                all_results.append(row)
                _write_csv(out_root / "misinfo_results.csv", all_results)

    print(f"\n[misinfo] {total} runs in {time.time()-t_sweep:.1f}s")

    aggregated = _aggregate_by_intervention(all_results)
    _write_csv(out_root / "misinfo_aggregated.csv", aggregated)
    _write_json(out_root / "misinfo_all.json", all_results)
    _make_plots(out_root, aggregated)
    _write_report(out_root, cfg, aggregated, false_claim)
    return out_root


def _safe(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _aggregate_by_intervention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if "_error" in r: continue
        buckets.setdefault(r["intervention"], []).append(r)
    keys = ("final_endorser_share", "delta_endorser_share",
            "rounds_to_majority_belief", "majority_believed")
    out: list[dict[str, Any]] = []
    for intervention in INTERVENTIONS:
        bucket = buckets.get(intervention, [])
        if not bucket: continue
        row: dict[str, Any] = {"intervention": intervention, "n_runs": len(bucket)}
        for key in keys:
            vals = [b[key] for b in bucket if b.get(key) is not None]
            if not vals:
                row[f"{key}_mean"] = 0.0; row[f"{key}_std"] = 0.0; continue
            if key == "rounds_to_majority_belief":
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


def _make_plots(out_root: Path, aggregated: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if not aggregated: return
    labels = [r["intervention"] for r in aggregated]

    plt.figure(figsize=(8, 4.5))
    means = [r["final_endorser_share_mean"] for r in aggregated]
    stds = [r["final_endorser_share_std"] for r in aggregated]
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel("Final share of non-seeder agents endorsing the false claim")
    plt.title("Misinformation containment by intervention")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "containment.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    flips = [r.get("flip_rate", 0.0) for r in aggregated]
    plt.bar(labels, flips)
    plt.ylabel("Fraction of runs where false claim reached majority")
    plt.title("Misinformation cascade rate by intervention")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / "cascade_rate.png", dpi=140)
    plt.close()


def _write_report(out_root: Path, cfg: dict[str, Any],
                  aggregated: list[dict[str, Any]], false_claim: str) -> None:
    lines = []
    lines.append("# EXP-6 — Misinformation cascade & interventions\n")
    lines.append("**Question:** When one seeded AI agent injects a false claim, "
                 "what intervention actually slows or stops the cascade?\n")
    lines.append(f"- Seeded false claim: _\"{false_claim}\"_")
    lines.append(f"- Honest agents: {cfg.get('num_agents')}")
    lines.append(f"- Rounds: {cfg.get('num_rounds')}")
    lines.append(f"- Models: {cfg.get('models')}")
    lines.append(f"- Seeds: {cfg.get('seeds')}\n")
    lines.append("## Results: intervention → containment\n")
    lines.append("| Intervention | n | Final endorser share | Δ vs initial | "
                 "Flip rate | Mean flip round |")
    lines.append("|---|---|---|---|---|---|")
    for r in aggregated:
        flip = r.get("flip_rate", 0.0)
        mfr = r.get("mean_flip_round_when_flipped")
        lines.append(
            f"| {r['intervention']} | {r['n_runs']} | "
            f"{r['final_endorser_share_mean']:.3f} ± "
            f"{r['final_endorser_share_std']:.3f} | "
            f"{r['delta_endorser_share_mean']:+.3f} | "
            f"{flip:.2f} | {mfr if mfr is not None else '—'} |"
        )
    lines.append("\n## Plots\n![containment](plots/containment.png)\n")
    lines.append("![cascade rate](plots/cascade_rate.png)\n")
    lines.append("## Interpretation framework\n")
    lines.append(
        "Each intervention is a real platform lever:\n"
        "- `factcheck_label` ≈ Meta/X 'disputed' labels\n"
        "- `deamplify` ≈ Twitter circa-2021 'visibility filtering'\n"
        "- `rebuttal` ≈ Community Notes / Birdwatch\n"
        "- `none` is the unmoderated baseline.\n\n"
        "The headline number is **Δ flip-rate vs none** — i.e., how many "
        "percentage points each intervention shaves off the cascade probability. "
        "That maps directly to A/B-test units that Trust & Safety teams report.\n"
    )
    (out_root / "report.md").write_text("\n".join(lines))
