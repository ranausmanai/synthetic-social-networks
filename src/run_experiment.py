"""CLI: run one condition, or all conditions, against a config."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agents import Agent, ask_agent, build_agents
from .environment import CONDITIONS, Environment, Post
from .llm import LLMConfig, OllamaClient
from .metrics import summarize


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_run_dir(base: str, label: str) -> Path:
    p = Path(base) / label
    (p / "logs").mkdir(parents=True, exist_ok=True)
    (p / "metrics").mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_one_condition(cfg: dict[str, Any], condition: str,
                      run_dir: Path, log_to_stdout: bool = True,
                      agents_override: list[Agent] | None = None
                      ) -> dict[str, Any]:
    """Run a single condition; write logs + metrics; return scalar summary.

    Optional kwarg `agents_override` is used by experiment runners (EXP-2/6/3)
    that need to inject coordinated, misinformation-seeded, or boosted agents
    instead of the default `build_agents()` population.
    """
    import random as _random

    client = OllamaClient(LLMConfig(
        model=cfg["model"],
        ollama_url=cfg.get("ollama_url", "http://localhost:11434"),
        temperature=cfg.get("temperature", 0.8),
        max_tokens=cfg.get("max_tokens", 220),
        request_timeout=cfg.get("request_timeout", 120),
    ))

    if agents_override is not None:
        agents = agents_override
    else:
        agents = build_agents(cfg["num_agents"], cfg["seed"])

    initial_opinions = {a.agent_id: a.initial_opinion for a in agents}
    agent_personas = {a.agent_id: f"{a.persona}: {a.style}" for a in agents}
    agent_groups = {a.agent_id: a.group for a in agents}

    use_peer_voting = bool(cfg.get("use_peer_voting", False))
    voting_k = int(cfg.get("voting_k_per_voter", 5))

    env = Environment(
        condition=condition,
        seed=cfg["seed"],
        likes_visibility_top_k=cfg.get("likes_visibility_top_k", 5),
        downvote_threshold=cfg.get("downvote_threshold", 0.3),
        use_peer_voting=use_peer_voting,
        misinfo_intervention=cfg.get("misinfo_intervention", "none"),
        misinfo_claim_keywords=cfg.get("misinfo_claim_keywords"),
        misinfo_rebuttal=cfg.get(
            "misinfo_rebuttal",
            "Independent fact-checks find no evidence for the disputed claim above."
        ),
        visibility_multipliers=cfg.get("visibility_multipliers"),
    )

    # Embedder is optional; only used by metrics if cfg.embed_model is set.
    embedder = None
    embed_model = cfg.get("embed_model")
    if embed_model:
        from .embeddings import OllamaEmbedder
        cache_dir = cfg.get("embed_cache_dir", str(run_dir.parent / ".embed_cache"))
        embedder = OllamaEmbedder(
            model=embed_model,
            ollama_url=cfg.get("ollama_url", "http://localhost:11434"),
            cache_dir=cache_dir,
        )

    post_log: list[dict[str, Any]] = []
    vote_log: list[dict[str, Any]] = []
    cond_dir = run_dir / condition
    (cond_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cond_dir / "metrics").mkdir(parents=True, exist_ok=True)

    num_rounds = int(cfg["num_rounds"])
    char_limit = int(cfg.get("post_char_limit", 280))
    topic = cfg["topic"]

    _write_json(cond_dir / "logs" / "agents_initial.json",
                [a.to_dict() for a in agents])

    rng_vote = _random.Random(int(cfg["seed"]) + 9001)

    for r in range(num_rounds):
        round_posts: list[Post] = []
        t_round = time.time()
        for agent in agents:
            feedback = env.render_feedback_for_agent(agent, r)
            t0 = time.time()
            try:
                out = ask_agent(client, agent, topic, feedback, char_limit)
            except Exception as e:  # noqa: BLE001
                if log_to_stdout:
                    print(f"  [warn] agent {agent.agent_id} round {r} failed: {e}",
                          file=sys.stderr)
                out = {
                    "post": "(generation failed)",
                    "opinion": agent.current_opinion,
                    "confidence": agent.confidence,
                    "reason": f"error:{e}",
                }
            dt = time.time() - t0
            agent.current_opinion = out["opinion"]
            agent.confidence = float(out["confidence"])
            agent.memory.append({
                "round": r,
                "post": out["post"],
                "opinion": out["opinion"],
                "confidence": out["confidence"],
                "feedback_seen": feedback,
            })
            round_posts.append(Post(
                round=r,
                agent_id=agent.agent_id,
                persona=agent.persona,
                text=out["post"],
                opinion=out["opinion"],
                confidence=float(out["confidence"]),
            ))
            if log_to_stdout:
                print(f"  [{condition} r{r}] {agent.agent_id} "
                      f"op={out['opinion']:>16} conf={out['confidence']:.2f} "
                      f"({dt:.1f}s)")

        # Peer voting OR heuristic feedback
        if use_peer_voting:
            from .voting import collect_peer_votes
            post_dicts = [p.to_dict() for p in round_posts]
            t_vote = time.time()
            votes_raw = collect_peer_votes(
                client, agents, post_dicts, topic,
                k_per_voter=voting_k, rng=rng_vote,
            )
            # Copy aggregated likes/downvotes back onto Post objects
            for i, p in enumerate(round_posts):
                p.likes = post_dicts[i].get("likes", 0)
                p.downvotes = post_dicts[i].get("downvotes", 0)
            # Persist the vote log for the round
            vote_log.append({
                "round": r,
                "votes": votes_raw,
                "vote_seconds": time.time() - t_vote,
            })
            if log_to_stdout:
                print(f"  [{condition}] peer voting r{r} "
                      f"in {time.time()-t_vote:.1f}s")
        else:
            env.assign_feedback(round_posts)

        env.record_round(round_posts)
        for p in round_posts:
            post_log.append(p.to_dict())
        if log_to_stdout:
            print(f"  [{condition}] round {r} done in {time.time()-t_round:.1f}s")

    _write_jsonl(cond_dir / "logs" / "posts.jsonl", post_log)
    _write_json(cond_dir / "logs" / "agents_final.json",
                [a.to_dict() for a in agents])
    if use_peer_voting:
        _write_json(cond_dir / "logs" / "votes.json", vote_log)

    summary = summarize(post_log, initial_opinions, agent_personas,
                        agent_groups=agent_groups, embedder=embedder)
    _write_json(cond_dir / "metrics" / "summary.json", summary)
    for name, rows in summary["per_round"].items():
        _write_csv(cond_dir / "metrics" / f"{name}.csv", rows)
    _write_json(cond_dir / "metrics" / "scalar.json", summary["scalar"])
    return {"condition": condition, **summary["scalar"]}


def _make_plots(run_dir: Path, all_summaries: dict[str, dict[str, Any]]) -> None:
    """Per-condition trajectories + cross-condition bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-metric line plots across conditions
    metric_specs = [
        ("convergence", "majority_fraction", "Majority fraction"),
        ("persona_retention", "persona_retention", "Persona retention (TF-IDF)"),
        ("shift_rate", "shift_rate", "Opinion shift rate vs initial"),
        ("confidence", "mean_confidence", "Mean confidence"),
        ("language_similarity", "mean_pairwise_sim", "Mean pairwise post similarity"),
    ]
    for fname, ykey, ylabel in metric_specs:
        plt.figure(figsize=(7, 4.2))
        plotted = False
        for cond, s in all_summaries.items():
            rows = s["per_round"].get(fname, [])
            if not rows:
                continue
            xs = [row["round"] for row in rows]
            ys = [row.get(ykey, 0.0) for row in rows]
            plt.plot(xs, ys, marker="o", label=cond)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.title(ylabel + " over rounds, by condition")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{fname}.png", dpi=140)
        plt.close()

    # Cross-condition bar chart for persona-collapse score
    plt.figure(figsize=(7, 4.2))
    conds = list(all_summaries.keys())
    scores = [all_summaries[c]["scalar"]["persona_collapse_score"] for c in conds]
    plt.bar(conds, scores)
    plt.ylabel("Persona collapse score (higher = worse)")
    plt.title("Persona collapse score by condition")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "persona_collapse_score.png", dpi=140)
    plt.close()


def run_all(config_path: str, conditions: list[str] | None = None,
            tag: str | None = None) -> Path:
    cfg = load_config(config_path)
    if conditions:
        run_conditions = conditions
    else:
        run_conditions = cfg.get("conditions", list(CONDITIONS))
    invalid = [c for c in run_conditions if c not in CONDITIONS]
    if invalid:
        raise ValueError(f"Unknown conditions: {invalid}. Valid: {CONDITIONS}")

    run_label = _ts() + (f"_{tag}" if tag else "")
    run_dir = _ensure_run_dir(cfg.get("output_dir", "runs"), run_label)
    _write_json(run_dir / "config.json", cfg)
    print(f"[runner] writing to {run_dir}")

    all_summaries: dict[str, dict[str, Any]] = {}
    scalar_rows: list[dict[str, Any]] = []
    for cond in run_conditions:
        print(f"\n=== Condition: {cond} ===")
        t0 = time.time()
        scalar = run_one_condition(cfg, cond, run_dir)
        scalar_rows.append(scalar)
        with (run_dir / cond / "metrics" / "summary.json").open() as f:
            all_summaries[cond] = json.load(f)
        print(f"[{cond}] done in {time.time()-t0:.1f}s")

    _write_json(run_dir / "all_summaries.json", all_summaries)
    _write_csv(run_dir / "scalar_metrics.csv", scalar_rows)
    _make_plots(run_dir, all_summaries)

    # Auto-write the report
    from .report import write_report
    write_report(run_dir, cfg, all_summaries)

    print(f"\n[runner] complete. See {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--condition",
        help=f"Run only one condition. One of: {', '.join(CONDITIONS)}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all conditions listed in the config.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the full (model × seed × condition) sweep "
                             "using `models:` and `seeds:` lists from the config.")
    parser.add_argument("--exp", default=None,
                        choices=["astroturf", "misinfo", "anchor", "human_replay",
                                  "influence"],
                        help="Run an industry-relevant experiment: "
                             "astroturf (EXP-2), misinfo (EXP-6), "
                             "influence (EXP-3 verified-badge amplification), "
                             "anchor (A1 bandwagon calibration), "
                             "or human_replay (A2 ChangeMyView calibration).")
    parser.add_argument("--models", default=None,
                        help="Override `models:` from config. Comma-separated.")
    parser.add_argument("--seeds", default=None,
                        help="Override `seeds:` from config. Comma-separated ints.")
    parser.add_argument("--tag", default=None, help="Optional run label suffix.")
    args = parser.parse_args()

    if args.exp:
        cfg = load_config(args.config)
        if args.models:
            cfg["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.seeds:
            cfg["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
        run_label = _ts() + f"_{args.exp}" + (f"_{args.tag}" if args.tag else "")
        out_root = Path(cfg.get("output_dir", "runs")) / run_label
        if args.exp == "astroturf":
            from .experiments.astroturf import run_astroturf_sweep
            run_astroturf_sweep(cfg, out_root)
        elif args.exp == "misinfo":
            from .experiments.misinfo import run_misinfo_sweep
            run_misinfo_sweep(cfg, out_root)
        elif args.exp == "anchor":
            from .experiments.anchor import run_anchor
            run_anchor(cfg, out_root)
        elif args.exp == "human_replay":
            from .experiments.human_replay import run_human_replay
            run_human_replay(cfg, out_root)
        elif args.exp == "influence":
            from .experiments.influence import run_influence_sweep
            run_influence_sweep(cfg, out_root)
        return

    if args.sweep and (args.all or args.condition):
        parser.error("Use --sweep alone, or --all / --condition (not both).")
    if not args.sweep and not args.all and not args.condition:
        parser.error("Pass --sweep, --all, --condition <name>, or --exp <name>.")

    if args.sweep:
        from .sweep import run_sweep
        cfg = load_config(args.config)
        models = (
            [m.strip() for m in args.models.split(",") if m.strip()]
            if args.models else cfg.get("models", [cfg.get("model")])
        )
        seeds = (
            [int(s) for s in args.seeds.split(",") if s.strip()]
            if args.seeds else cfg.get("seeds", [cfg.get("seed", 42)])
        )
        run_sweep(cfg, models=models, seeds=seeds,
                  conditions=cfg.get("conditions"), tag=args.tag)
        return

    conditions = None if args.all else [args.condition]
    run_all(args.config, conditions=conditions, tag=args.tag)


if __name__ == "__main__":
    main()
