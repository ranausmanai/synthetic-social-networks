"""A1 one-shot majority-cue response diagnostic.

Setup:
  - N agents initialized with a binary stance, evenly split 50/50
  - Each agent is shown a synthetic 'feed' claiming X% of platform supports
    the opposite of their stance
  - Measure: what fraction shift toward the claimed majority after one round?

Prior studies that motivated the cue:
  - Asch (1956) line-conformity: ~37 % of human responses conform to a unanimous
    incorrect majority. (Not directly comparable — line-judgement task — but
    establishes that humans conform even when wrong.)
  - Muchnik et al. (2013), "Social influence bias in a real-time news
    aggregator," Science: a single upvote on a comment caused a ~25 % increase
    in observed positive scores at end of run. Per-round shift estimates for
    online bandwagon are typically in the 12-18 % range.
  - Salganik et al. (2006), MusicLab: showing popularity counts increased
    polarization 2-3x, with bottom-quartile songs gaining ~10-15 % share when
    artificially marked "popular".

The 10-20% range in the original implementation was an author-derived
comparison band, not a like-for-like human benchmark. It is retained in the
output schema for reproducibility but must not be interpreted as calibration.
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents import Agent, PERSONAS, ask_agent
from ..llm import LLMConfig, OllamaClient


# Historical author-derived comparison band; not a validated human benchmark.
HUMAN_BAND_LOW = 0.10
HUMAN_BAND_HIGH = 0.20


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
            if k not in seen:
                keys.append(k); seen.add(k)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_anchor_agents(n: int, seed: int,
                         topic: str,
                         stance_a: str,
                         stance_b: str) -> list[Agent]:
    """Build N agents with personas, split 50/50 between stance_a and stance_b."""
    rng = random.Random(seed)
    pool = list(PERSONAS)
    rng.shuffle(pool)
    agents: list[Agent] = []
    for i in range(n):
        persona = pool[i % len(pool)]
        stance = stance_a if i % 2 == 0 else stance_b
        agents.append(Agent(
            agent_id=f"x{i:02d}",
            persona=persona["label"],
            style=persona["style"],
            group=persona.get("group", "mainstream"),
            initial_opinion=stance,
            current_opinion=stance,
            confidence=0.65,
            posting_style=persona["style"],
        ))
    return agents


def _ask_anchor_round(client: OllamaClient, agent: Agent, topic: str,
                      bandwagon_text: str, char_limit: int = 240) -> dict[str, Any]:
    """One-shot prompt: write a post AFTER seeing a clear majority claim."""
    prompt = f"""TOPIC: {topic}

YOUR PERSONA: {agent.persona}
YOUR POSTING STYLE: {agent.style}
YOUR CURRENT POSITION: {agent.current_opinion}
YOUR CONFIDENCE (0..1): {agent.confidence}

WHAT YOU SEE IN YOUR FEED RIGHT NOW:
{bandwagon_text}

TASK:
Write ONE short post (<= {char_limit} chars) reacting to the topic in your voice.
Then state your CURRENT position (it can match your prior position or change).

Reply with strictly valid JSON:
{{
  "post": "<your post>",
  "position": "<one of: {agent.initial_opinion!r} or the opposite>",
  "confidence": <0..1>,
  "reason": "<one sentence>"
}}
"""
    from ..agents import SYSTEM_PROMPT
    out = client.generate_json(prompt, system=SYSTEM_PROMPT, temperature=0.7)
    if out.get("_parse_error"):
        return {"post": "(parse error)", "position": agent.current_opinion,
                "confidence": agent.confidence, "_parse_error": True}
    pos = str(out.get("position", agent.current_opinion))
    return {
        "post": str(out.get("post", ""))[:char_limit],
        "position": pos,
        "confidence": float(out.get("confidence", agent.confidence) or 0.0),
        "reason": str(out.get("reason", ""))[:200],
    }


def run_anchor(cfg: dict[str, Any], out_root: Path) -> Path:
    """Run the bandwagon-anchor calibration.

    Sweep: stance combination × claimed-majority strength × model × seed.
    The headline metric is `conform_rate`: fraction of initial-minority agents
    who shifted to the claimed-majority position.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    models = cfg.get("models", [cfg.get("model")])
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    n_per_run = int(cfg.get("anchor_n_agents", 30))
    bandwagon_strengths = cfg.get("anchor_bandwagon_strengths", [0.55, 0.75, 0.90])
    char_limit = int(cfg.get("post_char_limit", 240))

    # Two-stance topic. The bandwagon prompt always claims the topic-majority.
    topic = cfg.get(
        "anchor_topic",
        "Should social-media platforms make it mandatory to label AI-generated content?",
    )
    stance_a = "support mandatory labels"
    stance_b = "oppose mandatory labels"

    _write_json(out_root / "config.json",
                {**cfg, "exp": "anchor_bandwagon",
                 "n_per_run": n_per_run,
                 "bandwagon_strengths": list(bandwagon_strengths),
                 "stance_a": stance_a, "stance_b": stance_b,
                 "topic": topic,
                 "human_band": [HUMAN_BAND_LOW, HUMAN_BAND_HIGH]})

    all_rows: list[dict[str, Any]] = []
    total = len(models) * len(seeds) * len(bandwagon_strengths) * 2  # 2 = which stance is "majority"
    done = 0
    t_start = time.time()

    for model in models:
        client = OllamaClient(LLMConfig(
            model=model,
            ollama_url=cfg.get("ollama_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.8),
            max_tokens=cfg.get("max_tokens", 220),
            request_timeout=cfg.get("request_timeout", 180),
        ))
        for seed in seeds:
            for claimed_majority in bandwagon_strengths:
                for claimed_majority_stance in (stance_a, stance_b):
                    done += 1
                    print(f"\n[anchor {done}/{total}] model={model} seed={seed} "
                          f"majority={claimed_majority:.0%} pushed={claimed_majority_stance!r}")
                    t0 = time.time()
                    agents = _build_anchor_agents(n_per_run, int(seed),
                                                   topic, stance_a, stance_b)
                    pct = int(round(100 * claimed_majority))
                    bandwagon_text = (
                        f"Live platform poll (10,847 votes counted): "
                        f"**{pct}% of users say they {claimed_majority_stance}.** "
                        f"The remaining {100-pct}% disagree. This is the visible "
                        f"consensus on the platform right now."
                    )
                    results = []
                    for agent in agents:
                        try:
                            out = _ask_anchor_round(client, agent, topic,
                                                     bandwagon_text,
                                                     char_limit=char_limit)
                        except Exception as e:  # noqa: BLE001
                            out = {"post": f"(error: {e})",
                                   "position": agent.current_opinion,
                                   "confidence": agent.confidence,
                                   "_error": True}
                        new_pos = out["position"]
                        # canonicalize position
                        new_pos_low = new_pos.lower()
                        if "support" in new_pos_low and "label" in new_pos_low:
                            canonical = stance_a
                        elif "oppose" in new_pos_low and "label" in new_pos_low:
                            canonical = stance_b
                        else:
                            canonical = agent.current_opinion  # default keep
                        results.append({
                            "agent_id": agent.agent_id,
                            "persona": agent.persona,
                            "group": agent.group,
                            "initial": agent.initial_opinion,
                            "final": canonical,
                            "raw_position": new_pos,
                            "shifted_toward_majority": (
                                agent.initial_opinion != claimed_majority_stance
                                and canonical == claimed_majority_stance
                            ),
                            "post": out["post"],
                            "confidence": out["confidence"],
                        })

                    # compute conformity rate (only among minority agents)
                    minority = [r for r in results
                                if r["initial"] != claimed_majority_stance]
                    conform_rate = (
                        sum(1 for r in minority if r["shifted_toward_majority"])
                        / max(1, len(minority))
                    )
                    in_band = HUMAN_BAND_LOW <= conform_rate <= HUMAN_BAND_HIGH

                    row = {
                        "model": model, "seed": int(seed),
                        "claimed_majority": claimed_majority,
                        "claimed_majority_stance": claimed_majority_stance,
                        "n_minority": len(minority),
                        "n_shifted": sum(1 for r in minority if r["shifted_toward_majority"]),
                        "conform_rate": round(conform_rate, 4),
                        "in_human_band": in_band,
                        "human_band_low": HUMAN_BAND_LOW,
                        "human_band_high": HUMAN_BAND_HIGH,
                        "duration_s": round(time.time() - t0, 1),
                    }
                    all_rows.append(row)
                    _write_csv(out_root / "anchor_results.csv", all_rows)

                    # Per-trial detail
                    trial_label = (
                        f"{_safe(model)}/seed{seed}/maj{pct:02d}_"
                        f"{'A' if claimed_majority_stance == stance_a else 'B'}"
                    )
                    _write_json(out_root / trial_label / "per_agent.json", results)
                    print(f"   {row['n_shifted']}/{row['n_minority']} minority shifted "
                          f"→ conform_rate={conform_rate:.3f} "
                          f"({'IN BAND' if in_band else 'OUT of band'})")

    print(f"\n[anchor] {total} trials in {time.time()-t_start:.0f}s")

    _aggregate_and_report(out_root, all_rows)
    return out_root


def _safe(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _aggregate_and_report(out_root: Path, rows: list[dict[str, Any]]) -> None:
    import numpy as np
    # Aggregate by (model, claimed_majority)
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["model"], r["claimed_majority"])
        buckets.setdefault(key, []).append(float(r["conform_rate"]))

    aggregated = []
    for (model, maj), vals in sorted(buckets.items()):
        arr = np.asarray(vals)
        aggregated.append({
            "model": model,
            "claimed_majority": maj,
            "n_trials": len(vals),
            "conform_rate_mean": round(float(arr.mean()), 4),
            "conform_rate_std": round(float(arr.std(ddof=0)), 4),
            "in_human_band_mean": round(float(np.mean(
                [HUMAN_BAND_LOW <= v <= HUMAN_BAND_HIGH for v in vals])), 3),
        })
    _write_csv(out_root / "anchor_aggregated.csv", aggregated)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots = out_root / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        for model in sorted({r["model"] for r in aggregated}):
            rows_m = [r for r in aggregated if r["model"] == model]
            xs = [r["claimed_majority"] for r in rows_m]
            ys = [r["conform_rate_mean"] for r in rows_m]
            es = [r["conform_rate_std"] for r in rows_m]
            plt.errorbar(xs, ys, yerr=es, marker="o", label=model, capsize=3)
        plt.axhspan(HUMAN_BAND_LOW, HUMAN_BAND_HIGH, color="green", alpha=0.18,
                    label=f"Historical comparison band [{HUMAN_BAND_LOW:.0%}-{HUMAN_BAND_HIGH:.0%}]")
        plt.xlabel("Claimed majority fraction shown to agents")
        plt.ylabel("Per-round bandwagon shift rate (minority → majority)")
        plt.title("One-shot majority-cue response (historical comparison band)")
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "anchor_vs_human.png", dpi=140)
        plt.close()
    except Exception:  # noqa: BLE001
        pass

    # report
    lines = ["# A1 — One-shot majority-cue response diagnostic\n"]
    lines.append("**Question:** How often do initially disagreeing agents shift "
                 "after a synthetic majority cue?\n")
    lines.append(f"**Historical comparison band:** {HUMAN_BAND_LOW:.0%}–"
                 f"{HUMAN_BAND_HIGH:.0%}. This author-derived range is retained "
                 "for schema compatibility and is not a human calibration.\n")
    lines.append("## Results (per model × claimed majority)\n")
    lines.append("| Model | Claimed maj. | n trials | Conform rate (mean ± std) | In human band? |")
    lines.append("|---|---|---|---|---|")
    for r in aggregated:
        verdict = "✓ in band" if r["in_human_band_mean"] >= 0.5 else "out of band"
        lines.append(
            f"| `{r['model']}` | {r['claimed_majority']:.0%} | {r['n_trials']} | "
            f"{r['conform_rate_mean']:.3f} ± {r['conform_rate_std']:.3f} | {verdict} |"
        )
    lines.append("\n![anchor_vs_human](plots/anchor_vs_human.png)\n")
    lines.append("## Interpretation\n")
    lines.append(
        "Interpret response rates by model. The historical band is descriptive "
        "only; neither agreement nor disagreement with it establishes human "
        "fidelity or licenses lower-bound claims for other experiments.\n"
    )
    (out_root / "report.md").write_text("\n".join(lines))
