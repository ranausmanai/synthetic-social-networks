"""Post-hoc thematic-similarity analysis for EXP-6 misinformation runs.

Strict keyword matching misses close restatements because agents paraphrase.
This analysis uses nomic-embed-text embeddings to measure thematic proximity
to the seeded claim. Cosine similarity is polarity-blind, so none of these
metrics measures endorsement or misinformation adoption.

Metrics per trial:
  - mean_claim_similarity (per round): cosine(post_emb, claim_emb) averaged over
    honest agents.
  - delta_claim_similarity: similarity in final round minus initial round.
    Positive delta means posts became more thematically similar, regardless of
    whether they endorsed or rebutted the claim.
  - thematic-threshold count: posts with cosine similarity above 0.55. This is
    a descriptive threshold, not an endorser count.

Cross-intervention differences are descriptive contrasts in thematic
similarity and must not be interpreted as intervention effectiveness.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import OllamaEmbedder


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


def analyze_trial(trial_dir: Path, seeded_id: str, false_claim: str,
                  embedder: OllamaEmbedder) -> dict[str, Any] | None:
    posts_path = trial_dir / "likes" / "logs" / "posts.jsonl"
    if not posts_path.exists():
        return None
    posts = [json.loads(line) for line in posts_path.open()]

    by_round: dict[int, list[dict[str, Any]]] = {}
    for p in posts:
        if p["agent_id"] == seeded_id: continue
        by_round.setdefault(p["round"], []).append(p)
    rounds = sorted(by_round)
    if not rounds: return None

    claim_emb = embedder.embed(false_claim)

    round_sim: list[float] = []
    round_max_sim: list[float] = []
    round_n_above_thresh: list[int] = []
    THRESH = 0.55  # cosine sim above which we call a post "thematically aligned"
    for r in rounds:
        sims: list[float] = []
        for p in by_round[r]:
            v = embedder.embed(p["text"])
            sims.append(float(np.dot(v, claim_emb)))
        round_sim.append(float(np.mean(sims)) if sims else 0.0)
        round_max_sim.append(float(np.max(sims)) if sims else 0.0)
        round_n_above_thresh.append(sum(1 for s in sims if s > THRESH))

    return {
        "n_honest": len(set(p["agent_id"] for p in by_round[rounds[0]])),
        "initial_mean_similarity": round_sim[0],
        "final_mean_similarity": round_sim[-1],
        "delta_mean_similarity": round_sim[-1] - round_sim[0],
        "max_round_similarity": float(max(round_sim)),
        "initial_n_thematically_aligned": round_n_above_thresh[0],
        "final_n_thematically_aligned": round_n_above_thresh[-1],
        "delta_n_thematically_aligned": round_n_above_thresh[-1] - round_n_above_thresh[0],
        "similarity_trajectory": [round(s, 4) for s in round_sim],
        "n_aligned_trajectory": round_n_above_thresh,
    }


def main(sweep_dir: Path) -> None:
    config = json.loads((sweep_dir / "config.json").read_text())
    false_claim = config.get("misinfo_claim", config.get("misinfo_false_claim", ""))
    seeded_id = config.get("misinfo_seeder_id", "a00")
    if not false_claim:
        print("ERROR: no misinfo_claim in config")
        sys.exit(1)

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        ollama_url=config.get("ollama_url", "http://localhost:11434"),
        cache_dir=str(sweep_dir.parent / ".embed_cache"),
    )

    out_rows: list[dict[str, Any]] = []
    for model_dir in sorted(sweep_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("plots", "logs"): continue
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir(): continue
            for intervention_dir in sorted(seed_dir.iterdir()):
                if not intervention_dir.is_dir(): continue
                model = model_dir.name.replace("__", "/").replace("_", ":", 1)
                seed = int(seed_dir.name.replace("seed", ""))
                intervention = intervention_dir.name
                res = analyze_trial(intervention_dir, seeded_id, false_claim, embedder)
                if res is None: continue
                row = {"model": model, "seed": seed, "intervention": intervention, **res}
                out_rows.append(row)
                print("%-12s seed=%d %-16s  initial_sim=%.3f  final_sim=%.3f  Δ=%+.3f  n_aligned=%d→%d" % (
                    model, seed, intervention,
                    res["initial_mean_similarity"], res["final_mean_similarity"],
                    res["delta_mean_similarity"],
                    res["initial_n_thematically_aligned"], res["final_n_thematically_aligned"]))

    if not out_rows:
        print("no trials found"); return

    _write_csv(sweep_dir / "misinfo_semantic_results.csv", out_rows)

    # Aggregate by intervention across (model, seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in out_rows: buckets[r["intervention"]].append(r)

    agg = []
    none_delta = None
    for intervention in ("none", "factcheck_label", "deamplify", "rebuttal"):
        if intervention not in buckets: continue
        b = buckets[intervention]
        deltas = [r["delta_mean_similarity"] for r in b]
        finals = [r["final_mean_similarity"] for r in b]
        n_aligned_final = [r["final_n_thematically_aligned"] for r in b]
        row = {
            "intervention": intervention,
            "n_runs": len(b),
            "delta_similarity_mean": round(float(np.mean(deltas)), 4),
            "delta_similarity_std": round(float(np.std(deltas)), 4),
            "final_similarity_mean": round(float(np.mean(finals)), 4),
            "final_n_aligned_mean": round(float(np.mean(n_aligned_final)), 2),
        }
        if intervention == "none":
            none_delta = row["delta_similarity_mean"]
        agg.append(row)

    # Add the same descriptive intervention-minus-none contrast used in the paper.
    for row in agg:
        if none_delta is not None:
            row["delta_similarity_vs_none"] = round(
                row["delta_similarity_mean"] - none_delta, 4
            )
    _write_csv(sweep_dir / "misinfo_semantic_aggregated.csv", agg)

    print()
    print("=== Aggregated thematic-similarity analysis ===")
    print("intervention       n   Δ_sim (mean ± std)   final_sim   n_aligned   Δ_sim vs none")
    for row in agg:
        contrast = row.get("delta_similarity_vs_none", 0.0)
        print("%-16s  %2d   %+6.3f ± %5.3f      %.3f       %4.1f         %+6.3f" % (
            row["intervention"], row["n_runs"],
            row["delta_similarity_mean"], row["delta_similarity_std"],
            row["final_similarity_mean"], row["final_n_aligned_mean"], contrast))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.analyze_misinfo <sweep-dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
