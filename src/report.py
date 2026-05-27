"""Generate a markdown research report from a completed run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(x: float, p: int = 3) -> str:
    try:
        return f"{x:.{p}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def write_report(run_dir: Path, cfg: dict[str, Any],
                 all_summaries: dict[str, dict[str, Any]]) -> Path:
    """Write report.md inside run_dir. Returns the path."""
    rows = []
    for cond, s in all_summaries.items():
        sc = s["scalar"]
        rows.append({
            "condition": cond,
            "majority_fraction_final": sc.get("final_majority_fraction", 0.0),
            "persona_retention_final": sc.get("final_persona_retention", 0.0),
            "pairwise_sim_final": sc.get("final_mean_pairwise_sim", 0.0),
            "minority_survival_rate": sc.get("minority_survival_rate", 0.0),
            "delta_confidence": sc.get("delta_confidence", 0.0),
            "delta_majority_fraction": sc.get("delta_majority_fraction", 0.0),
            "persona_collapse_score": sc.get("persona_collapse_score", 0.0),
        })

    # rank by collapse score descending
    ranked = sorted(rows, key=lambda r: r["persona_collapse_score"], reverse=True)
    control_row = next((r for r in rows if r["condition"] == "control"), None)

    lines: list[str] = []
    lines.append("# The Like Button Is Enough — Pilot Results\n")
    lines.append("**Research question:** Do simple social-media reward signals "
                 "(likes, visible majority, leaderboard, downvotes) cause LLM agents "
                 "to abandon persona diversity and converge into synthetic consensus?\n")
    lines.append("## Setup\n")
    lines.append(f"- Model: `{cfg.get('model')}` via Ollama at `{cfg.get('ollama_url')}`")
    lines.append(f"- Agents: {cfg.get('num_agents')}")
    lines.append(f"- Rounds: {cfg.get('num_rounds')}")
    lines.append(f"- Topic: _{cfg.get('topic')}_")
    lines.append(f"- Seed: {cfg.get('seed')}")
    lines.append(f"- Conditions tested: {', '.join(all_summaries.keys())}\n")

    lines.append("## Scalar results\n")
    lines.append("| Condition | Maj. frac (final) | Persona retention | Post sim | "
                 "Minority survival | Δ confidence | Δ majority | "
                 "**Collapse score** |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['condition']} | {_fmt(r['majority_fraction_final'])} | "
            f"{_fmt(r['persona_retention_final'])} | "
            f"{_fmt(r['pairwise_sim_final'])} | "
            f"{_fmt(r['minority_survival_rate'])} | "
            f"{_fmt(r['delta_confidence'])} | "
            f"{_fmt(r['delta_majority_fraction'])} | "
            f"**{_fmt(r['persona_collapse_score'])}** |"
        )
    lines.append("")
    lines.append("Persona-collapse score is an operational composite, defined as "
                 "`0.5·majority_fraction + 0.3·mean_pairwise_sim + 0.2·(1 - persona_retention)`. "
                 "Higher means stronger consensus, more homogeneous language, and less "
                 "persona-faithful posting.\n")

    if control_row:
        lines.append("## Comparison vs control\n")
        lines.append("| Condition | Δ collapse vs control | Δ majority vs control | "
                     "Δ persona retention vs control |")
        lines.append("|---|---|---|---|")
        for r in rows:
            if r["condition"] == "control":
                continue
            d_coll = r["persona_collapse_score"] - control_row["persona_collapse_score"]
            d_maj = r["majority_fraction_final"] - control_row["majority_fraction_final"]
            d_per = r["persona_retention_final"] - control_row["persona_retention_final"]
            lines.append(
                f"| {r['condition']} | {_fmt(d_coll, 3)} | {_fmt(d_maj, 3)} | "
                f"{_fmt(d_per, 3)} |"
            )
        lines.append("")

    lines.append("## Ranking by persona collapse\n")
    for i, r in enumerate(ranked, 1):
        lines.append(f"{i}. **{r['condition']}** — collapse score {_fmt(r['persona_collapse_score'])}, "
                     f"final majority {_fmt(r['majority_fraction_final'])}, "
                     f"minority survival {_fmt(r['minority_survival_rate'])}")
    lines.append("")

    lines.append("## Plots\n")
    for fname in [
        "persona_collapse_score.png",
        "convergence.png",
        "persona_retention.png",
        "shift_rate.png",
        "confidence.png",
        "language_similarity.png",
    ]:
        if (run_dir / "plots" / fname).exists():
            lines.append(f"![{fname}](plots/{fname})\n")

    lines.append("## Discussion (auto-template)\n")
    if control_row and ranked:
        worst = ranked[0]
        if worst["condition"] != "control":
            d = worst["persona_collapse_score"] - control_row["persona_collapse_score"]
            direction = "above" if d > 0 else "below"
            lines.append(
                f"The strongest persona-collapse signal in this pilot came from the "
                f"`{worst['condition']}` condition, {_fmt(abs(d))} {direction} the control "
                f"on the composite collapse score. The control's final majority fraction "
                f"was {_fmt(control_row['majority_fraction_final'])} vs "
                f"{_fmt(worst['majority_fraction_final'])} under `{worst['condition']}`. "
                f"Minority-opinion survival under `{worst['condition']}` was "
                f"{_fmt(worst['minority_survival_rate'])} vs "
                f"{_fmt(control_row['minority_survival_rate'])} in control."
            )
        else:
            lines.append("In this pilot, the control condition itself produced the highest "
                         "collapse score — likely a sample-size artifact at small "
                         "agent/round counts. Increase `num_agents` and `num_rounds` and rerun "
                         "before drawing conclusions.")
    lines.append("\n## Limitations\n")
    lines.append(
        "- Small pilot (default 20 agents × 5 rounds × 1 seed) — treat results as illustrative.\n"
        "- Likes/downvotes are environment-assigned by a simple majority-alignment heuristic, "
        "not by other agents voting. This makes the social signal deterministic given the post.\n"
        "- Persona retention uses TF-IDF cosine vs a short persona descriptor; swap in "
        "embeddings for a more faithful measure.\n"
        "- Single local model means we cannot yet test whether collapse generalizes across "
        "model families.\n"
    )
    lines.append("## Reproduce\n")
    lines.append("```bash\npython -m src.run_experiment --all --config configs/default.yaml\n```\n")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


def write_sweep_report(out_root: Path, cfg: dict[str, Any],
                       all_rows: list[dict[str, Any]],
                       aggregated: list[dict[str, Any]],
                       cross_model: list[dict[str, Any]]) -> Path:
    """Aggregated multi-model, multi-seed report."""
    lines: list[str] = []
    lines.append("# The Like Button Is Enough — Multi-model Sweep\n")
    lines.append("**Research question:** Do simple social-media reward signals "
                 "(likes, visible majority, leaderboard, downvotes) cause LLM agents "
                 "to abandon persona diversity and converge into synthetic consensus?\n")

    lines.append("## Setup\n")
    lines.append(f"- Models: {', '.join(f'`{m}`' for m in cfg.get('sweep_models', []))}")
    lines.append(f"- Seeds: {cfg.get('sweep_seeds', [])}")
    lines.append(f"- Agents per run: {cfg.get('num_agents')}")
    lines.append(f"- Rounds: {cfg.get('num_rounds')}")
    lines.append(f"- Topic: _{cfg.get('topic')}_")
    lines.append(f"- Conditions: {', '.join(cfg.get('sweep_conditions', []))}")
    failures = [r for r in all_rows if "_error" in r]
    if failures:
        lines.append(f"- ⚠ Failed runs: {len(failures)} (see `sweep_results.csv`)")
    lines.append("")

    # ----- Cross-model pooled view -----
    lines.append("## Pooled results (across all models × seeds)\n")
    lines.append("This is the headline view: every condition has its scalars pooled "
                 "across every (model, seed) combination. Std is across all runs.\n")
    lines.append("| Condition | n runs | Collapse score (mean ± std) | "
                 "Majority frac | Persona retention | Minority survival |")
    lines.append("|---|---|---|---|---|---|")
    control_pooled = next((r for r in cross_model if r["condition"] == "control"), None)
    for r in cross_model:
        lines.append(
            f"| {r['condition']} | {r['n_runs']} | "
            f"{_fmt(r['persona_collapse_score_mean'])} ± {_fmt(r['persona_collapse_score_std'])} | "
            f"{_fmt(r['final_majority_fraction_mean'])} ± {_fmt(r['final_majority_fraction_std'])} | "
            f"{_fmt(r['final_persona_retention_mean'])} ± {_fmt(r['final_persona_retention_std'])} | "
            f"{_fmt(r['minority_survival_rate_mean'])} ± {_fmt(r['minority_survival_rate_std'])} |"
        )
    lines.append("")

    if control_pooled:
        lines.append("### Δ vs control (pooled)\n")
        lines.append("| Condition | Δ collapse | Δ majority | Δ persona retention | "
                     "Δ minority survival |")
        lines.append("|---|---|---|---|---|")
        for r in cross_model:
            if r["condition"] == "control":
                continue
            lines.append(
                f"| {r['condition']} | "
                f"{_fmt(r['persona_collapse_score_mean'] - control_pooled['persona_collapse_score_mean'])} | "
                f"{_fmt(r['final_majority_fraction_mean'] - control_pooled['final_majority_fraction_mean'])} | "
                f"{_fmt(r['final_persona_retention_mean'] - control_pooled['final_persona_retention_mean'])} | "
                f"{_fmt(r['minority_survival_rate_mean'] - control_pooled['minority_survival_rate_mean'])} |"
            )
        lines.append("")

    # ----- Per-model breakdown -----
    lines.append("## Per-model breakdown (mean ± std across seeds)\n")
    models = sorted({r["model"] for r in aggregated})
    for model in models:
        lines.append(f"### `{model}`\n")
        rows = [r for r in aggregated if r["model"] == model]
        rows.sort(key=lambda r: ["control", "likes", "majority",
                                  "leaderboard", "downvote"].index(r["condition"])
                  if r["condition"] in ["control", "likes", "majority",
                                         "leaderboard", "downvote"] else 99)
        ctrl = next((r for r in rows if r["condition"] == "control"), None)
        lines.append("| Condition | n | Collapse | Majority frac | "
                     "Persona retention | Minority survival | Δ collapse vs control |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            d = (r["persona_collapse_score_mean"]
                 - ctrl["persona_collapse_score_mean"]) if ctrl else 0.0
            lines.append(
                f"| {r['condition']} | {r['n_seeds']} | "
                f"{_fmt(r['persona_collapse_score_mean'])} ± "
                f"{_fmt(r['persona_collapse_score_std'])} | "
                f"{_fmt(r['final_majority_fraction_mean'])} ± "
                f"{_fmt(r['final_majority_fraction_std'])} | "
                f"{_fmt(r['final_persona_retention_mean'])} ± "
                f"{_fmt(r['final_persona_retention_std'])} | "
                f"{_fmt(r['minority_survival_rate_mean'])} ± "
                f"{_fmt(r['minority_survival_rate_std'])} | "
                f"{_fmt(d)} |"
            )
        lines.append("")

    # ----- Plots -----
    lines.append("## Plots\n")
    plots_dir = out_root / "plots"
    if plots_dir.exists():
        # Pooled plots first
        for fname in sorted(p.name for p in plots_dir.glob("pooled_*.png")):
            lines.append(f"### {fname[7:-4]} (pooled)\n")
            lines.append(f"![{fname}](plots/{fname})\n")
        for fname in sorted(p.name for p in plots_dir.glob("by_model_*.png")):
            lines.append(f"### {fname[9:-4]} (per model)\n")
            lines.append(f"![{fname}](plots/{fname})\n")

    # ----- Verdict template -----
    lines.append("## Verdict (auto-template)\n")
    if control_pooled and cross_model:
        deltas = [(r["condition"],
                   r["persona_collapse_score_mean"]
                   - control_pooled["persona_collapse_score_mean"])
                  for r in cross_model if r["condition"] != "control"]
        deltas.sort(key=lambda x: x[1], reverse=True)
        consistent = all(d > 0 for _, d in deltas)
        lines.append(
            f"Across {len(models)} model family/families and "
            f"{len(set((r['model'], r['seed']) for r in all_rows if 'seed' in r))} "
            f"(model, seed) combinations:\n"
        )
        if consistent:
            lines.append(
                f"- **Every social-pressure condition increased the collapse score** "
                f"vs control. The strongest was `{deltas[0][0]}` "
                f"(Δ = {_fmt(deltas[0][1])}), the weakest was `{deltas[-1][0]}` "
                f"(Δ = {_fmt(deltas[-1][1])}). This is the direction predicted by "
                f"the paper's hypothesis.\n"
            )
        else:
            up = [c for c, d in deltas if d > 0]
            down = [c for c, d in deltas if d <= 0]
            lines.append(
                f"- Conditions above control on collapse: {up or '(none)'}. "
                f"Conditions below or equal: {down or '(none)'}. The pattern is "
                f"mixed — examine per-model breakdown for which models drive "
                f"the inconsistency.\n"
            )

    lines.append("## Limitations\n")
    lines.append(
        "- TF-IDF persona-retention is shallow; replace with an embedding model "
        "(see Ollama `/api/embeddings`) before publishing.\n"
        "- Likes/downvotes are environment-assigned heuristics, not peer-voted. "
        "Convergence pressure is therefore conservative.\n"
        "- Topic singleton: replicate on at least one more topic before claiming "
        "the result generalizes.\n"
        "- Collapse score weights (0.5/0.3/0.2) are operational. Report individual "
        "components alongside the composite in the paper.\n"
    )
    lines.append("## Reproduce\n")
    lines.append("```bash\npython -m src.run_experiment --sweep --config configs/sweep.yaml\n```\n")

    out = out_root / "report.md"
    out.write_text("\n".join(lines))
    return out
