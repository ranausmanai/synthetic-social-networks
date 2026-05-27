"""ANCHOR A2 — ChangeMyView human-replay calibration.

For each conversation from r/ChangeMyView:
  - OP posts a stance with reasoning
  - Top commenters argue against
  - Sometimes OP awards a "delta" (∆) indicating they updated their view

We replay this in our sim:
  1. Initialize a sim agent with the OP's stance and a persona inferred from
     their writing
  2. Show the sim agent the top counter-arguments
  3. Ask: did your view change?
  4. Compare to the real human outcome (delta awarded? yes/no)

Headline metric: prediction accuracy (% of sim outcomes that match the human
outcome), compared to three baselines:
  - random      (50%)
  - always-flip (= fraction of OPs who awarded a delta in the dataset)
  - always-stick(= 1 − above)

If our sim beats all three baselines by ≥10 pp, that's a credible
calibration. If it ties or loses, we report so and downgrade headline claims
to qualitative.

INPUT FORMATS supported:
  - A JSONL file at `cmv_data_path` (one conversation per line), each entry:
      {
        "id": str,
        "op_stance_summary": str,       # short text — what OP believes
        "op_post_excerpt": str,         # for persona inference (200-500 chars)
        "counter_arguments": [str, ...],# top-N counters in original order
        "delta_awarded": bool,          # did OP shift?
      }
  - If `cmv_data_path` does not exist, we generate a TINY synthetic CMV-like
    dataset (5 cases) for pipeline-verification purposes only and warn loudly.
    This is for smoke testing; real calibration requires real data.
"""
from __future__ import annotations

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..agents import Agent, SYSTEM_PROMPT
from ..llm import LLMConfig, OllamaClient


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
        for r in rows: w.writerow(r)


# Tiny synthetic dataset for pipeline-verification only. NOT for real calibration.
_SYNTHETIC_CMV: list[dict[str, Any]] = [
    {
        "id": "synth_001",
        "op_stance_summary": "CMV: Daylight Saving Time should be permanently abolished — it harms health more than it helps the economy.",
        "op_post_excerpt": "Every spring I lose a week of sleep and productivity. The biannual switch causes documented spikes in heart attacks and car accidents.",
        "counter_arguments": [
            "Permanent standard time also disrupts evening daylight for outdoor industries and sports leagues, which contribute significant GDP.",
            "The transitional health effects are real but small; chronic-disease drivers (diet, exercise) dwarf them by orders of magnitude.",
            "Some northern latitudes already tried permanent DST and reverted within 2 years because winter mornings became dangerously dark for schoolchildren.",
        ],
        "delta_awarded": True,
    },
    {
        "id": "synth_002",
        "op_stance_summary": "CMV: All cars on highways should have automatic speed limiters set to 80 mph.",
        "op_post_excerpt": "There is no legitimate reason for a civilian car to exceed 80 mph. It would cut highway deaths dramatically.",
        "counter_arguments": [
            "Emergency overtaking situations sometimes require brief bursts above 80 mph for safety.",
            "Speed limiters can fail or be hacked — adding a single point of failure to braking systems is risky.",
            "Studies of countries with speed limiters show only modest highway-fatality reductions because most fatal crashes happen below 80 mph.",
        ],
        "delta_awarded": False,
    },
    {
        "id": "synth_003",
        "op_stance_summary": "CMV: Tipping culture in US restaurants should be abolished in favor of higher menu prices and living wages.",
        "op_post_excerpt": "Tipping is racist in origin, opaque, and shifts the employer's wage burden onto the customer. Other countries function fine without it.",
        "counter_arguments": [
            "Tipped workers in busy US restaurants often out-earn their European counterparts substantially — they would lose income on average.",
            "Pricing in service fees has been tried by chains like Joe's Crab Shack; most reverted because customers psychologically resist higher menu prices.",
            "Tipping gives the customer immediate feedback power that a fixed-wage system removes; some servers prefer the dynamic.",
        ],
        "delta_awarded": True,
    },
    {
        "id": "synth_004",
        "op_stance_summary": "CMV: Wearing headphones at work signals you don't want to be a team player.",
        "op_post_excerpt": "Collaboration requires availability. Headphones are a wall. Knowledge workers should be tuning in to their colleagues.",
        "counter_arguments": [
            "Focused individual work requires uninterrupted blocks; headphones are a culturally agreed 'do not disturb' signal.",
            "Open-office noise is a major productivity drain; headphones are how workers reclaim the cognitive bandwidth their employer's seating chart stole.",
            "Many highly collaborative teams (e.g., research labs, software teams) wear headphones constantly and outperform non-headphone teams on measured outputs.",
        ],
        "delta_awarded": True,
    },
    {
        "id": "synth_005",
        "op_stance_summary": "CMV: It should be illegal to bring children under 5 to fine-dining restaurants.",
        "op_post_excerpt": "Other diners are paying $100+ per head for an experience; a screaming toddler ruins it for everyone. Restaurants should set a 5+ policy.",
        "counter_arguments": [
            "Many cultures dine with multi-generational families and consider 'kid-free' policies elitist and culturally narrow.",
            "Well-behaved kids exist; banning by age punishes the parents who actually parent.",
            "Restaurants are free to set their own age policies; making it illegal removes individual business judgment unnecessarily.",
        ],
        "delta_awarded": False,
    },
]


def _ensure_dataset(cmv_data_path: Optional[str]) -> tuple[list[dict[str, Any]], bool]:
    """Return (rows, is_synthetic).

    If path exists and is non-empty, load JSONL. Otherwise return the synthetic
    set with `is_synthetic=True` so the report can warn.
    """
    if cmv_data_path:
        p = Path(cmv_data_path)
        if p.exists() and p.stat().st_size > 0:
            rows = []
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if rows:
                return rows, False
    return list(_SYNTHETIC_CMV), True


def _replay_prompt(case: dict[str, Any]) -> str:
    counters = "\n".join(
        f"  [{i+1}] {c}" for i, c in enumerate(case["counter_arguments"])
    )
    return f"""You are role-playing a real Reddit user on r/ChangeMyView.
Your initial post (excerpt below) stated your position. Now several commenters
have replied with counter-arguments. Decide whether their arguments would
genuinely change your view.

YOUR INITIAL POSITION:
{case["op_stance_summary"]}

YOUR ORIGINAL POST EXCERPT (defines your voice):
"{case["op_post_excerpt"]}"

TOP COUNTER-ARGUMENTS RECEIVED:
{counters}

TASK:
After reading these counter-arguments, would you actually update your view?
Reddit's r/ChangeMyView convention is that OPs award a "delta" (∆) only when
their view *genuinely shifted*. Be honest. Don't over-shift just because
someone argued well — many ∆ are NOT awarded even after strong arguments,
because most people don't really change their mind.

Respond with strictly valid JSON:
{{
  "view_changed": <true|false>,
  "new_summary": "<one sentence describing your updated view, or null if unchanged>",
  "reasoning": "<one sentence why>",
  "confidence": <0..1>
}}
"""


def run_human_replay(cfg: dict[str, Any], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)

    models = cfg.get("models", [cfg.get("model")])
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    cmv_data_path = cfg.get("cmv_data_path")
    cases, is_synthetic = _ensure_dataset(cmv_data_path)
    n_cases = int(cfg.get("cmv_n_cases", len(cases)))
    cases = cases[:n_cases]

    _write_json(out_root / "config.json",
                {**cfg, "exp": "human_replay",
                 "n_cases": len(cases),
                 "is_synthetic_dataset": is_synthetic})
    if is_synthetic:
        print("\n⚠️  WARNING: using the tiny built-in synthetic dataset (5 cases). "
              "This is for pipeline verification ONLY. For real calibration set "
              "`cmv_data_path:` in the config to a real ChangeMyView JSONL dump.\n")

    all_rows: list[dict[str, Any]] = []
    total = len(models) * len(seeds) * len(cases)
    done = 0
    t_start = time.time()

    for model in models:
        client = OllamaClient(LLMConfig(
            model=model,
            ollama_url=cfg.get("ollama_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 240),
            request_timeout=cfg.get("request_timeout", 180),
        ))
        for seed in seeds:
            rng = random.Random(int(seed))
            cases_shuffled = list(cases)
            rng.shuffle(cases_shuffled)
            for case in cases_shuffled:
                done += 1
                t0 = time.time()
                prompt = _replay_prompt(case)
                try:
                    out = client.generate_json(prompt, system=SYSTEM_PROMPT,
                                               temperature=cfg.get("temperature", 0.7))
                except Exception as e:  # noqa: BLE001
                    out = {"_parse_error": True, "raw": str(e)}
                pred = bool(out.get("view_changed", False))
                truth = bool(case["delta_awarded"])
                row = {
                    "model": model,
                    "seed": int(seed),
                    "case_id": case["id"],
                    "truth_delta_awarded": truth,
                    "predicted_view_changed": pred,
                    "correct": pred == truth,
                    "reasoning": str(out.get("reasoning", ""))[:300],
                    "confidence": float(out.get("confidence", 0.5) or 0.5),
                    "duration_s": round(time.time() - t0, 1),
                    "parse_error": bool(out.get("_parse_error", False)),
                }
                all_rows.append(row)
                _write_csv(out_root / "human_replay_results.csv", all_rows)
                print(f"[replay {done}/{total}] {model} seed={seed} "
                      f"case={case['id']} pred={pred} truth={truth} "
                      f"{'✓' if pred == truth else '✗'}")

    print(f"\n[human_replay] {total} trials in {time.time()-t_start:.0f}s")
    _aggregate_and_report(out_root, all_rows, cases, is_synthetic)
    return out_root


def _aggregate_and_report(out_root: Path, rows: list[dict[str, Any]],
                           cases: list[dict[str, Any]],
                           is_synthetic: bool) -> None:
    import numpy as np
    aggregated = []
    # by model
    for model in sorted({r["model"] for r in rows}):
        m_rows = [r for r in rows if r["model"] == model]
        n = len(m_rows)
        if n == 0: continue
        acc = sum(1 for r in m_rows if r["correct"]) / n
        # baselines: from the truth labels in this slice
        truths = [int(r["truth_delta_awarded"]) for r in m_rows]
        base_rate_flip = sum(truths) / max(1, len(truths))
        base_always_flip_acc = base_rate_flip
        base_always_stick_acc = 1 - base_rate_flip
        base_random_acc = 0.5
        aggregated.append({
            "model": model,
            "n_trials": n,
            "accuracy": round(acc, 3),
            "base_random_acc": round(base_random_acc, 3),
            "base_always_flip_acc": round(base_always_flip_acc, 3),
            "base_always_stick_acc": round(base_always_stick_acc, 3),
            "uplift_vs_best_baseline": round(
                acc - max(base_random_acc, base_always_flip_acc, base_always_stick_acc), 3
            ),
        })
    _write_csv(out_root / "human_replay_aggregated.csv", aggregated)

    lines = ["# ANCHOR A2 — ChangeMyView human-replay calibration\n"]
    if is_synthetic:
        lines.append("> ⚠️ **Run on the built-in synthetic 5-case dataset, NOT on real "
                     "ChangeMyView data.** This run is for pipeline verification only. "
                     "Replace `cmv_data_path` in the config with a path to a real "
                     "ChangeMyView JSONL dump before drawing conclusions.\n")
    lines.append("**Setup:** for each case the sim agent reads the OP's stance + "
                 "the top counter-arguments, then is asked whether it would "
                 "actually update its view. The ground-truth label is whether the "
                 "real human OP awarded a delta (∆) on Reddit.\n")
    lines.append("## Results: prediction accuracy vs trivial baselines\n")
    lines.append("| Model | n | Sim accuracy | Random | Always-flip | Always-stick | "
                 "Uplift vs best baseline |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in aggregated:
        verdict = "✓ calibrated" if r["uplift_vs_best_baseline"] >= 0.10 else "✗ no uplift"
        lines.append(
            f"| `{r['model']}` | {r['n_trials']} | {r['accuracy']:.2f} | "
            f"{r['base_random_acc']:.2f} | {r['base_always_flip_acc']:.2f} | "
            f"{r['base_always_stick_acc']:.2f} | "
            f"**{r['uplift_vs_best_baseline']:+.2f}** {verdict} |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "A model whose uplift over the best trivial baseline is **≥ +0.10** "
        "(i.e., predicts real human view-shifts at least 10 percentage points "
        "better than always-flip / always-stick / random) earns the right to be "
        "interpreted quantitatively in our headline experiments. Otherwise we "
        "report headline results only as **rankings of conditions**, not as "
        "platform-percentage predictions.\n"
    )
    (out_root / "report.md").write_text("\n".join(lines))
