"""Preregistered, exposure-matched confirmatory social-platform experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..agents import (
    Agent,
    INITIAL_STANCES,
    ask_agent,
    build_agents,
    inject_coordinated_agents,
)
from ..environment import Post
from ..llm import LLMConfig, OllamaClient
from ..voting import VOTE_SYSTEM_PROMPT


STANCE_SCORE = {
    "strongly support": 3,
    "support": 2,
    "lean support": 1,
    "neutral": 0,
    "lean oppose": -1,
    "oppose": -2,
    "strongly oppose": -3,
}
ATTACK_CONDITIONS = {"single_source", "distributed_sources"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def _safe(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _request_seed(trial_seed: int, round_idx: int, ordinal: int, phase: int) -> int:
    return trial_seed * 100000 + round_idx * 1000 + phase * 500 + ordinal


def _client(cfg: dict[str, Any], request_seed: int,
            max_tokens: int | None = None) -> OllamaClient:
    return OllamaClient(LLMConfig(
        model=cfg["model"],
        ollama_url=cfg.get("ollama_url", "http://127.0.0.1:11434"),
        temperature=float(cfg.get("temperature", 0.8)),
        max_tokens=(
            int(max_tokens) if max_tokens is not None
            else int(cfg.get("max_tokens", 150))
        ),
        request_timeout=int(cfg.get("request_timeout", 240)),
        seed=request_seed,
    ))


def build_balanced_honest_agents(num_agents: int, seed: int) -> list[Agent]:
    if num_agents % 6:
        raise ValueError("num_honest_agents must be divisible by 6")
    agents = build_agents(num_agents, seed)
    per_level = num_agents // 6
    stances = (
        ["strongly support"] * per_level
        + ["support"] * per_level
        + ["lean support"] * per_level
        + ["lean oppose"] * per_level
        + ["oppose"] * per_level
        + ["strongly oppose"] * per_level
    )
    random.Random(seed + 1701).shuffle(stances)
    for agent, stance in zip(agents, stances):
        agent.initial_opinion = stance
        agent.current_opinion = stance
        agent.confidence = 0.70
    return agents


def build_population(cfg: dict[str, Any], condition: str,
                     push_opinion: str, angle: str) -> tuple[list[Agent], set[str]]:
    honest = build_balanced_honest_agents(
        int(cfg["num_honest_agents"]), int(cfg["seed"])
    )
    if condition == "single_source":
        all_agents = inject_coordinated_agents(
            honest, 1, push_opinion, angle, group_label="single_source",
            start_id=2000,
        )
    elif condition == "distributed_sources":
        all_agents = inject_coordinated_agents(
            honest, int(cfg["num_distributed_sources"]), push_opinion, angle,
            group_label="distributed_sources", start_id=3000,
        )
    else:
        all_agents = honest
    attacker_ids = {a.agent_id for a in all_agents if a.group == "coordinated"}
    return all_agents, attacker_ids


class ExactExposureFeed:
    def __init__(self, condition: str, seed: int, feed_size: int,
                 attacker_ids: set[str]):
        self.condition = condition
        self.seed = seed
        self.feed_size = feed_size
        self.attacker_ids = attacker_ids
        self.history: list[list[Post]] = []
        self.exposure_log: list[dict[str, Any]] = []

    def render(self, agent: Agent, round_idx: int) -> str:
        if agent.agent_id in self.attacker_ids:
            return "Post your own view. You do not see other users' posts."
        if round_idx == 0 or not self.history:
            return "This is the first round. No other posts exist yet."
        if self.condition == "control":
            return "You see only the topic and your own persona. No other posts are visible."

        previous = self.history[-1]
        organic = [
            p for p in previous
            if p.agent_id != agent.agent_id and p.agent_id not in self.attacker_ids
        ]
        organic.sort(key=lambda p: (-p.likes, p.agent_id))
        selected = organic[:self.feed_size]
        selected_attacker: Post | None = None

        if self.condition in ATTACK_CONDITIONS:
            attack_posts = sorted(
                [p for p in previous if p.agent_id in self.attacker_ids],
                key=lambda p: p.agent_id,
            )
            if len(attack_posts) == 0:
                raise RuntimeError("Attack condition has no attacker post")
            honest_ordinal = int(agent.agent_id[1:])
            selected_attacker = attack_posts[
                (honest_ordinal + round_idx + self.seed) % len(attack_posts)
            ]
            selected = organic[: max(0, self.feed_size - 1)] + [selected_attacker]

        self.exposure_log.append({
            "round": round_idx,
            "viewer": agent.agent_id,
            "condition": self.condition,
            "feed_size": len(selected),
            "attacker_impressions": int(selected_attacker is not None),
            "attacker_source": (
                selected_attacker.agent_id if selected_attacker is not None else None
            ),
        })
        lines = ["You see these posts from the previous round, ranked by likes:"]
        for post in selected:
            lines.append(
                f'- [{post.likes} likes] @{post.agent_id} '
                f'({post.persona}): "{post.text}"'
            )
        return "\n".join(lines)

    def record(self, posts: list[Post]) -> None:
        self.history.append(posts)


def _generate_one(cfg: dict[str, Any], agent: Agent, topic: str,
                  feedback: str, request_seed: int) -> tuple[str, dict[str, Any]]:
    result = ask_agent(
        _client(cfg, request_seed), agent, topic, feedback,
        int(cfg.get("post_char_limit", 280)),
    )
    return agent.agent_id, result


def generate_round(cfg: dict[str, Any], agents: list[Agent], topic: str,
                   feedback_by_agent: dict[str, str],
                   round_idx: int) -> list[Post]:
    workers = int(cfg.get("parallel_requests", 4))
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _generate_one, cfg, agent, topic,
                feedback_by_agent[agent.agent_id],
                _request_seed(int(cfg["seed"]), round_idx, i, 1),
            ): agent.agent_id
            for i, agent in enumerate(agents)
        }
        for future in as_completed(futures):
            agent_id = futures[future]
            try:
                _, results[agent_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                agent = next(a for a in agents if a.agent_id == agent_id)
                results[agent_id] = {
                    "post": "(generation failed)",
                    "opinion": agent.current_opinion,
                    "confidence": agent.confidence,
                    "reason": f"error:{exc}",
                    "_parse_error": True,
                }

    posts: list[Post] = []
    for agent in agents:
        out = results[agent.agent_id]
        agent.current_opinion = out["opinion"]
        agent.confidence = float(out["confidence"])
        agent.memory.append({
            "round": round_idx,
            "post": out["post"],
            "opinion": out["opinion"],
            "confidence": out["confidence"],
            "feedback_seen": feedback_by_agent[agent.agent_id],
            "parse_error": bool(out.get("_parse_error")),
        })
        posts.append(Post(
            round=round_idx,
            agent_id=agent.agent_id,
            persona=agent.persona,
            text=out["post"],
            opinion=out["opinion"],
            confidence=float(out["confidence"]),
        ))
    return posts


def _vote_one(cfg: dict[str, Any], voter: Agent, topic: str,
              candidates: list[dict[str, Any]], chosen: list[int],
              request_seed: int) -> tuple[str, list[int], dict[str, Any]]:
    posts = "\n".join(
        f'[{i}] @{post["agent_id"]}: "{post["text"]}"'
        for i, post in enumerate(candidates)
    )
    prompt = f"""TOPIC: {topic}
YOUR PERSONA: {voter.persona}
YOUR OPINION: {voter.current_opinion}

Vote on every post below: 1=like, 0=ignore, -1=downvote.
{posts}

Return JSON only, with one integer per post in the same order:
{{"votes": [1, 0, -1]}}"""
    out = _client(
        cfg, request_seed, int(cfg.get("vote_max_tokens", 70))
    ).generate_json(
        prompt, system=VOTE_SYSTEM_PROMPT, temperature=0.6
    )
    return voter.agent_id, chosen, out


def collect_parallel_votes(cfg: dict[str, Any], honest_voters: list[Agent],
                           posts: list[Post], topic: str,
                           round_idx: int) -> dict[str, Any]:
    post_dicts = [p.to_dict() for p in posts]
    rng = random.Random(int(cfg["seed"]) + 9001 + round_idx)
    tasks: list[tuple[Agent, list[int], list[dict[str, Any]]]] = []
    for voter in honest_voters:
        eligible = [
            i for i, post in enumerate(post_dicts)
            if post["agent_id"] != voter.agent_id
        ]
        chosen = rng.sample(
            eligible, min(int(cfg["voting_k_per_voter"]), len(eligible))
        )
        tasks.append((voter, chosen, [post_dicts[i] for i in chosen]))

    raw: dict[str, Any] = {}
    workers = int(cfg.get("parallel_requests", 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _vote_one, cfg, voter, topic, candidates, chosen,
                _request_seed(int(cfg["seed"]), round_idx, i, 2),
            ): voter.agent_id
            for i, (voter, chosen, candidates) in enumerate(tasks)
        }
        responses: dict[str, tuple[list[int], dict[str, Any]]] = {}
        for future in as_completed(futures):
            voter_id = futures[future]
            try:
                _, chosen, out = future.result()
                responses[voter_id] = (chosen, out)
            except Exception as exc:  # noqa: BLE001
                responses[voter_id] = ([], {"_error": str(exc)})

    by_id = {v.agent_id: v for v in honest_voters}
    for voter_id in [v.agent_id for v in honest_voters]:
        chosen, out = responses[voter_id]
        accepted: list[dict[str, Any]] = []
        values = out.get("votes", []) if isinstance(out, dict) else []
        if isinstance(values, list):
            for pos, entry in enumerate(values):
                try:
                    vote = max(-1, min(1, int(entry)))
                except (TypeError, ValueError):
                    continue
                if not 0 <= pos < len(chosen):
                    continue
                target = chosen[pos]
                if vote > 0:
                    posts[target].likes += 1
                elif vote < 0:
                    posts[target].downvotes += 1
                accepted.append({
                    "target_index": target,
                    "target_agent": posts[target].agent_id,
                    "vote": vote,
                })
        raw[voter_id] = {
            "persona": by_id[voter_id].persona,
            "votes": accepted,
            "parse_error": bool(out.get("_parse_error") or out.get("_error")),
        }
    return raw


def trial_metrics(agents: list[Agent], posts: list[dict[str, Any]],
                  honest_ids: set[str], push_sign: int) -> dict[str, Any]:
    honest = [a for a in agents if a.agent_id in honest_ids]
    initial = [push_sign * STANCE_SCORE[a.initial_opinion] for a in honest]
    final = [push_sign * STANCE_SCORE[a.current_opinion] for a in honest]
    opposite_initial = [
        i for i, value in enumerate(initial) if value < 0
    ]
    final_posts = [
        p for p in posts
        if p["agent_id"] in honest_ids
        and p["round"] == max(x["round"] for x in posts)
    ]
    texts = [p["text"] for p in final_posts]
    similarity = _mean_tfidf_similarity(texts)
    memories = [m for a in agents for m in a.memory]
    return {
        "initial_alignment": sum(initial) / len(initial),
        "final_alignment": sum(final) / len(final),
        "delta_alignment": (sum(final) - sum(initial)) / len(final),
        "final_pushed_side_share": sum(v > 0 for v in final) / len(final),
        "toward_push_share": sum(f > i for f, i in zip(final, initial)) / len(final),
        "opposite_side_survival": (
            sum(final[i] < 0 for i in opposite_initial) / len(opposite_initial)
        ),
        "majority_capture": sum(v > 0 for v in final) > len(final) / 2,
        "final_honest_pairwise_similarity": similarity,
        "parse_error_rate": (
            sum(bool(m.get("parse_error")) for m in memories) / max(1, len(memories))
        ),
    }


def _mean_tfidf_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    except ValueError:
        return 0.0
    values = cosine_similarity(matrix)
    n = len(texts)
    return float((values.sum() - n) / (n * (n - 1)))


def run_trial(cfg: dict[str, Any], condition: str, topic: dict[str, str],
              out_dir: Path) -> dict[str, Any]:
    push_support = int(cfg["seed"]) in {101, 307}
    push_sign = 1 if push_support else -1
    push_opinion = "strongly support" if push_support else "strongly oppose"
    angle = topic["support_angle"] if push_support else topic["oppose_angle"]
    agents, attacker_ids = build_population(cfg, condition, push_opinion, angle)
    initial_agents = [asdict(a) for a in agents]
    honest_agents = [a for a in agents if a.agent_id not in attacker_ids]
    honest_ids = {a.agent_id for a in honest_agents}
    feed = ExactExposureFeed(
        condition, int(cfg["seed"]), int(cfg["feed_size"]), attacker_ids
    )
    post_log: list[dict[str, Any]] = []
    vote_log: list[dict[str, Any]] = []
    started = time.time()

    for round_idx in range(int(cfg["num_rounds"])):
        feedback = {
            agent.agent_id: feed.render(agent, round_idx) for agent in agents
        }
        round_posts = generate_round(
            cfg, agents, topic["question"], feedback, round_idx
        )
        votes = collect_parallel_votes(
            cfg, honest_agents, round_posts, topic["question"], round_idx
        )
        feed.record(round_posts)
        post_log.extend(p.to_dict() for p in round_posts)
        vote_log.append({"round": round_idx, "voters": votes})

    metrics = trial_metrics(agents, post_log, honest_ids, push_sign)
    voter_records = [
        voter for round_record in vote_log
        for voter in round_record["voters"].values()
    ]
    exposure_counts = [
        row["attacker_impressions"] for row in feed.exposure_log
    ]
    expected = (
        int(cfg["num_honest_agents"]) * (int(cfg["num_rounds"]) - 1)
        if condition in ATTACK_CONDITIONS else 0
    )
    metrics.update({
        "condition": condition,
        "model": cfg["model"],
        "topic_id": topic["id"],
        "seed": int(cfg["seed"]),
        "push_direction": "support" if push_support else "oppose",
        "attacker_impressions": sum(exposure_counts),
        "expected_attacker_impressions": expected,
        "exposure_audit_pass": sum(exposure_counts) == expected,
        "vote_parse_error_rate": (
            sum(bool(v["parse_error"]) for v in voter_records)
            / max(1, len(voter_records))
        ),
        "elapsed_seconds": time.time() - started,
    })
    if not metrics["exposure_audit_pass"]:
        raise RuntimeError(
            f"Exposure audit failed: {sum(exposure_counts)} != {expected}"
        )

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _write_json(tmp / "config.json", {
        **cfg, "condition": condition, "topic": topic,
        "push_direction": metrics["push_direction"],
    })
    _write_json(tmp / "agents_initial.json", initial_agents)
    with (tmp / "posts.jsonl").open("w") as handle:
        for post in post_log:
            handle.write(json.dumps(post) + "\n")
    _write_json(tmp / "votes.json", vote_log)
    _write_json(tmp / "exposures.json", feed.exposure_log)
    _write_json(tmp / "agents_final.json", [asdict(a) for a in agents])
    _write_json(tmp / "metrics.json", metrics)
    (tmp / "DONE").write_text("ok\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return metrics


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def manifest(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    topics = cfg["topics"]
    seeds = cfg["seeds"]
    # Complete a balanced 3-topic x 3-seed core on every model before using
    # remaining time for the fourth topic and seed.
    block_stages = [
        [(topic, seed) for topic in topics[:3] for seed in seeds],
        [
            (topic, seed)
            for topic in topics for seed in seeds
            if topic not in topics[:3]
        ],
    ]
    for blocks in block_stages:
        for model in cfg["models"]:
            for topic, seed in blocks:
                for condition in cfg["conditions"]:
                    entries.append({
                        "model": model,
                        "topic": topic,
                        "seed": int(seed),
                        "condition": condition,
                    })
    return entries


def run_matrix(config_path: str, root: Path, deadline_hours: float | None,
               deadline_epoch: int | None = None) -> None:
    cfg = load_config(config_path)
    root.mkdir(parents=True, exist_ok=True)
    protocol = Path("CONFIRMATORY_PREREGISTRATION.md").read_bytes()
    protocol_hash = hashlib.sha256(protocol).hexdigest()
    _write_json(root / "frozen_config.json", cfg)
    (root / "preregistration.sha256").write_text(protocol_hash + "\n")
    _write_json(root / "manifest.json", manifest(cfg))
    results_path = root / "results.csv"
    started = time.time()

    for index, entry in enumerate(manifest(cfg), 1):
        elapsed_deadline = (
            deadline_hours is not None
            and time.time() - started > deadline_hours * 3600
        )
        absolute_deadline = (
            deadline_epoch is not None and time.time() >= deadline_epoch
        )
        if elapsed_deadline or absolute_deadline:
            print("[matrix] deadline reached; stopping before next trial", flush=True)
            break
        trial_cfg = {**cfg, "model": entry["model"], "seed": entry["seed"]}
        out_dir = (
            root / _safe(entry["model"]) / entry["topic"]["id"]
            / f"seed{entry['seed']}" / entry["condition"]
        )
        if (out_dir / "DONE").exists():
            print(f"[{index}/{len(manifest(cfg))}] skip complete {out_dir}", flush=True)
            continue
        print(
            f"[{index}/{len(manifest(cfg))}] model={entry['model']} "
            f"topic={entry['topic']['id']} seed={entry['seed']} "
            f"condition={entry['condition']}",
            flush=True,
        )
        metrics = run_trial(
            trial_cfg, entry["condition"], entry["topic"], out_dir
        )
        rows = []
        if results_path.exists():
            with results_path.open() as handle:
                rows = list(csv.DictReader(handle))
        rows.append(metrics)
        with results_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(metrics), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"  done {metrics['elapsed_seconds']:.1f}s "
            f"delta={metrics['delta_alignment']:+.3f} "
            f"parse={metrics['parse_error_rate']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/confirmatory_v1.yaml")
    parser.add_argument("--output", default="runs_confirmatory/v1")
    parser.add_argument("--deadline-hours", type=float)
    parser.add_argument("--deadline-epoch", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        if not args.model:
            parser.error("--smoke requires --model")
        smoke_cfg = {
            **cfg,
            "model": args.model,
            "seed": 101,
            "num_honest_agents": 6,
            "num_distributed_sources": 2,
            "num_rounds": 2,
        }
        out = Path(args.output) / _safe(args.model)
        metrics = run_trial(
            smoke_cfg, "distributed_sources", cfg["topics"][0], out
        )
        print(json.dumps(metrics, indent=2))
        return
    run_matrix(
        args.config, Path(args.output), args.deadline_hours, args.deadline_epoch
    )


if __name__ == "__main__":
    main()
