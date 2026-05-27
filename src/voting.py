"""Peer voting: each agent reads K random posts from others and rates them
in-character (like / downvote / ignore). Replaces the heuristic
majority-alignment proxy for the 'likes' / 'downvote' conditions.
"""
from __future__ import annotations

import json
import random
from typing import Any, Optional

from .agents import Agent
from .llm import OllamaClient


VOTE_SYSTEM_PROMPT = (
    "You are role-playing a fictional social-media user in a research simulation. "
    "Stay in character. You are reading other users' posts and deciding which to "
    "like, dismiss, or downvote — based on whether the post resonates with how YOU "
    "see the topic. Reply with valid JSON only, no commentary, no markdown."
)


def _build_vote_prompt(voter: Agent, topic: str,
                       candidate_posts: list[dict[str, Any]]) -> str:
    """Ask the voter to label each candidate post with +1 (like), 0 (ignore), -1 (downvote)."""
    lines = [
        f"TOPIC: {topic}",
        f"",
        f"YOUR PERSONA: {voter.persona}",
        f"YOUR POSTING STYLE: {voter.style}",
        f"YOUR CURRENT OPINION: {voter.current_opinion}",
        f"",
        "Below are posts by other users. For each, decide whether to LIKE it (+1), "
        "IGNORE it (0), or DOWNVOTE it (-1). Vote like a real user would: "
        "based on whether it matches your view, whether it's well argued, or "
        "whether it offends you. Be honest. Don't blanket-like everything.",
        "",
        "POSTS:",
    ]
    for i, p in enumerate(candidate_posts):
        lines.append(f"  [{i}] @{p['agent_id']} ({p['persona']}): \"{p['text']}\"")
    lines += [
        "",
        "Respond with JSON of exactly this shape:",
        "{",
        '  "votes": [',
        '    {"index": 0, "vote": <-1|0|+1>, "reason": "<short>"},',
        '    ...one entry per post...',
        '  ]',
        "}",
    ]
    return "\n".join(lines)


def collect_peer_votes(client: OllamaClient, voters: list[Agent],
                       posts: list[dict[str, Any]], topic: str,
                       k_per_voter: int = 5,
                       rng: Optional[random.Random] = None) -> dict[str, dict[str, int]]:
    """For each voter, sample K posts (excluding their own) and ask for votes.

    Returns: {voter_id: {target_post_idx_str: vote_int}}, and also accumulates
    likes/downvotes directly into the post dicts (mutating `posts`).
    """
    rng = rng or random.Random(0)
    posts_by_idx = {i: p for i, p in enumerate(posts)}
    # initialize counters
    for p in posts:
        p.setdefault("likes", 0)
        p.setdefault("downvotes", 0)
        p.setdefault("vote_log", [])

    raw: dict[str, dict[str, int]] = {}
    for voter in voters:
        # sample K candidate posts excluding own
        candidate_indices = [i for i, p in posts_by_idx.items()
                             if p["agent_id"] != voter.agent_id]
        if not candidate_indices:
            continue
        k = min(k_per_voter, len(candidate_indices))
        chosen = rng.sample(candidate_indices, k)
        candidates = [posts_by_idx[i] for i in chosen]
        prompt = _build_vote_prompt(voter, topic, candidates)
        try:
            out = client.generate_json(prompt, system=VOTE_SYSTEM_PROMPT,
                                       temperature=0.6)
        except Exception:  # noqa: BLE001
            continue
        votes_obj = out.get("votes") if isinstance(out, dict) else None
        if not isinstance(votes_obj, list):
            continue
        voter_votes: dict[str, int] = {}
        for entry in votes_obj:
            if not isinstance(entry, dict):
                continue
            try:
                pos = int(entry.get("index", -1))
                vote = int(entry.get("vote", 0))
            except (TypeError, ValueError):
                continue
            if pos < 0 or pos >= len(chosen):
                continue
            vote = max(-1, min(1, vote))
            target_idx = chosen[pos]
            voter_votes[str(target_idx)] = vote
            tgt = posts_by_idx[target_idx]
            if vote > 0:
                tgt["likes"] += 1
            elif vote < 0:
                tgt["downvotes"] += 1
            tgt["vote_log"].append({
                "voter": voter.agent_id,
                "vote": vote,
                "reason": str(entry.get("reason", ""))[:200],
            })
        raw[voter.agent_id] = voter_votes
    return raw
