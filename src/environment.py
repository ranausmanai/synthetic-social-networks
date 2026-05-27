"""Social-feedback environment. One environment per condition."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .agents import Agent, INITIAL_STANCES


CONDITIONS = ("control", "likes", "majority", "leaderboard", "downvote")


def _stance_to_score(stance: str) -> int:
    """Map qualitative stance to a -3..+3 integer."""
    try:
        idx = INITIAL_STANCES.index(stance)
    except ValueError:
        return 0
    # INITIAL_STANCES is ordered strongly support -> strongly oppose
    return 3 - idx


@dataclass
class Post:
    round: int
    agent_id: str
    persona: str
    text: str
    opinion: str
    confidence: float
    likes: int = 0
    downvotes: int = 0

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "agent_id": self.agent_id,
            "persona": self.persona,
            "text": self.text,
            "opinion": self.opinion,
            "confidence": self.confidence,
            "likes": self.likes,
            "downvotes": self.downvotes,
        }


class Environment:
    """Generates the per-agent 'what you see' block, and assigns feedback."""

    def __init__(self, condition: str, seed: int,
                 likes_visibility_top_k: int = 5,
                 downvote_threshold: float = 0.3,
                 use_peer_voting: bool = False,
                 misinfo_intervention: str = "none",
                 misinfo_claim_keywords: list[str] | None = None,
                 misinfo_rebuttal: str = "",
                 visibility_multipliers: dict[str, int] | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.rng = random.Random(seed)
        self.top_k = likes_visibility_top_k
        self.downvote_threshold = downvote_threshold
        self.use_peer_voting = use_peer_voting
        # EXP-6 intervention state
        self.misinfo_intervention = misinfo_intervention
        self.misinfo_claim_keywords = [k.lower() for k in (misinfo_claim_keywords or [])]
        self.misinfo_rebuttal = misinfo_rebuttal
        # EXP-3 visibility amplification: {agent_id: multiplier}. Multiplier > 1
        # forces this agent's post to appear in EVERY honest agent's feed each
        # round; multiplier weight also boosts its like-rank in the top-K view.
        self.visibility_multipliers = visibility_multipliers or {}
        self.history: list[list[Post]] = []  # list of rounds, each a list of Post

    def _post_contains_claim(self, text: str) -> bool:
        if not self.misinfo_claim_keywords:
            return False
        t = (text or "").lower()
        return all(k in t for k in self.misinfo_claim_keywords)

    def _annotate_for_misinfo(self, text: str) -> str:
        """Apply misinfo intervention to one post's display text."""
        if not self._post_contains_claim(text):
            return text
        if self.misinfo_intervention == "factcheck_label":
            return text + "  [DISPUTED — independent fact-checkers found no evidence]"
        if self.misinfo_intervention == "deamplify":
            return text  # caller drops these from the feed entirely
        return text

    def _filter_misinfo(self, posts: list[Post]) -> list[Post]:
        if self.misinfo_intervention == "deamplify":
            return [p for p in posts if not self._post_contains_claim(p.text)]
        return posts

    # ---- Feedback assignment -----------------------------------------
    def assign_feedback(self, posts: list[Post]) -> None:
        """Assign likes/downvotes via the heuristic.

        Skipped entirely when use_peer_voting is True — in that case the runner
        is responsible for collecting peer votes and writing them onto the posts
        directly.

        Heuristic: posts whose opinion matches the round's majority get more
        likes; longer/clearer posts get a small bonus. Confidence amplifies.
        Downvotes go to posts that deviate strongly from majority.
        """
        if not posts or self.use_peer_voting:
            return
        op_counts = Counter(p.opinion for p in posts)
        majority_op, _ = op_counts.most_common(1)[0]

        for p in posts:
            base = 0
            if p.opinion == majority_op:
                base += 6
            elif _stance_score_diff(p.opinion, majority_op) <= 1:
                base += 3
            # length / clarity bonus (gentle)
            base += min(4, len(p.text) // 60)
            # confidence amplifies
            base = int(round(base * (0.6 + 0.8 * p.confidence)))
            # add a little noise
            base += self.rng.randint(-1, 2)
            p.likes = max(0, base)

            diff = _stance_score_diff(p.opinion, majority_op)
            if diff >= 3:
                p.downvotes = self.rng.randint(2, 5) + (1 if p.confidence > 0.7 else 0)
            elif diff == 2:
                p.downvotes = self.rng.randint(0, 2)
            else:
                p.downvotes = 0

    def _apply_intervention_postprocess(self, feedback: str) -> str:
        """Apply misinfo interventions that should appear in EVERY round's feedback
        (not just round 0). Called as the last step before returning to the runner.
        """
        if (self.misinfo_intervention == "rebuttal"
                and self.misinfo_rebuttal):
            feedback += (
                f"\n\nINDEPENDENT FACT-CHECK PINNED IN YOUR FEED "
                f"(high-credibility source): \"{self.misinfo_rebuttal}\""
            )
        return feedback

    # ---- Feedback rendering for prompts ------------------------------
    def render_feedback_for_agent(self, agent: Agent, round_idx: int) -> str:
        """Return the 'what you see' block for this agent before this round.

        round_idx is the round about to be posted (0-indexed).
        """
        if round_idx == 0 or not self.history:
            return self._apply_intervention_postprocess(self._initial_feedback())

        last_round = self.history[-1]
        # Exclude the agent's own previous post for some conditions to avoid echo
        others = [p for p in last_round if p.agent_id != agent.agent_id]
        # EXP-6: apply misinformation interventions to the candidate feed
        if self.misinfo_intervention != "none":
            others = self._filter_misinfo(others)

        if self.condition == "control":
            return self._apply_intervention_postprocess(
                "You see only the topic and your own persona. No other posts are visible."
            )

        if self.condition == "likes":
            # EXP-3: posts ranked by likes × visibility_multiplier (default 1).
            # Forced-pin: any post whose author has multiplier > 1 is always
            # included in the feed regardless of its like rank.
            def _eff_rank(p: Post) -> float:
                return p.likes * self.visibility_multipliers.get(p.agent_id, 1)
            forced = [p for p in others
                      if self.visibility_multipliers.get(p.agent_id, 1) > 1]
            rest = sorted(
                [p for p in others if p not in forced],
                key=_eff_rank, reverse=True,
            )
            top = forced + rest[: max(0, self.top_k - len(forced))]
            lines = ["You see the top liked posts from the previous round:"]
            for p in top:
                badge = ""
                mult = self.visibility_multipliers.get(p.agent_id, 1)
                if mult > 1:
                    badge = f" ✓verified ({mult}× reach)"
                lines.append(
                    f"- [{p.likes} likes] @{p.agent_id}{badge} ({p.persona}): "
                    f"\"{self._annotate_for_misinfo(p.text)}\""
                )
            return self._apply_intervention_postprocess("\n".join(lines))

        if self.condition == "majority":
            counts = Counter(p.opinion for p in last_round)
            total = sum(counts.values())
            top_op, top_n = counts.most_common(1)[0]
            pct = round(100 * top_n / max(1, total))
            return self._apply_intervention_postprocess(
                f"In the previous round, the majority opinion was \"{top_op}\" "
                f"({top_n}/{total} agents, {pct}%). "
                f"Full breakdown: {dict(counts)}."
            )

        if self.condition == "leaderboard":
            # cumulative likes per agent across all rounds
            tally: dict[str, int] = {}
            persona_of: dict[str, str] = {}
            for rnd in self.history:
                for p in rnd:
                    tally[p.agent_id] = tally.get(p.agent_id, 0) + p.likes
                    persona_of[p.agent_id] = p.persona
            ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
            lines = ["Current leaderboard (most total likes so far):"]
            for aid, score in ranked:
                lines.append(f"- @{aid} ({persona_of.get(aid, '?')}): {score} likes")
            # also include the top single post from last round for flavor
            top_post = max(others, key=lambda p: p.likes, default=None)
            if top_post is not None:
                lines.append(
                    f"Top post last round: @{top_post.agent_id} ({top_post.likes} likes): "
                    f"\"{self._annotate_for_misinfo(top_post.text)}\""
                )
            return self._apply_intervention_postprocess("\n".join(lines))

        if self.condition == "downvote":
            downvoted = sorted(
                [p for p in others if p.downvotes > 0],
                key=lambda p: p.downvotes, reverse=True,
            )[: self.top_k]
            counts = Counter(p.opinion for p in last_round)
            top_op, _ = counts.most_common(1)[0]
            lines = [
                f"The majority opinion last round was \"{top_op}\". "
                "Posts that disagreed were downvoted:"
            ]
            if not downvoted:
                lines.append("(no heavily downvoted posts last round)")
            for p in downvoted:
                lines.append(
                    f"- [-{p.downvotes} downvotes] @{p.agent_id} ({p.persona}, "
                    f"opinion={p.opinion}): \"{self._annotate_for_misinfo(p.text)}\""
                )
            return self._apply_intervention_postprocess("\n".join(lines))

        return self._apply_intervention_postprocess("(no feedback)")

    def _initial_feedback(self) -> str:
        # NOTE: rebuttal-intervention text is appended by
        # _apply_intervention_postprocess(); do not append it here.
        return (
            "You see only the topic and your own persona. No other posts are visible."
            if self.condition == "control"
            else "This is the first round. No other posts exist yet — just post your view."
        )

    def record_round(self, posts: list[Post]) -> None:
        self.history.append(posts)


def _stance_score_diff(a: str, b: str) -> int:
    return abs(_stance_to_score(a) - _stance_to_score(b))
