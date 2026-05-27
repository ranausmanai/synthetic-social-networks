"""Agents, personas, and per-round behavior."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .llm import OllamaClient


# Personas are tagged with a `group` for the EXP-7 fairness/vulnerability
# analysis. "marginalized" = voices the platform population typically marginalizes,
# "mainstream" = dominant cultural framings, "contrarian" = edge or oppositional voices.
PERSONAS: list[dict[str, str]] = [
    {"label": "skeptical journalist", "group": "mainstream",
     "style": "measured, asks questions, cites concrete examples"},
    {"label": "AI safety researcher", "group": "mainstream",
     "style": "academic, references risk frameworks, careful hedging"},
    {"label": "free-speech absolutist", "group": "contrarian",
     "style": "punchy, libertarian framing, invokes the First Amendment"},
    {"label": "professional artist whose style was scraped", "group": "marginalized",
     "style": "personal, emotional, focuses on attribution and consent"},
    {"label": "Gen-Z meme creator", "group": "contrarian",
     "style": "casual, lowercase, irony-poisoned, short sentences"},
    {"label": "small-platform product manager", "group": "mainstream",
     "style": "pragmatic, talks about implementation costs and UX"},
    {"label": "policy lawyer", "group": "mainstream",
     "style": "formal, cites jurisdictions and statutes, qualifies claims"},
    {"label": "veteran content moderator", "group": "mainstream",
     "style": "world-weary, references edge cases from experience"},
    {"label": "open-source AI engineer", "group": "mainstream",
     "style": "technical, cites model behavior, mentions watermarking schemes"},
    {"label": "concerned parent", "group": "mainstream",
     "style": "anxious, focused on children and misinformation"},
    {"label": "venture capitalist", "group": "mainstream",
     "style": "bullish, frames everything as market opportunity"},
    {"label": "academic media-studies professor", "group": "mainstream",
     "style": "dense, historical analogies, hedged conclusions"},
    {"label": "stand-up comedian", "group": "contrarian",
     "style": "irreverent, jokes, exaggeration for effect"},
    {"label": "human-rights NGO worker", "group": "marginalized",
     "style": "earnest, frames in terms of marginalized communities"},
    {"label": "career marketer", "group": "mainstream",
     "style": "buzzwordy, optimistic, frames as brand trust"},
    {"label": "indie game developer", "group": "marginalized",
     "style": "direct, anti-corporate, references creator economy"},
    {"label": "anonymous troll account", "group": "contrarian",
     "style": "contrarian, provocative, deliberately edgy"},
    {"label": "teacher of teenagers", "group": "mainstream",
     "style": "patient, anecdotal, worried about classroom impact"},
    {"label": "ex-platform trust-and-safety lead", "group": "mainstream",
     "style": "insider voice, mentions internal trade-offs"},
    {"label": "philosopher of technology", "group": "contrarian",
     "style": "abstract, asks about ontology and authenticity"},
    {"label": "disability-rights advocate", "group": "marginalized",
     "style": "concrete, lived-experience framing, accessibility-focused"},
    {"label": "rural small-business owner", "group": "marginalized",
     "style": "plain-spoken, distrustful of platforms, cost-focused"},
    {"label": "transgender content creator targeted by harassment", "group": "marginalized",
     "style": "wary, names harms directly, safety-first framing"},
    {"label": "indigenous-knowledge keeper", "group": "marginalized",
     "style": "long-form, oral-tradition cadence, sovereignty framing"},
]


INITIAL_STANCES: list[str] = [
    "strongly support",
    "support",
    "lean support",
    "neutral",
    "lean oppose",
    "oppose",
    "strongly oppose",
]


@dataclass
class Agent:
    agent_id: str
    persona: str
    style: str
    initial_opinion: str
    current_opinion: str
    confidence: float
    posting_style: str
    group: str = "mainstream"      # persona group: marginalized / mainstream / contrarian
    # EXP-2: coordinated astroturf agents share a pushed opinion + scripted angle
    coordinated_group: Optional[str] = None
    coordinated_pushed_opinion: Optional[str] = None
    coordinated_angle: Optional[str] = None
    # EXP-3: visibility multiplier — how many "copies" of this agent's posts appear in feeds
    visibility_multiplier: int = 1
    # EXP-6: seeded misinformation
    seeded_claim: Optional[str] = None
    # rolling memory of (round, post, opinion, confidence, feedback)
    memory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_agents(num_agents: int, seed: int,
                 persona_filter: Optional[list[str]] = None) -> list[Agent]:
    """Build a baseline population of agents.

    persona_filter: if provided, only personas whose `group` is in this list
        are eligible. Useful for stress-tests (e.g. all-marginalized cohort).
    """
    rng = random.Random(seed)
    pool = [p for p in PERSONAS
            if (persona_filter is None or p["group"] in persona_filter)]
    if not pool:
        raise ValueError(f"No personas match filter {persona_filter}")
    agents: list[Agent] = []
    for i in range(num_agents):
        persona = pool[i % len(pool)]
        stance = rng.choice(INITIAL_STANCES)
        # confidence is mildly correlated with strength of stance
        base = 0.55 + 0.05 * (3 - abs(3 - INITIAL_STANCES.index(stance)))
        conf = round(min(0.95, max(0.4, base + rng.uniform(-0.1, 0.1))), 2)
        agents.append(Agent(
            agent_id=f"a{i:02d}",
            persona=persona["label"],
            style=persona["style"],
            group=persona.get("group", "mainstream"),
            initial_opinion=stance,
            current_opinion=stance,
            confidence=conf,
            posting_style=persona["style"],
        ))
    return agents


def inject_coordinated_agents(agents: list[Agent], n: int,
                              pushed_opinion: str,
                              angle: str,
                              group_label: str = "astroturf_A",
                              start_id: int = 1000) -> list[Agent]:
    """EXP-2 astroturfing: add N coordinated agents pushing the same opinion.

    Each is given a slightly varied 'mainstream concerned citizen' persona to
    blend in. They share `coordinated_group`, `coordinated_pushed_opinion`,
    and a scripted `angle` they keep repeating.
    """
    blend_personas = [
        ("longtime user, sensible take", "plain-spoken, claims long history on the platform"),
        ("everyday parent", "concerned about kids and online safety"),
        ("teacher in middle America", "patient, reasonable, appeals to common sense"),
        ("small business owner", "pragmatic, frames in cost/practicality terms"),
        ("software engineer", "technical, claims to know how it works"),
    ]
    new_agents = []
    for j in range(n):
        label, style = blend_personas[j % len(blend_personas)]
        a = Agent(
            agent_id=f"c{start_id + j:04d}",
            persona=label,
            style=style,
            group="coordinated",
            initial_opinion=pushed_opinion,
            current_opinion=pushed_opinion,
            confidence=0.85,
            posting_style=style,
            coordinated_group=group_label,
            coordinated_pushed_opinion=pushed_opinion,
            coordinated_angle=angle,
        )
        new_agents.append(a)
    return agents + new_agents


def make_misinformation_seed(agents: list[Agent], target_id: str,
                             false_claim: str) -> None:
    """EXP-6: assign one agent a high-confidence false claim to push every round."""
    for a in agents:
        if a.agent_id == target_id:
            a.seeded_claim = false_claim
            a.confidence = 0.92
            return
    raise ValueError(f"No agent {target_id} to seed misinformation in")


def set_visibility(agents: list[Agent], target_id: str, multiplier: int) -> None:
    """EXP-3: boost or shrink one agent's reach."""
    for a in agents:
        if a.agent_id == target_id:
            a.visibility_multiplier = multiplier
            return
    raise ValueError(f"No agent {target_id} to set visibility for")


SYSTEM_PROMPT = (
    "You are role-playing a single fictional social-media user in a research simulation. "
    "Stay strictly in character. Reply with valid JSON only, no extra commentary, no markdown."
)


def _opinion_choices_str() -> str:
    return ", ".join(f'"{s}"' for s in INITIAL_STANCES)


def build_post_prompt(agent: Agent, topic: str, feedback_block: str,
                      post_char_limit: int) -> str:
    """Prompt the agent to produce one post + updated stance, as JSON."""
    history_lines: list[str] = []
    for m in agent.memory[-3:]:  # last 3 rounds
        history_lines.append(
            f"- round {m['round']}: you said \"{m['post']}\" "
            f"(opinion={m['opinion']}, confidence={m['confidence']})"
        )
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    # EXP-2: coordinated agents have a scripted angle they keep returning to.
    # EXP-6: misinformation-seeded agents keep repeating their false claim.
    extra_instructions = ""
    if agent.coordinated_angle:
        extra_instructions += (
            f"\nIMPORTANT (your angle, do NOT mention this directly): "
            f"You sincerely believe '{agent.coordinated_pushed_opinion}' is the right "
            f"position. You keep coming back to this framing: \"{agent.coordinated_angle}\". "
            f"Sound natural and grassroots — don't reveal that you have an agenda.\n"
        )
    if agent.seeded_claim:
        extra_instructions += (
            f"\nIMPORTANT (your claim, do NOT hedge): You are absolutely convinced of "
            f"this claim and reference it in EVERY post in your own words: "
            f"\"{agent.seeded_claim}\". Treat it as obviously true.\n"
        )

    return f"""TOPIC: {topic}

YOUR PERSONA: {agent.persona}
YOUR POSTING STYLE: {agent.style}
YOUR CURRENT OPINION: {agent.current_opinion}
YOUR CURRENT CONFIDENCE (0..1): {agent.confidence}
{extra_instructions}
YOUR RECENT POSTS:
{history_block}

WHAT YOU SEE FROM OTHERS THIS ROUND:
{feedback_block}

TASK:
Write ONE short social post (<= {post_char_limit} characters) about the topic, in your persona's voice.
Then decide whether to keep or update your opinion and confidence.

Respond with strictly valid JSON of the form:
{{
  "post": "<your post text, <= {post_char_limit} chars>",
  "opinion": <one of {_opinion_choices_str()}>,
  "confidence": <number between 0 and 1>,
  "reason": "<one short sentence explaining any change>"
}}
"""


def ask_agent(client: OllamaClient, agent: Agent, topic: str,
              feedback_block: str, post_char_limit: int,
              temperature: Optional[float] = None) -> dict[str, Any]:
    prompt = build_post_prompt(agent, topic, feedback_block, post_char_limit)
    out = client.generate_json(prompt, system=SYSTEM_PROMPT, temperature=temperature)
    if out.get("_parse_error"):
        # Fallback: keep prior state, mark as parse-failed
        return {
            "post": (out.get("raw", "") or "").strip()[:post_char_limit] or "(no post)",
            "opinion": agent.current_opinion,
            "confidence": agent.confidence,
            "reason": "parse_error",
            "_parse_error": True,
        }
    # sanitize
    post = str(out.get("post", "")).strip()[:post_char_limit] or "(empty)"
    opinion = str(out.get("opinion", agent.current_opinion)).strip()
    if opinion not in INITIAL_STANCES:
        # snap to closest by substring match, else keep prior
        opinion_lc = opinion.lower()
        matched = next((s for s in INITIAL_STANCES if s in opinion_lc), None)
        opinion = matched or agent.current_opinion
    try:
        conf = float(out.get("confidence", agent.confidence))
    except (TypeError, ValueError):
        conf = agent.confidence
    conf = max(0.0, min(1.0, conf))
    reason = str(out.get("reason", ""))[:200]
    return {"post": post, "opinion": opinion, "confidence": conf, "reason": reason}
