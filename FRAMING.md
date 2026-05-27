# Framing — what this paper does and does not claim

> Committed alongside `PREREGISTRATION.md` *before* production sweeps. This is the
> document we will paste into the discussion section verbatim, so a reviewer can
> verify our framing matches our methods.

## The core methodological honesty

This paper studies **populations of LLM agents on a simulated social platform**.
It is **not** a prediction of how human users on Instagram, Snapchat, or Meta
will behave. Anyone reading our findings as literal human-population predictions
is misreading us. We say so explicitly in the abstract, introduction, and
discussion.

## What the sim *can* defensibly tell you

1. **Threat-model probes.** If a coordinated-account attack succeeds in a
   simplified LLM-only system, it warrants investigation in real systems.
   This is the framing DARPA SocialSim, Stanford Internet Observatory, and
   Graphika use for their agent-based modeling work. It is a recognized,
   citable methodology.

2. **A study of a near-future configuration that already exists.** LLM-driven
   accounts are present and growing on every major platform — Character.AI
   companions, Replika, AI-driven engagement bots, automated customer-service
   accounts, AI influencers on TikTok and Instagram. Studies of LLM-only
   populations are studies of a real platform configuration, not science
   fiction.

3. **Comparative interventions.** Even if absolute numbers do not transfer to
   humans, the *ranking* of interventions often does. "Community-note-style
   rebuttals outperform fact-check labels in our sim" is a useful prior for a
   Trust & Safety team's next A/B test, even if the exact effect size differs.

4. **Mechanism isolation.** Real platforms cannot ethically ablate their
   recommender, switch off their badge system, or seed misinformation to
   measure cascade speed. Sims can. This complementary role is why agent
   simulation persists as a methodology alongside, not in opposition to,
   human studies.

## What the sim *cannot* defensibly tell you

1. **Exact transfer to humans.** A coordinated-account ratio that flips
   majority opinion in our sim should be treated as a lower-bound *attack
   feasibility signal*, not a predicted attack threshold on Instagram.
2. **Emotional, identity, social-graph, or off-platform effects.** Humans
   quit apps, block accounts, talk to friends offline, lose jobs over posts.
   None of that is in our sim.
3. **Long-tail dynamics over months or years.** We run 10–20 rounds; real
   social media is months-long feedback loops with offline events.
4. **Network-level cascades through real follow graphs.** Our agents see
   the platform's feed; they do not have personal follow graphs that pre-date
   the simulation.

## Calibration commitments

To earn the right to make any quantitative claim, we run two calibration
experiments **before** drawing conclusions from the headline sweeps:

### A1. Bandwagon-effect anchor (built, runs locally)
We reproduce a documented human result: when 50/50-split subjects are shown
a clear majority signal (~75% claimed agreement), a measurable fraction
conform. Across multiple online studies the per-round shift is in the
**12–18% range** (Salganik et al., MusicLab 2006; Muchnik et al., 2013). If
our sim's bandwagon shift rate falls outside this range, we lose credibility
to make new claims and we say so in the paper.

### A2. ChangeMyView human-replay (built, runs on pod)
We sample real Reddit r/ChangeMyView conversations where the OP did (or did
not) award a delta after being argued with. We initialize a sim agent with
the OP's stance, show our agent the top counter-arguments, and ask it to
update its position. We compare our sim's flip-rate to the real OP's flip
decisions and report accuracy vs three baselines: random (50%), always-flip,
always-stick. If our sim does not beat baseline by a meaningful margin, the
calibration is reported as failed and our headline claims are downgraded
from quantitative to qualitative.

## How findings will be reported

For every headline claim we will say, in this order, in the paper:

1. The *sim* result, with numbers and error bars.
2. The *calibration* result — how well our sim reproduces the matched human
   result. If poorly: "this metric did not calibrate against humans, treat
   the following as qualitative."
3. The *threat-model* implication — what the result means for real-world
   policy, in conditional language ("if calibration holds for this metric,
   then…").
4. The *limitations* paragraph — what we did not measure and why.

This is the same shape the *Generative Agents* paper (Park et al., Stanford
2023) used and has been adopted by follow-on work at Meta AI, Google
Research, and Microsoft Research.

## What would make us retract a claim

- Calibration A1 failing AND we run the headline experiment anyway: that is
  scientific malpractice. We don't.
- Calibration A2 failing on flip-rate prediction (≤ baseline): we retract any
  per-platform-percentage claim and switch to ranking-only language.
- Cross-model effect direction reversing between qwen and llama families:
  we report the divergence and note the result is model-family-specific.

## What we will NOT claim, ever

- "X% of fake AI accounts will flip Instagram." We will say "in a 30-agent
  simulated platform, X coordinated agents shifted majority position in Y% of
  trials; this provides a lower-bound feasibility signal for an attack with
  similar coordination on real platforms."
- "Community notes are the best moderation strategy." We will say "in our
  sim, community-notes-style rebuttals reduced cascade flip-rate more than
  labels or demotion; this provides a hypothesis for Trust & Safety A/B
  testing on real users."
- "Our paper proves LLM agents collapse persona." If the effect is small or
  noisy, we report it as small or noisy. The pre-registration commits us to
  the null direction too.

— end framing doc —
