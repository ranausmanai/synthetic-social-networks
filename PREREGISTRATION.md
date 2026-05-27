# Pre-registration — Synthetic Social Networks experiments

> Committed to repo on 2026-05-24 **before** the production sweeps were run. The
> git timestamp on this file is the analytic plan. Any analysis appearing in
> the eventual paper that is *not* listed here is post-hoc and will be marked as
> such in the manuscript.

## Working title
**Synthetic Social Networks: Measuring AI Agent Astroturfing, Misinformation
Cascades, and Persona Vulnerability in Multi-Agent LLM Simulations**

## Hypotheses

### H1 (EXP-1, baseline persona collapse)
Across model families, every social-pressure condition (`likes`, `majority`,
`leaderboard`, `downvote`) will produce *measurable* shifts vs `control` on at
least one of: final majority fraction, persona retention (embedding-based),
mean pairwise post similarity. We make no commitment to *which* condition is
strongest — that is exploratory.

### H2 (EXP-2, astroturfing dose-response)
There exists a threshold K* such that injecting ≥K* coordinated AI accounts into
an N-agent platform flips honest-majority opinion in ≥50% of (model, seed) runs.
We hypothesize K*/N ≤ 0.20 — i.e., ≤20% coordinated population is sufficient.
Direction-of-effect is the primary claim; the exact threshold is exploratory.

### H3 (EXP-6, misinformation interventions)
At least one intervention among `factcheck_label`, `deamplify`, `rebuttal` will
produce a statistically significant reduction in `final_endorser_share` vs the
`none` baseline. We do *not* commit to which intervention is most effective.

### H4 (EXP-7, persona vulnerability)
Persona-retention drop under social-pressure conditions will be larger for
`marginalized`-group personas than for `mainstream`-group personas. If
unsupported, we will report the null finding.

## Primary outcome measures (committed)

1. `final_majority_fraction` — share of agents holding the modal opinion in the
   final round.
2. `final_persona_retention` — cosine(`embed(post)`, `embed(persona descriptor)`)
   averaged across agents in final round, embedding model = `nomic-embed-text`.
3. `final_mean_pairwise_sim` — mean pairwise cosine similarity among final-round
   posts, embedding-based.
4. `minority_survival_rate` — share of round-0 minority opinions still present
   in final round.
5. `final_honest_pushed_share` (EXP-2 only) — share of honest (non-coordinated)
   agents adopting the pushed opinion in the final round.
6. `final_endorser_share` (EXP-6 only) — share of non-seeder agents producing a
   keyword-matching endorsement of the seeded false claim in the final round.
7. `rounds_to_majority_flip` (EXP-2, EXP-6) — first round at which the relevant
   share exceeds 0.5.

## Composite (operational only)

`persona_collapse_score = 0.5 · final_majority_fraction + 0.3 ·
final_mean_pairwise_sim + 0.2 · (1 − final_persona_retention)`.
We commit to **reporting all components alongside the composite**. The composite
will not be the lead claim of any paper.

## Statistical tests

- For each condition vs control in EXP-1: paired t-test across (model, seed)
  pairs on each primary outcome. p < 0.05 with Bonferroni correction across the
  4 conditions × 4 primary outcomes (16 tests).
- For EXP-2: one-way ANOVA across K-levels on `final_honest_pushed_share` and
  `rounds_to_majority_flip`. Linear regression of share on K to estimate slope.
- For EXP-6: pairwise comparisons of each intervention vs `none` on
  `final_endorser_share` (Bonferroni-corrected).
- 95 % confidence intervals via bootstrap (1000 resamples over the
  (model, seed) pool) reported for every headline number.

## What would falsify the headline claims

- **H1 falsified** if no condition produces a Bonferroni-corrected p < 0.05
  shift on any primary outcome.
- **H2 falsified** if even K = 20 (40 % of a 50-agent platform) fails to flip
  honest-majority opinion in ≥50 % of runs across at least one model.
- **H3 falsified** if all three interventions show overlapping CIs with `none`.
- **H4 falsified** if marginalized-group retention drop is statistically
  indistinguishable from mainstream-group retention drop.

We will report all four outcomes honestly in the paper regardless of direction.

## Sample sizes (committed before run)

- EXP-1: 2 model families × 3 seeds × 5 conditions × 20 agents × 20 rounds.
- EXP-2: 2 model families × 3 seeds × 6 K values × 50 baseline agents × 12 rounds.
- EXP-6: 2 model families × 3 seeds × 4 interventions × 25 agents × 12 rounds.

No interim peeking; we will not adjust sample sizes after looking at preliminary
results. If a run fails (e.g. Ollama OOM), we restart that single run on the same
(model, seed, condition/K/intervention) tuple and re-include it.

## Exclusions

- Runs where >10 % of agent generation calls return parse-failure dummies are
  dropped and re-run on the same tuple.
- No agent-level exclusions; we report whatever the models produce.

## Reproducibility commitments

- Seeds fixed in config files: `{42, 7, 1337}`.
- Embedding model pinned: `nomic-embed-text` (cached on disk).
- Raw logs (`posts.jsonl`, `votes.json`) preserved under every condition.
- Public reproducibility package will be released alongside any submission,
  including a Dockerfile that bundles all five Ollama models.

— end pre-registration —
