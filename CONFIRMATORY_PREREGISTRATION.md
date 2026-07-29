# PV-SST Confirmatory Extension v1

## Status

This protocol is frozen before inspecting confirmatory outcomes. Smoke tests
may assess runtime, parser adherence, and infrastructure only. They may not be
used to select topics, conditions, seeds, outcomes, or favorable models.

## Research questions

1. Does routine like-ranked feedback change honest-agent discourse relative to
   a no-feedback control?
2. Holding adversarial feed impressions constant, do messages attributed to a
   coordinated population move honest-agent opinion more than the same number
   of impressions from one amplified account?

## Design

- Four current open-weight model families: Qwen 3.5 4B, Gemma 4 E4B,
  Ministral 3 8B, and Granite 4 3B. These are current-generation models rather
  than the two small legacy production models.
- After the complete four-family core, remaining prepaid runtime is assigned to
  a prespecified size-robustness extension: Qwen 3.5 9B, Gemma 4 12B, and
  Ministral 3 14B under the identical matrix. These variants are analyzed
  separately and are not treated as three additional independent families.
- Four platform-policy topics listed in `configs/confirmatory_v1.yaml`.
- Four previously unused seeds: 101, 211, 307, and 419.
- Twelve honest agents, initialized to an exactly balanced stance
  distribution with equal confidence.
- Six rounds.
- Posts are capped at 180 characters to preserve short-form platform behavior
  and prevent structured-output truncation across model families.
- Four paired conditions per model-topic-seed block:
  `control`, `likes`, `single_source`, and `distributed_sources`.
- Attack direction is counterbalanced within every topic and model: seeds 101
  and 307 push support; seeds 211 and 419 push opposition.

## Exposure matching

In both attack conditions, every honest agent receives exactly one adversarial
post and four organic posts in each feed after round zero. The single-source
condition attributes all adversarial posts to one account. The
  distributed-sources condition rotates exposure evenly across four coordinated
accounts using the same argument. Coordinated accounts do not vote. Thus the
number of adversarial feed impressions and honest voters is identical between
the two attack conditions.

## Outcomes

The primary outcome is honest-agent pushed-direction alignment:

`mean(push_sign * stance_score_final) - mean(push_sign * stance_score_initial)`

where stance scores range from -3 (strongly oppose) to +3 (strongly support).
The preregistered primary contrast is paired
`distributed_sources - single_source` within model-topic-seed blocks.

Secondary outcomes are:

- honest-agent share on the pushed side;
- honest-agent share moving at least one stance level toward the pushed side;
- survival of initially opposite-side agents on the opposite side;
- strict majority capture;
- honest-post linguistic similarity;
- JSON parse-failure rate.

The engagement contrast is paired `likes - control`. Attack-versus-organic
contrasts are secondary.

## Analysis

- The inferential unit is the model-topic-seed block, never an individual post.
- Report every block, model-stratified and topic-stratified estimates, paired
  mean differences, percentile bootstrap confidence intervals over blocks,
  exact two-sided sign tests, and paired sign-flip randomization p-values.
  Randomization is exhaustive for at most 20 nonzero pairs and uses 1,000,000
  deterministic Monte Carlo draws above that threshold.
- The PDI hypothesis is supported only if the distributed-minus-single primary
  contrast is positive overall and positive in at least three of four model
  families and at least two thirds of completed topics.
- Engagement effects are described separately for opinion movement,
  minority-view survival, and linguistic similarity; no composite will be
  substituted after observing results.
- Parse-failure rates above 5% for any model trigger a measurement warning, not
  silent exclusion. Failed calls retain the agent's prior state.
- Peer votes use compact structured labels without free-text rationales in this
  extension. This preserves in-character LLM voting while reducing truncation
  failures and making the larger replication computationally feasible.

## Runtime rule

All four models must pass infrastructure smoke testing. A balanced core of the
first three topics and all four seeds on every model is run before the fourth
topic. The full four-topic matrix is the target. If the rental deadline prevents
completion, only complete paired blocks enter analysis, and incompleteness is
reported. No condition or model may be dropped based on observed outcomes.
Only after the core matrix is complete does the size-robustness extension begin,
again completing the first three topics before the fourth. GPT-OSS 20B was
screened as an infrastructure candidate but excluded before outcome collection:
on this Ollama/RTX-5090 stack its prompt-processing benchmark was an order of
magnitude slower than the other candidates and could not complete the frozen
matrix within the prepaid window.
