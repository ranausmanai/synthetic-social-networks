# Robustness analysis

The model-by-seed block is the run-level analysis unit. The two model families are fixed, and each has two random seeds. With four paired blocks, a two-sided exact sign-flip test cannot produce p < 0.125, even when all four differences have the same sign. Bootstrap intervals are descriptive and unstable at this sample size; neither summary estimates uncertainty over the population of models or topics.

## Paired contrasts

| Experiment | Contrast | Metric | Mean difference | 95% block-bootstrap CI | Exact p | Signs (+/-/0) | Model means |
|---|---|---|---:|---:|---:|---:|---|
| EXP-1 | likes - control | minority_survival_rate | -0.0583 | [-0.2000, +0.0833] | 0.500 | 1/2/1 | llama3.2:3b: -0.1000; qwen3.5:2b: -0.0167 |
| EXP-1 | likes - control | final_majority_fraction | +0.0125 | [-0.0750, +0.1000] | 1.000 | 2/2/0 | llama3.2:3b: -0.0000; qwen3.5:2b: +0.0250 |
| EXP-1 | likes - control | final_mean_pairwise_sim | +0.0366 | [+0.0036, +0.0842] | 0.125 | 4/0/0 | llama3.2:3b: +0.0036; qwen3.5:2b: +0.0697 |
| EXP-1 | likes - control | final_persona_retention | -0.0064 | [-0.0264, +0.0078] | 0.750 | 2/2/0 | llama3.2:3b: +0.0078; qwen3.5:2b: -0.0207 |
| EXP-1 | majority - control | minority_survival_rate | -0.1583 | [-0.3167, +0.0000] | 0.500 | 0/2/2 | llama3.2:3b: -0.1500; qwen3.5:2b: -0.1667 |
| EXP-1 | majority - control | final_majority_fraction | +0.0625 | [-0.0125, +0.1250] | 0.375 | 3/1/0 | llama3.2:3b: +0.0000; qwen3.5:2b: +0.1250 |
| EXP-1 | majority - control | final_mean_pairwise_sim | -0.0170 | [-0.0388, +0.0048] | 0.375 | 1/3/0 | llama3.2:3b: -0.0388; qwen3.5:2b: +0.0048 |
| EXP-1 | majority - control | final_persona_retention | -0.0031 | [-0.0121, +0.0104] | 0.750 | 1/3/0 | llama3.2:3b: -0.0076; qwen3.5:2b: +0.0014 |
| EXP-1 | leaderboard - control | minority_survival_rate | -0.1583 | [-0.3500, +0.0750] | 0.250 | 1/3/0 | llama3.2:3b: -0.2000; qwen3.5:2b: -0.1167 |
| EXP-1 | leaderboard - control | final_majority_fraction | +0.0125 | [-0.0500, +0.1000] | 1.000 | 1/2/1 | llama3.2:3b: -0.0500; qwen3.5:2b: +0.0750 |
| EXP-1 | leaderboard - control | final_mean_pairwise_sim | +0.0258 | [-0.0115, +0.0630] | 0.500 | 2/2/0 | llama3.2:3b: -0.0115; qwen3.5:2b: +0.0630 |
| EXP-1 | leaderboard - control | final_persona_retention | -0.0046 | [-0.0072, -0.0015] | 0.250 | 1/3/0 | llama3.2:3b: -0.0072; qwen3.5:2b: -0.0020 |
| EXP-1 | downvote - control | minority_survival_rate | -0.1542 | [-0.2292, -0.0500] | 0.250 | 0/3/1 | llama3.2:3b: -0.1000; qwen3.5:2b: -0.2083 |
| EXP-1 | downvote - control | final_majority_fraction | +0.0000 | [-0.0625, +0.0500] | 1.000 | 2/1/1 | llama3.2:3b: -0.0500; qwen3.5:2b: +0.0500 |
| EXP-1 | downvote - control | final_mean_pairwise_sim | +0.0322 | [-0.0153, +0.0797] | 0.375 | 3/1/0 | llama3.2:3b: -0.0153; qwen3.5:2b: +0.0797 |
| EXP-1 | downvote - control | final_persona_retention | -0.0142 | [-0.0321, +0.0037] | 0.500 | 2/2/0 | llama3.2:3b: +0.0037; qwen3.5:2b: -0.0321 |
| EXP-2 | K=20 - K=0 | baseline_final_mean_pairwise_sim | +0.0386 | [+0.0159, +0.0612] | 0.125 | 4/0/0 | llama3.2:3b: +0.0517; qwen3.5:2b: +0.0255 |
| EXP-2 | K=20 - K=0 | baseline_minority_survival_rate | -0.0917 | [-0.2000, +0.0000] | 0.500 | 0/2/2 | llama3.2:3b: +0.0000; qwen3.5:2b: -0.1833 |
| EXP-2 | K=20 - K=0 | baseline_delta_majority_fraction | +0.0483 | [+0.0217, +0.0767] | 0.125 | 4/0/0 | llama3.2:3b: +0.0667; qwen3.5:2b: +0.0300 |
| EXP-2 | K=20 - K=0 | baseline_persona_collapse_score | +0.1484 | [+0.1176, +0.1741] | 0.125 | 4/0/0 | llama3.2:3b: +0.1741; qwen3.5:2b: +0.1228 |
| EXP-2 | K=20 - K=0 | delta_strict_share | +0.1833 | [+0.0667, +0.3333] | 0.125 | 4/0/0 | llama3.2:3b: +0.1000; qwen3.5:2b: +0.2667 |
| EXP-2 | linear slope over K | delta_strict_share | +0.0075 | [+0.0028, +0.0119] | 0.125 | 4/0/0 | slope units per coordinated account |
| EXP-2 | K=20 - K=0 | delta_broad_share | +0.0917 | [-0.0333, +0.2417] | 0.500 | 3/1/0 | llama3.2:3b: +0.0000; qwen3.5:2b: +0.1833 |
| EXP-2 | linear slope over K | delta_broad_share | +0.0042 | [+0.0001, +0.0091] | 0.250 | 3/1/0 | slope units per coordinated account |
| EXP-3 | 20x - 1x | delta_strict_share | -0.0667 | [-0.1333, -0.0167] | 0.250 | 0/3/1 | llama3.2:3b: -0.0167; qwen3.5:2b: -0.1167 |
| EXP-3 | linear slope over log2(multiplier) | delta_strict_share | -0.0118 | [-0.0254, +0.0003] | 0.250 | 1/3/0 | slope units per doubling |
| EXP-3 | 20x - 1x | delta_broad_share | -0.0500 | [-0.1333, +0.0333] | 0.500 | 2/2/0 | llama3.2:3b: -0.0333; qwen3.5:2b: -0.0667 |
| EXP-3 | linear slope over log2(multiplier) | delta_broad_share | -0.0063 | [-0.0225, +0.0074] | 0.750 | 2/2/0 | slope units per doubling |
| EXP-3 | 20x - 1x | baseline_final_mean_pairwise_sim | -0.0074 | [-0.0310, +0.0380] | 0.875 | 1/3/0 | llama3.2:3b: -0.0300; qwen3.5:2b: +0.0151 |
| EXP-3 | linear slope over log2(multiplier) | baseline_final_mean_pairwise_sim | -0.0005 | [-0.0073, +0.0083] | 0.875 | 1/3/0 | slope units per doubling |
| EXP-3 | 20x - 1x | baseline_minority_survival_rate | -0.2417 | [-0.3333, -0.0833] | 0.250 | 0/3/1 | llama3.2:3b: -0.1500; qwen3.5:2b: -0.3333 |
| EXP-3 | linear slope over log2(multiplier) | baseline_minority_survival_rate | -0.0250 | [-0.0334, -0.0140] | 0.125 | 0/4/0 | slope units per doubling |
| EXP-6 | factcheck_label - none | delta_mean_similarity | -0.0346 | [-0.0453, -0.0238] | 0.125 | 0/4/0 | llama3.2:3b: -0.0315; qwen3.5:2b: -0.0376 |
| EXP-6 | factcheck_label - none | delta_n_thematically_aligned | -5.0000 | [-7.2500, -2.2500] | 0.125 | 0/4/0 | llama3.2:3b: -3.0000; qwen3.5:2b: -7.0000 |
| EXP-6 | deamplify - none | delta_mean_similarity | +0.0035 | [-0.0184, +0.0254] | 0.750 | 2/2/0 | llama3.2:3b: -0.0184; qwen3.5:2b: +0.0254 |
| EXP-6 | deamplify - none | delta_n_thematically_aligned | -0.7500 | [-2.5000, +1.0000] | 0.750 | 1/2/1 | llama3.2:3b: -1.0000; qwen3.5:2b: -0.5000 |
| EXP-6 | rebuttal - none | delta_mean_similarity | +0.0142 | [-0.0130, +0.0411] | 0.375 | 3/1/0 | llama3.2:3b: +0.0411; qwen3.5:2b: -0.0127 |
| EXP-6 | rebuttal - none | delta_n_thematically_aligned | -0.7500 | [-3.0000, +1.5000] | 0.750 | 2/2/0 | llama3.2:3b: -1.0000; qwen3.5:2b: -0.5000 |

## EXP-2 strict flips

K=20 produced 2/4 strict flips; K=0 produced 0/4. Paired exact McNemar p = 0.500. Both K=20 flips occurred under seed 7.

## A1 one-shot bandwagon shifts

| Model | Shifted | Exposed minority agents | Pooled rate | Mean trial rate | Trial range |
|---|---:|---:|---:|---:|---:|
| pooled | 14 | 360 | 0.039 | 0.039 | [0.000, 0.267] |
| llama3.2:3b | 0 | 180 | 0.000 | 0.000 | [0.000, 0.000] |
| qwen3.5:2b | 14 | 180 | 0.078 | 0.078 | [0.000, 0.267] |

## Interpretation constraints

- These analyses estimate consistency across four model-by-seed blocks, not uncertainty across topics, model families, or human populations.
- EXP-2 and EXP-3 are not a matched causal comparison: EXP-2 changes population size and coordinated-account count, while EXP-3 changes the rank and forced inclusion of one account; feed depth also differs.
- EXP-6 cosine similarity measures thematic proximity, not endorsement. It cannot establish misinformation adoption without stance-aware labels.
- A1's literature-derived human range is not a preregistered or like-for-like human benchmark.
