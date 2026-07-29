# Synthetic Social Networks

A reproducible, peer-voted LLM-agent social-platform testbed and accompanying
multi-vector vulnerability study.

> This repository is a controlled stress-test artifact, not a predictive
> model of human users or a production social platform. It contains an
> exploratory stage and a separately frozen, preregistered 448-trial
> matched-exposure confirmation.

## What's here

This repository contains the code, configurations, paper, and small per-run
summary CSVs for the *Synthetic Social Networks* study. The artifact now
contains 59,776 production posts across 528 production trials.

### Quick map

| Path | What |
|---|---|
| `paper/paper.md`, `paper/paper.pdf` | The manuscript (markdown source + rendered PDF) |
| `paper/figures/` | Nine publication figures, all generated from raw data |
| `paper/make_figures.py` | Script to regenerate every figure from CSVs |
| `src/` | Experimental pipeline: agents, environment, metrics, peer voting, embeddings, experiment runners |
| `configs/` | YAML configuration files for each experiment |
| `PREREGISTRATION.md` | Analytic plan committed before any production run |
| `CONFIRMATORY_PREREGISTRATION.md` | Frozen matched-exposure extension protocol |
| `runs_confirmatory/` | Raw confirmatory traces, audit metadata, and block-level analysis |
| `FRAMING.md` | Timestamped pre-run framing; the paper's audited claim scope supersedes it |
| `RUNNING.md` | How to reproduce experiments on your own GPU |
| `launch_overnight.sh`, `launch_calibration.sh`, `watchdog.sh` | Pod-side orchestration scripts |
| `requirements.txt` | Python dependencies |

## TL;DR of the study

The primary evidence is a separately frozen 448-trial confirmation spanning
four topics, four unused seeds, four open-weight model families, and three
prespecified larger variants:

1. **Feed-induced lexical convergence.** Relative to a topic-only control, a
   feed of prior peer posts ranked by peer-generated likes increases
   final-post TF-IDF similarity in both model-size panels. The contrast
   bundles peer exposure with ranking and does not identify ranking alone.
2. **No reliable matched-exposure advantage for distributed sources.** Four
   distributed sources do not reliably move stance more than one source when
   adversarial impressions are held fixed. The preregistered
   population-driven-influence criterion fails.
3. **Substantial model heterogeneity.** Minority-view survival falls in the
   four-family core panel but not conclusively in the larger variants. The
   result does not support a model-general suppression law.
4. **Measurement warning from the exploratory stage.** An all-keyword
   detector misses close restatements, while polarity-blind embedding
   similarity conflates endorsement and rebuttal. We call these failures
   paraphrase leakage and stance confounding. Two independent LLM stance
   judges agree on only 43.6% of endpoint labels (kappa=0.258), so exploratory
   misinformation-intervention effectiveness remains unresolved.

The earlier two-model, one-topic stage generated the confirmatory questions
and remains available for provenance. Its small cells and unmatched attack
mechanisms are not used as confirmatory evidence.

Full details: `paper/paper.pdf`.

## Dataset

The complete raw experimental outputs — every in-character agent post, every
peer-vote record, all per-trial metrics, and all per-experiment configs —
are released on Hugging Face:

[`ranausmans/synthetic-social-networks`](https://huggingface.co/datasets/ranausmans/synthetic-social-networks)

- 59,776 production posts from 528 production trials
- 64,562 posts total when the original pipeline-verification runs are included
- 448 new confirmatory vote-log files plus the exploratory-stage traces
- Per-trial scalar metrics + per-round trajectories
- License: CC-BY 4.0

To recreate the local `runs/` directory:

```bash
pip install huggingface_hub
huggingface-cli download ranausmans/synthetic-social-networks --repo-type dataset --local-dir runs/
```

## Reproducing experiments

See [`RUNNING.md`](RUNNING.md) for the exploratory pipeline. The confirmatory
matrix is configured by `configs/confirmatory_v1.yaml` and
`configs/confirmatory_large_v1.yaml` and launched by
`launch_confirmatory.sh`.

## Reproducing the paper

```bash
python src/analyze_robustness.py # regenerate paired exact tests
python src/analyze_misinfo_stance.py # reproduce the cached stance-label audit
python -m src.analyze_confirmatory runs_confirmatory/v1_core
python -m src.analyze_confirmatory runs_confirmatory/v1_size_extension
python -m src.audit_confirmatory      # verify raw files, exposures, hashes, and headlines
python paper/make_figures.py    # regenerate figures from CSVs
cd paper && pandoc paper.md \
  --pdf-engine=tectonic --resource-path=. \
  -V geometry:margin=0.9in -V fontsize=11pt -V colorlinks=true \
  -o paper.pdf
```

## Citing

> Usman, R. M. (2026). *Peer-Voted LLM-Agent Stress Tests Find Feed-Induced
> Lexical Convergence but No Reliable Matched-Exposure Advantage for
> Distributed Sources.* (Working paper.)

BibTeX:

```bibtex
@misc{usman2026synthetic,
  author = {Usman, Rana Muhammad},
  title  = {Peer-Voted LLM-Agent Stress Tests Find Feed-Induced Lexical
            Convergence but No Reliable Matched-Exposure Advantage for
            Distributed Sources},
  year   = {2026},
  url    = {https://github.com/ranausmanai/synthetic-social-networks},
  note   = {Working paper; dataset at
            https://huggingface.co/datasets/ranausmans/synthetic-social-networks}
}
```

## License

- **Code** (this repository): MIT — see [`LICENSE`](LICENSE).
- **Dataset** (on Hugging Face): CC-BY 4.0.

## Contact

Rana Muhammad Usman — `usmanashrafrana@gmail.com`

Issues and pull requests are welcome. Human validation and replications on
non-platform-policy topics are the most useful next extensions.
