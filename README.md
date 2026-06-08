# Synthetic Social Networks

A reproducible, peer-voted LLM-agent social-platform testbed and accompanying
multi-vector vulnerability study.

> *On a platform of LLM agents, design persuades before any actor speaks,
> a coordinated crowd carries opinion further than any single amplified
> voice, and a paraphrase slips past every filter built to catch the words
> it has changed. Our agents were more stubborn than humans, and still moved.*

## What's here

This repository contains the code, configurations, paper, and small per-run
summary CSVs for the *Synthetic Social Networks* study. The full raw dataset
(28,946 in-character agent posts with paired peer-vote traces) is released
separately on Hugging Face — see "Dataset" below.

### Quick map

| Path | What |
|---|---|
| `paper/paper.md`, `paper/paper.pdf` | The manuscript (markdown source + rendered PDF) |
| `paper/figures/` | Seven publication figures, all generated from raw data |
| `paper/make_figures.py` | Script to regenerate every figure from CSVs |
| `src/` | Experimental pipeline: agents, environment, metrics, peer voting, embeddings, experiment runners |
| `configs/` | YAML configuration files for each experiment |
| `PREREGISTRATION.md` | Analytic plan committed before any production run |
| `FRAMING.md` | Explicit "what we claim / don't claim" doc |
| `RUNNING.md` | How to reproduce experiments on your own GPU |
| `launch_overnight.sh`, `launch_calibration.sh`, `watchdog.sh` | Pod-side orchestration scripts |
| `requirements.txt` | Python dependencies |

## TL;DR of the study

Four threat-model probes on a simulated social platform whose users are LLM
agents (qwen3.5:2b and llama3.2:3b, peer-voted feedback, embedding-based
metrics, preregistered analysis plan):

1. **UX-driven diversity loss (EXP-1).** Standard engagement features (likes,
   visible majority, leaderboards, downvotes) reduce minority-view survival by
   6–16 pp vs a no-signal control — *with no adversarial actor present*.
2. **Population-Driven Influence (PDI) hypothesis (EXP-2 vs EXP-3).**
   Coordinated AI populations produce a 50% strict opinion-flip rate at
   K=20 (40% of a 50-agent platform); a single amplified account at 1× to
   20× visibility shows no coherent dose-response. *Coordination, not
   amplification, moves opinion in our simulation.*
3. **Paraphrase leakage (EXP-6).** Keyword-based misinformation defenses
   miss thematic propagation because LLM agents paraphrase claims rather
   than repeating them verbatim. Current misinformation measurement
   methods for LLM-agent systems are broken in a specific, named way.
4. **Calibration anchor A1.** Our agents conform to one-shot bandwagon
   signals at 0.039 weighted pooled — well below the 10–20% documented
   human range. The fact that effects (1)–(3) emerge despite this
   stubbornness makes the directional findings more concerning, not less.

Full details: `paper/paper.pdf`.

## Dataset

The complete raw experimental outputs — every in-character agent post, every
peer vote with rationale, all per-trial metrics, all per-experiment configs —
are released on Hugging Face:

**🤗 [`ranausmans/synthetic-social-networks`](https://huggingface.co/datasets/ranausmans/synthetic-social-networks)**

- 24,160 production posts from the four reported experiments
- 28,946 posts total when pipeline-verification smoke runs are included
- 80 production vote-log files (one per condition × model × seed)
- Per-trial scalar metrics + per-round trajectories
- License: CC-BY 4.0

To recreate the local `runs/` directory:

```bash
pip install huggingface_hub
huggingface-cli download ranausmans/synthetic-social-networks --repo-type dataset --local-dir runs/
```

## Reproducing experiments

See [`RUNNING.md`](RUNNING.md) for end-to-end instructions. Total compute for
the production experiments was ~57 GPU-hours on a single NVIDIA RTX 4000 Ada
Generation workstation (~$26 in cloud compute).

## Reproducing the paper

```bash
python paper/make_figures.py    # regenerate figures from CSVs
cd paper && pandoc paper.md \
  --pdf-engine=tectonic --resource-path=. \
  -V geometry:margin=0.9in -V fontsize=11pt -V colorlinks=true \
  -o paper.pdf
```

## Citing

> Usman, R. M. (2026). *Engagement Signals and Coordinated Crowds Reshape
> LLM-Agent Discourse.* (Working paper.)

BibTeX:

```bibtex
@misc{usman2026synthetic,
  author = {Usman, Rana Muhammad},
  title  = {Engagement Signals and Coordinated Crowds Reshape
            LLM-Agent Discourse},
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

Issues and pull requests welcome. If you reproduce, extend, or falsify any
of the findings — particularly the Population-Driven Influence hypothesis
or the paraphrase-leakage observation — please open an issue or get in
touch directly.
