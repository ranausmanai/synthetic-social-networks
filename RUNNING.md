# Running the experiments

These notes describe how to reproduce the production experiments end-to-end.
The original development used a rented RunPod RTX 4000 Ada (20 GB VRAM)
workstation, but any system with a CUDA-capable GPU (≥10 GB VRAM) and Ollama
installed should work.

## Setup

1. Install [Ollama](https://ollama.com) and pull the required models:
   ```bash
   ollama pull qwen3.5:2b
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
2. Start Ollama: `ollama serve` (this runs on `127.0.0.1:11434` by default)
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Single experiment

```bash
# EXP-1 baseline (~10 hr on RTX 4000 Ada)
python -m src.run_experiment --sweep --config configs/exp1_baseline.yaml --tag exp1

# EXP-2 astroturfing (~15 hr)
python -m src.run_experiment --exp astroturf --config configs/exp2_astroturf.yaml --tag exp2

# EXP-3 influence amplification (~15 hr)
python -m src.run_experiment --exp influence --config configs/exp3_influence.yaml --tag exp3

# EXP-6 misinformation interventions (~9 hr)
python -m src.run_experiment --exp misinfo --config configs/exp6_misinfo.yaml --tag exp6

# A1 bandwagon calibration anchor (~40 min)
python -m src.run_experiment --exp anchor --config configs/anchor.yaml --tag a1

# A2 ChangeMyView human-replay (~5 min on synthetic dataset; production
# requires populating cmv_data_path in configs/human_replay.yaml with a
# real CMV JSONL dump)
python -m src.run_experiment --exp human_replay --config configs/human_replay.yaml --tag a2
```

## All experiments back-to-back

```bash
bash launch_overnight.sh
```

This chains all four production experiments in order. Approximate total
runtime: ~50 hours on a single RTX 4000 Ada.

## Outputs

Each run writes to `runs/<timestamp>_<exp>_<tag>/`:
- `config.json` — the exact configuration used
- `<exp>_results.csv` — every (model, seed, level) as a row
- `<exp>_aggregated.csv` — mean ± std across seeds
- `report.md` — human-readable summary
- `plots/` — auto-generated figures
- `<run>/likes/logs/posts.jsonl` + `votes.json` — raw per-agent posts and peer votes

## Reproducing the paper figures

After running the experiments (or after downloading the dataset from
Hugging Face), regenerate all paper figures with:

```bash
python paper/make_figures.py
```

Figures land in `paper/figures/`.

## Reproducing the paper PDF

The paper is written in Markdown at `paper/paper.md` and rendered via
pandoc + tectonic. Build script:

```bash
cd paper
pandoc paper.md \
  --pdf-engine=tectonic --resource-path=. \
  -V geometry:margin=0.9in -V fontsize=11pt -V colorlinks=true \
  -o paper.pdf
```
