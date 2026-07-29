# Confirmatory artifact audit

The audit was run after both frozen matrices completed and before manuscript
revision.

| Check | Core | Size extension |
|---|---:|---:|
| Trials | 256/256 | 192/192 |
| Complete four-condition blocks | 64/64 | 48/48 |
| Exposure-audit failures | 0 | 0 |
| Raw post files | 256 | 192 |
| Raw vote files | 256 | 192 |
| Posts | 20,352 | 15,264 |
| Honest-agent voter calls | 18,432 | 13,824 |
| Post parse failures | 53 (0.26%) | 30 (0.20%) |
| Vote parse failures | 140 (0.76%) | 145 (1.05%) |

Both result sets record the SHA-256 hash
`53f6778d07f8e0a526de5e8f182d4c6fda3218b8c28e707bb4e744a3e6fa04d0`,
which matches `CONFIRMATORY_PREREGISTRATION.md`.

The preregistered analysis is reproduced with:

```bash
python -m src.analyze_confirmatory runs_confirmatory/v1_core
python -m src.analyze_confirmatory runs_confirmatory/v1_size_extension
```

Headline outputs and all model/topic strata are stored in each run root as
`confirmatory_analysis.md` and `confirmatory_analysis.csv`.
