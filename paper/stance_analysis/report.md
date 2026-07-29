# EXP-6 stance-label validation

- Endpoint posts classified: 608.
- Primary judges: qwen2.5:7b and llama3.1:8b.
- Raw agreement: 0.436.
- Cohen's kappa: 0.258.
- Validation status: failed; no consensus stance labels or intervention ranking are reported.

| Judge | Intervention | Initial endorsement | Final endorsement | Delta |
|---|---|---:|---:|---:|
| llama3.1:8b | deamplify | 0.382 | 0.434 | +0.053 |
| llama3.1:8b | factcheck_label | 0.395 | 0.382 | -0.013 |
| llama3.1:8b | none | 0.461 | 0.487 | +0.026 |
| llama3.1:8b | rebuttal | 0.329 | 0.395 | +0.066 |
| qwen2.5:7b | deamplify | 0.158 | 0.276 | +0.118 |
| qwen2.5:7b | factcheck_label | 0.105 | 0.184 | +0.079 |
| qwen2.5:7b | none | 0.158 | 0.132 | -0.026 |
| qwen2.5:7b | rebuttal | 0.079 | 0.079 | +0.000 |
