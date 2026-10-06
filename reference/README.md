# reference/

External sources kept for citation/comparison while building this project.
Not built on top of, not modified — read-only reference material.

## gepa-ai-gepa

Git submodule pointing directly at the official upstream GEPA implementation:
[gepa-ai/gepa](https://github.com/gepa-ai/gepa) ([paper: arXiv:2507.19457](https://arxiv.org/abs/2507.19457), MIT license).

Pinned at commit `fb1ed589fd83372caef499cffc2c73173d3b096b` (2026-10-02).

This is the real engine our `src/gepa_optimizer` package depends on as a pip
package (`gepa>=0.1.4`) and whose adapter contract (`GEPAAdapter`,
`EvaluationBatch`, `reflection_lm`/`task_lm` resolution) our code was written
against. Kept here, pinned, so the exact source used during development is
always reproducible — not because our code vendors or forks its logic.

To populate it after cloning this repo:

```bash
git submodule update --init --recursive
```
