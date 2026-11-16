# nids-graph-rl

Cross-dataset generalizable network intrusion detection with graph-based deep
reinforcement learning.

**Research question:** does a windowed host/flow graph encoder trained with a
Dueling Double DQN and topology-aware evolutionary experience replay degrade
less than baselines when trained on one dataset and tested zero-shot on others?

> Status: **Phase 0 (foundations)**. Results tables and figures will appear here
> as phases complete. See `docs/PHASES.md` for the roadmap and decision log.

## Setup (Windows, Git Bash)

Requires [uv](https://docs.astral.sh/uv/). Python is managed by uv.

```bash
# 1. Install the pinned Python (reads .python-version)
uv python install

# 2. Create .venv and install everything, including the GPU build of PyTorch
uv sync --extra gpu

# 3. Verify
uv run python scripts/check_env.py
uv run pytest
uv run ruff check src tests scripts
```

`check_env.py` should report your GPU (RTX 3050 Ti) and `seeding : OK`.
Commit `uv.lock` so every machine resolves identical versions.
Add a dependency with `uv add <package>`, and a dev tool with `uv add --dev <package>`.
Plain `uv sync` (no extra) installs without torch, which is what CI does.

## Repository layout

```
configs/      YAML configs (one per experiment, later phases)
src/nidsrl/   the package: data, graph, models, rl, eval, utils
scripts/      entry points (check_env.py now; train/evaluate later)
tests/        unit tests (seeding, config, run dirs; leakage/env/graph later)
experiments/  run outputs (git-ignored) and results summaries
docs/         roadmap, decision log (ADRs)
paper/        LaTeX source
data/         datasets (git-ignored; see data/README.md)
```

## Reproducibility rules

1. Every experiment is driven by a YAML config plus dotted overrides
   (`project.seed=1`). Unknown keys raise errors.
2. Every run writes its resolved config and git commit into
   `experiments/runs/<timestamp>_<tag>_s<seed>/`.
3. Reported numbers are mean ± std over 5 seeds.
4. Scalers and any learned preprocessing are fit on training data only.

