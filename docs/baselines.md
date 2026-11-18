# Supervised baselines (Phase 2a)

These set the numbers the RL models must beat. Models: logistic regression
(`logreg`, linear sanity check), random forest (`rf`) and histogram gradient
boosting (`hgb`, usually the strongest on tabular flow features).

## Protocol
1. Fit on a category-capped sample of the source train rows (default 150k rows per
   category: benign and each attack group; small groups are kept whole).
2. Choose the decision threshold on a source validation sample so that benign
   FPR is 1%. No target data is used for fitting or thresholding.
3. Evaluate on every protocol pair of that source: in-dataset on the source's test
   rows, transfer on the whole target dataset, with the source scaler.

## Running

```bash
# development pass: one seed, evaluation thinned to every 10th row
uv run python scripts/run_baseline.py --models hgb --sources ton --seeds 0 --row-stride 10

# reported numbers: all models, all sources, 5 seeds, full evaluation
uv run python scripts/run_baseline.py --models logreg rf hgb --sources unsw cic ton --seeds 0 1 2 3 4

# aggregate into source-by-target matrices (mean ± std over seeds)
uv run python scripts/make_report.py     # writes experiments/results/baselines.md
```

Each run writes `experiments/runs/<timestamp>_baseline-<model>-<source>_s<seed>/`
with `config.yaml`, `provenance.yaml` and `results.json`.

Cost notes: evaluation dominates. Transfer from a source to ToN or CIC scores
20-27M rows per pair; random forest prediction is the slowest. Start with
`hgb` and `--row-stride 10`, then run the full evaluation once the pipeline looks right.

## Reading the matrices
- Diagonal cells are in-dataset references (optimistic, see ADR 0005).
- Off-diagonal cells are the zero-shot transfer results; the gap between a row's
  diagonal and its off-diagonals is the generalization gap the thesis targets.
- BoT is 99.7% attacks: look at TPR and FPR there, not macro-F1.

## Feature-group ablations (shortcut diagnosis)

Groups (defined a priori in `nidsrl.models.features`): volume, timing, pkt_size,
retrans, host_fingerprint (TTL, TCP window sizes), tcp_flags, proto.

```bash
# drop one group at a time (Git Bash)
for g in volume timing pkt_size retrans host_fingerprint tcp_flags proto; do
  uv run python scripts/run_baseline.py --models hgb --sources ton --seeds 0 --row-stride 10 --drop-groups $g
done
uv run python scripts/make_report.py     # variants appear as hgb-no_<group>
```

Integrity rules:
1. Report every ablation, not just the one that helps.
2. Never choose the main model's feature policy by looking at the performance on a
   target dataset. The principled route is leave-one-dataset-out validation among
   the sources, with BoT-IoT kept as a final untouched check.
3. A large jump in transfer AUC after removing a group is evidence of shortcut
   learning; it is a finding for the paper, not a tuning knob.
