# Evaluation

## Protocol (see ADR 0004 and ADR 0005)
- Train and tune on source train/val rows only.
- In-dataset reference: the source's own test rows.
- Zero-shot transfer: the whole target dataset, source scaler, no adaptation.

## Metrics (`nidsrl.eval.metrics`)
Accuracy is never reported: prevalence ranges from 5% to 99.7% across datasets.

| Metric | Why |
|--------|-----|
| TPR, FPR, macro-F1, balanced accuracy | insensitive to prevalence |
| ROC-AUC | threshold-free, prevalence-insensitive |
| TPR at FPR 1% and 0.1% | operational: detection at a fixed false-alarm budget |
| PR-AUC (reported with prevalence) | useful but depends on prevalence; never compare across datasets alone |
| Per-group recall (shared groups) | groups seen in source training, at least 500 flows |
| Unseen-group detection rate | groups absent from source training |
| flows per second | inference cost (paper latency table) |

BoT-IoT has only 52k benign flows, so FPR is resolved to about 2e-5 and TPR at
FPR 0.1% rests on about 52 false positives. Interpret with care.

## Plugging in a model (`nidsrl.eval.harness`)

```python
from nidsrl.eval.harness import evaluate_pair
import json

protocol = json.load(open("data/processed/protocol.json"))
def predict(X):            # X: float32 (n, 55), already scaled with the source scaler
    scores = model_score(X)            # continuous attack score
    return (scores > thr).astype("uint8"), scores   # for the RL agent: Q(attack) - Q(benign)

report = evaluate_pair(predict, "data/processed", protocol["pairs"][1])
```

Data access for training uses `nidsrl.data.loading.iter_split(dir, plan, split, scaler)`,
which streams one hourly block at a time (`train`, `val`, `test`, or `all`).

## Statistics (Phase 6)
Report mean and standard deviation over 5 seeds. Per-flow bootstrap intervals are
invalid because flows within a run are correlated; use a block bootstrap over runs.
