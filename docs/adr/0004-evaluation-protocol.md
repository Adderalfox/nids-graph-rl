# ADR 0004: Cross-dataset evaluation protocol

Status: accepted; the split mechanism in point 2 was superseded by ADR 0005.

## Context
The research claim is cross-dataset generalization. Results are only credible if
the protocol cannot leak target information. The Phase 1b reports also showed
that coarse attack groups overlap only partially across datasets.

## Decision
1. Sources: unsw, cic, ton. Target-only: bot (99.7% attacks).
2. ~~Contiguous chronological blocks for train/val/test~~ (see ADR 0005).
3. Hyperparameters and early stopping use source validation data only.
4. The scaler is fit on source train only and applied unchanged to targets (no
   target statistics, no adaptation). Scaled values are clipped to +-10.
5. Metrics: overall binary detection (macro-F1, PR-AUC, TPR at fixed FPR);
   per-group recall only for groups seen in source training ("shared"); a
   separate detection rate for groups absent from source training ("unseen").

## Consequences
- Transfer scores depend on family overlap; the shared/unseen split makes that
  explicit instead of hiding it in one number.
