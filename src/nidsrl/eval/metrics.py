"""Evaluation metrics for cross-dataset intrusion detection.

Design rules:
- Prevalence differs enormously across datasets (5% to 99.7% attacks), so accuracy
  is never reported. Headline numbers are TPR, FPR, macro-F1, ROC-AUC and TPR at
  fixed FPR, all of which are insensitive to class prevalence. PR-AUC is reported
  together with the prevalence it depends on.
- Per-group recall is reported only for attack groups with enough flows, split into
  groups seen in source training ("shared") and groups never seen ("unseen").
- Everything is exact and numpy-only (no sklearn needed); tests cross-check against
  sklearn when it is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nidsrl.data.taxonomy import BENIGN, GROUPS

FPR_TARGETS = (0.01, 0.001)


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Confusion-matrix metrics for hard predictions (1 = attack)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    precision = _safe_div(tp, tp + fp)
    f1_attack = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_benign = _safe_div(2 * tn, 2 * tn + fn + fp)
    macro_f1 = None if f1_attack is None or f1_benign is None else (f1_attack + f1_benign) / 2
    tnr = None if fpr is None else 1.0 - fpr
    balanced = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1_attack": f1_attack,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
    }


def _curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative (tp, fp) at each distinct score threshold, descending."""
    order = np.argsort(-y_score, kind="stable")
    ys = y_true[order].astype(np.int64)
    ss = y_score[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.concatenate([np.nonzero(np.diff(ss))[0], [len(ss) - 1]])
    return tp[last].astype(np.float64), fp[last].astype(np.float64)


def ranking_report(
    y_true: np.ndarray, y_score: np.ndarray, fpr_targets: tuple[float, ...] = FPR_TARGETS
) -> dict[str, Any]:
    """ROC-AUC, average precision (PR-AUC) and TPR at fixed FPR from continuous scores."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    out: dict[str, Any] = {
        "roc_auc": None,
        "pr_auc": None,
        "prevalence": _safe_div(n_pos, len(y_true)),
    }
    for f in fpr_targets:
        out[f"tpr_at_fpr_{f:g}"] = None
    if n_pos == 0 or n_neg == 0:
        return out

    tp, fp = _curve(y_true, y_score)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    out["roc_auc"] = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    out["pr_auc"] = float(np.sum((recall - prev_recall) * precision))

    for f in fpr_targets:
        idx = int(np.searchsorted(fpr, f, side="right")) - 1
        out[f"tpr_at_fpr_{f:g}"] = float(tpr[idx])
    return out


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr: float = 0.01) -> float:
    """Score threshold such that about `fpr` of benign rows score above it.

    Predict attack when score > threshold. Intended for source validation data only.
    """
    y_true = np.asarray(y_true).astype(bool)
    benign = np.asarray(y_score, dtype=np.float64)[~y_true]
    if benign.size == 0:
        raise ValueError("Need benign rows to choose a threshold")
    return float(np.quantile(benign, 1.0 - fpr))


def group_report(
    y_pred: np.ndarray,
    group: np.ndarray,
    shared: list[str],
    unseen: list[str],
    min_group_flows: int = 500,
) -> dict[str, Any]:
    """Per-group recall for shared groups, detection rate for unseen groups."""
    y_pred = np.asarray(y_pred).astype(bool)
    group = np.asarray(group)

    def one(name: str) -> dict[str, Any] | None:
        mask = group == GROUPS.index(name)
        n = int(mask.sum())
        if n < min_group_flows:
            return None
        return {"n": n, "recall": float(y_pred[mask].mean())}

    shared_out = {g: r for g in shared if g != BENIGN and (r := one(g)) is not None}
    unseen_out = {g: r for g in unseen if g != BENIGN and (r := one(g)) is not None}

    unseen_mask = np.isin(group, [GROUPS.index(g) for g in unseen_out])
    n_unseen = int(unseen_mask.sum())
    return {
        "shared": shared_out,
        "unseen": unseen_out,
        "shared_macro_recall": (
            float(np.mean([r["recall"] for r in shared_out.values()])) if shared_out else None
        ),
        "unseen_detection_rate": float(y_pred[unseen_mask].mean()) if n_unseen else None,
        "unseen_flows": n_unseen,
    }


def evaluate_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
    shared: list[str] | None = None,
    unseen: list[str] | None = None,
    min_group_flows: int = 500,
) -> dict[str, Any]:
    """Full metric report for one (source -> target) evaluation."""
    report: dict[str, Any] = {
        "n_flows": int(len(y_true)),
        "binary": binary_report(y_true, y_pred),
        "ranking": ranking_report(y_true, y_score) if y_score is not None else None,
        "groups": group_report(y_pred, group, shared or [], unseen or [], min_group_flows),
    }
    return report
