"""Supervised baselines (scikit-learn). These set the numbers the RL models must beat.

All baselines expose predict_proba; the attack score is P(attack).
"""

from __future__ import annotations

from typing import Any

BASELINES = ("logreg", "rf", "hgb")


def make_baseline(name: str, seed: int) -> Any:
    """Build an unfitted estimator. Imports sklearn lazily so the package imports without it."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown baseline {name!r}; choose from {BASELINES}")
