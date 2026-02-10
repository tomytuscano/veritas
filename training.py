"""Veritas classifier pipeline — model training (LightGBM)."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from config import CV_FOLDS, LGBM_PARAM_GRID, RANDOM_SEED, TEST_SIZE

logger = logging.getLogger("veritas")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    output_dir: Path,
    subsample: int | None = None,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Train a LightGBM classifier with GridSearchCV.

    Returns a dict with keys: model, X_test, y_test, cv_results, best_params,
    feature_names, class_labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Encode labels ──────────────────────────────────────────────────────
    class_labels = sorted(y.unique().tolist())
    label_map = {name: i for i, name in enumerate(class_labels)}
    y_encoded = y.map(label_map).astype(int)

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y_encoded,
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ── Optional subsampling for faster GridSearchCV ───────────────────────
    if subsample and subsample < len(X_train):
        logger.info("Subsampling %d rows for GridSearchCV", subsample)
        idx = np.random.RandomState(seed).choice(
            len(X_train), size=subsample, replace=False,
        )
        X_grid, y_grid = X_train.iloc[idx], y_train.iloc[idx]
    else:
        X_grid, y_grid = X_train, y_train

    # ── GridSearchCV ───────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        estimator=LGBMClassifier(
            random_state=seed,
            is_unbalance=True,
            n_jobs=-1,
            verbose=-1,
        ),
        param_grid=LGBM_PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=False,
    )
    grid.fit(X_grid, y_grid)
    best_params = grid.best_params_
    logger.info("Best params: %s  (CV f1_macro=%.4f)", best_params, grid.best_score_)

    # ── Retrain on full training set with best params ──────────────────────
    best_model = LGBMClassifier(
        **best_params,
        random_state=seed,
        is_unbalance=True,
        n_jobs=-1,
        verbose=-1,
    )
    best_model.fit(X_train, y_train)

    # ── Cross-validation on best estimator ─────────────────────────────────
    cv_scores = cross_validate(
        best_model, X_train, y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        n_jobs=1,
    )

    # ── Export artifacts ───────────────────────────────────────────────────
    joblib.dump(best_model, output_dir / "lgbm_model.pkl")
    joblib.dump(feature_names, output_dir / "feature_names.pkl")
    joblib.dump(class_labels, output_dir / "class_labels.pkl")
    joblib.dump(grid.cv_results_, output_dir / "grid_search_results.pkl")
    logger.info("Saved model artifacts to %s", output_dir)

    return {
        "model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "cv_results": cv_scores,
        "best_params": best_params,
        "feature_names": feature_names,
        "class_labels": class_labels,
    }
