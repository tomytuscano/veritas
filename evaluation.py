"""Veritas classifier pipeline — evaluation and reporting."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import CONFIDENCE_THRESHOLD

logger = logging.getLogger("veritas")


def evaluate_model(results: dict, output_dir: Path) -> None:
    """Generate evaluation report, confusion matrix, and feature importance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    class_labels = results["class_labels"]
    feature_names = results["feature_names"]
    cv_results = results["cv_results"]
    best_params = results["best_params"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_conf = y_proba.max(axis=1)

    lines: list[str] = []

    # ── 1. Classification report ───────────────────────────────────────────
    report = classification_report(
        y_test, y_pred, target_names=class_labels, digits=4,
    )
    lines.append("=" * 60)
    lines.append("VERITAS — Training Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Best hyperparameters:")
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Classification Report (test set):")
    lines.append(report)

    # ── 2. Cross-validation summary ────────────────────────────────────────
    lines.append("-" * 60)
    lines.append("Cross-Validation Scores (training set, 5-fold):")
    for metric in ["test_accuracy", "test_f1_macro", "test_precision_macro", "test_recall_macro"]:
        scores = cv_results[metric]
        clean_name = metric.replace("test_", "")
        lines.append(f"  {clean_name:20s}: {scores.mean():.4f} +/- {scores.std():.4f}")
    lines.append("")

    # ── 3. Confidence threshold analysis ───────────────────────────────────
    confident_mask = y_conf >= CONFIDENCE_THRESHOLD
    coverage = confident_mask.mean() * 100
    if confident_mask.sum() > 0:
        acc_confident = (y_pred[confident_mask] == y_test.values[confident_mask]).mean() * 100
    else:
        acc_confident = 0.0
    uncertain_mask = ~confident_mask
    if uncertain_mask.sum() > 0:
        acc_uncertain = (y_pred[uncertain_mask] == y_test.values[uncertain_mask]).mean() * 100
    else:
        acc_uncertain = 0.0

    lines.append("-" * 60)
    lines.append(f"Confidence Threshold Analysis (>= {CONFIDENCE_THRESHOLD:.0%}):")
    lines.append(f"  Coverage:                {coverage:.2f}%")
    lines.append(f"  Accuracy (confident):    {acc_confident:.2f}%")
    lines.append(f"  Accuracy (uncertain):    {acc_uncertain:.2f}%")
    lines.append(f"  Uncertain samples:       {uncertain_mask.sum()}")
    lines.append("")

    # ── 4. Dataset stats ───────────────────────────────────────────────────
    lines.append("-" * 60)
    lines.append("Dataset:")
    lines.append(f"  Test samples:   {len(y_test)}")
    lines.append(f"  Features:       {len(feature_names)}")
    lines.append(f"  Classes:        {class_labels}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    report_path = output_dir / "training_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved training report to %s", report_path)
    print(f"\n{report_text}")

    # ── 5. Confusion matrix plot ───────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", cm_path)

    # ── 6. Top-20 feature importance ───────────────────────────────────────
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_names)), top_vals, color="#4C72B0")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fi_path = output_dir / "feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance chart to %s", fi_path)
