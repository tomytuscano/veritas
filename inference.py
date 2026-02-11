"""Veritas classifier — inference on new Zeek logs."""

import logging

import joblib
import numpy as np

from config import CONFIDENCE_THRESHOLD, OUTPUT_DIR
from data_loader import load_all_classes
from feature_engineering import engineer_features
from utils import setup_logging

logger = logging.getLogger("veritas")


def load_model(model_dir=OUTPUT_DIR):
    """Load trained model and its artifacts."""
    model = joblib.load(model_dir / "lgbm_model.pkl")
    feature_names = joblib.load(model_dir / "feature_names.pkl")
    class_labels = joblib.load(model_dir / "class_labels.pkl")
    return model, feature_names, class_labels


def predict(model, X, class_labels, threshold=CONFIDENCE_THRESHOLD):
    """
    Run predictions with confidence scoring.

    Returns a list of dicts with keys: label, confidence, is_confident.
    """
    probas = model.predict_proba(X)
    pred_indices = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    results = []
    for idx, conf in zip(pred_indices, confidences):
        results.append({
            "label": class_labels[idx],
            "confidence": float(conf),
            "is_confident": conf >= threshold,
        })
    return results


if __name__ == "__main__":
    setup_logging()

    logger.info("Loading model artifacts from %s", OUTPUT_DIR)
    model, feature_names, class_labels = load_model()

    logger.info("Loading and engineering features from Zeek logs …")
    df = load_all_classes()
    X, _ = engineer_features(df)

    logger.info("Running inference on %d flows …", len(X))
    results = predict(model, X, class_labels)

    confident = sum(1 for r in results if r["is_confident"])
    logger.info(
        "Results: %d flows — %d confident (%.1f%%), %d uncertain",
        len(results), confident, 100 * confident / len(results),
        len(results) - confident,
    )

    for i, r in enumerate(results[:20]):
        status = "UNCERTAIN" if not r["is_confident"] else r["label"]
        logger.info("  Flow %d: %-12s (confidence: %.2f%%)", i, status, r["confidence"] * 100)
