"""Veritas classifier pipeline — CLI entry point."""

import argparse
import sys

from config import OUTPUT_DIR, RANDOM_SEED
from utils import setup_logging, timer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Veritas — encrypted traffic classifier training pipeline",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Directory for model artifacts and reports",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Subsample N rows for GridSearchCV (faster tuning)",
    )
    args = parser.parse_args()

    from pathlib import Path
    output_dir = Path(args.output_dir)

    log = setup_logging()
    log.info("Veritas training pipeline starting (seed=%d)", args.seed)

    # ── Stage 1: Load data ─────────────────────────────────────────────────
    with timer("Data loading", log):
        from data_loader import load_all_classes
        df = load_all_classes()

    # ── Stage 2: Feature engineering ───────────────────────────────────────
    with timer("Feature engineering", log):
        from feature_engineering import engineer_features
        X, feature_names = engineer_features(df)
        y = df["label"]

    # ── Stage 3: Model training ────────────────────────────────────────────
    with timer("Model training", log):
        from training import train_model
        results = train_model(
            X, y, feature_names,
            output_dir=output_dir,
            subsample=args.subsample,
            seed=args.seed,
        )

    # ── Stage 4: Evaluation ────────────────────────────────────────────────
    with timer("Evaluation", log):
        from evaluation import evaluate_model
        evaluate_model(results, output_dir)

    log.info("Pipeline complete. Artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
