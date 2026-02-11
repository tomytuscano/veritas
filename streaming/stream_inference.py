"""Veritas streaming pipeline — broadcast LightGBM model + mapInPandas UDF.

Loads the trained model once, broadcasts it to all Spark executors, and
runs ``model.predict_proba()`` inside a ``mapInPandas`` UDF so that each
micro-batch partition is scored in vectorised pandas.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from streaming.config import CONFIDENCE_THRESHOLD, MODEL_DIR

logger = logging.getLogger("veritas.streaming")

# Schema for the detection output rows.
DETECTION_SCHEMA = StructType([
    StructField("uid", StringType()),
    StructField("ts", TimestampType()),
    StructField("orig_h", StringType()),
    StructField("resp_h", StringType()),
    StructField("label", StringType()),
    StructField("confidence", DoubleType()),
    StructField("is_confident", BooleanType()),
])


def load_and_broadcast_model(
    spark: SparkSession,
    model_dir: Path = MODEL_DIR,
) -> tuple:
    """Load model artifacts and broadcast them across the cluster.

    Returns (broadcast_model, feature_names, class_labels).
    """
    model = joblib.load(model_dir / "lgbm_model.pkl")
    feature_names = joblib.load(model_dir / "feature_names.pkl")
    class_labels = joblib.load(model_dir / "class_labels.pkl")

    bc_model = spark.sparkContext.broadcast(model)
    bc_class_labels = spark.sparkContext.broadcast(class_labels)

    logger.info(
        "Broadcast model (%d features, %d classes) to executors",
        len(feature_names), len(class_labels),
    )
    return bc_model, feature_names, bc_class_labels


def run_inference(
    features_df: DataFrame,
    bc_model,
    feature_names: list[str],
    bc_class_labels,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> DataFrame:
    """Score each micro-batch partition using the broadcast LightGBM model.

    Uses ``mapInPandas`` to process partitions as pandas DataFrames, which
    avoids per-row serialisation overhead.

    The output DataFrame has the ``DETECTION_SCHEMA`` columns:
    uid, ts, orig_h, resp_h, label, confidence, is_confident.
    """
    # Capture feature names and threshold in closure (picklable scalars).
    _feature_names = list(feature_names)
    _threshold = float(threshold)

    def _predict_partition(iterator):
        """Yield detection DataFrames for each pandas partition."""
        model = bc_model.value
        class_labels = bc_class_labels.value

        for pdf in iterator:
            if pdf.empty:
                yield pd.DataFrame(columns=[f.name for f in DETECTION_SCHEMA.fields])
                continue

            # Extract the feature matrix in the correct column order.
            X = pdf[_feature_names].values.astype(np.float32)

            probas = model.predict_proba(X)
            pred_indices = np.argmax(probas, axis=1)
            confidences = np.max(probas, axis=1)

            out = pd.DataFrame({
                "uid": pdf["uid"].values,
                "ts": pdf["event_time"].values if "event_time" in pdf.columns else pd.NaT,
                "orig_h": pdf["id_orig_h"].values if "id_orig_h" in pdf.columns else None,
                "resp_h": pdf["id_resp_h"].values if "id_resp_h" in pdf.columns else None,
                "label": [class_labels[i] for i in pred_indices],
                "confidence": confidences.astype(np.float64),
                "is_confident": confidences >= _threshold,
            })
            yield out

    return features_df.mapInPandas(_predict_partition, schema=DETECTION_SCHEMA)
