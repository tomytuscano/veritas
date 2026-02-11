"""Veritas streaming pipeline — windowed source-IP concurrency features.

In the batch pipeline, ``data_loader._compute_concurrency()`` groups the
entire conn table by source IP.  In streaming we cannot see the full
dataset, so we approximate with a 10-minute sliding window (5-min slide).

Features produced (same names as batch):
  - src_flow_count          — flows per source IP in the window
  - unique_dest_count       — distinct destination IPs
  - unique_dest_port_count  — distinct destination ports
  - dest_port_entropy       — Shannon entropy over destination ports
"""

import math
from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from streaming.config import CONCURRENCY_SLIDE, CONCURRENCY_WINDOW


# Schema for the pandas grouped-map UDF that computes per-source-IP stats.
_CONCURRENCY_SCHEMA = StructType([
    StructField("id_orig_h", StringType()),
    StructField("window_start", TimestampType()),
    StructField("src_flow_count", LongType()),
    StructField("unique_dest_count", LongType()),
    StructField("unique_dest_port_count", LongType()),
    StructField("dest_port_entropy", DoubleType()),
])


def _compute_port_entropy(ports: pd.Series) -> float:
    """Shannon entropy over destination port distribution."""
    if ports.empty:
        return 0.0
    counts = ports.value_counts()
    total = counts.sum()
    probs = counts / total
    entropy = -(probs * probs.map(math.log2)).sum()
    return float(entropy)


def _concurrency_pandas_udf(key, pdf: pd.DataFrame) -> pd.DataFrame:
    """Grouped-map UDF: compute concurrency stats for one (source_ip, window) group."""
    src_ip = key[0]
    window_start = key[1]
    return pd.DataFrame([{
        "id_orig_h": src_ip,
        "window_start": window_start,
        "src_flow_count": len(pdf),
        "unique_dest_count": pdf["id_resp_h"].nunique(),
        "unique_dest_port_count": pdf["id_resp_p"].nunique(),
        "dest_port_entropy": _compute_port_entropy(pdf["id_resp_p"]),
    }])


def compute_concurrency(conn_df: DataFrame) -> DataFrame:
    """Compute windowed concurrency features for the conn stream.

    Returns a DataFrame with ``uid`` plus the four concurrency columns,
    ready to be joined back to the main flow.

    Strategy:
      1. Group conn into sliding windows by source IP.
      2. For each (source IP, window), compute aggregated stats using
         built-in Spark aggregations (flow count, unique dests, unique ports).
      3. For port entropy, use a ``pandas_udf`` grouped aggregate.
      4. Join the window-level stats back to each flow by source IP and
         window membership.
    """
    # Create the sliding window column.
    windowed = conn_df.withColumn(
        "window",
        F.window("event_time", CONCURRENCY_WINDOW, CONCURRENCY_SLIDE),
    )

    # Built-in aggregations for the simple counters.
    agg_stats = (
        windowed
        .groupBy("id_orig_h", "window")
        .agg(
            F.count("uid").alias("src_flow_count"),
            F.countDistinct("id_resp_h").alias("unique_dest_count"),
            F.countDistinct("id_resp_p").alias("unique_dest_port_count"),
        )
    )

    # Port entropy via applyInPandas grouped map.
    entropy_df = (
        windowed
        .select("id_orig_h", F.col("window.start").alias("window_start"), "id_resp_h", "id_resp_p")
        .groupBy("id_orig_h", "window_start")
        .applyInPandas(_concurrency_pandas_udf, schema=_CONCURRENCY_SCHEMA)
        .select("id_orig_h", "window_start", "dest_port_entropy")
    )

    # Merge entropy into the aggregate stats.
    concurrency = agg_stats.join(
        entropy_df,
        (agg_stats["id_orig_h"] == entropy_df["id_orig_h"])
        & (agg_stats["window.start"] == entropy_df["window_start"]),
        "left",
    ).select(
        agg_stats["id_orig_h"],
        agg_stats["window"],
        "src_flow_count",
        "unique_dest_count",
        "unique_dest_port_count",
        "dest_port_entropy",
    )

    # Join back to individual flows by source IP + window membership.
    # Each flow falls into one or more sliding windows; we take the latest.
    flow_windows = windowed.select("uid", "id_orig_h", "window")
    enriched = flow_windows.join(
        concurrency,
        on=["id_orig_h", "window"],
        how="left",
    ).drop("window")

    # A flow may appear in multiple overlapping windows — keep only the first.
    enriched = enriched.dropDuplicates(["uid"])

    return enriched.select(
        "uid",
        F.col("src_flow_count").cast("float"),
        F.col("unique_dest_count").cast("float"),
        F.col("unique_dest_port_count").cast("float"),
        F.col("dest_port_entropy").cast("float"),
    )
