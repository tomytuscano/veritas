"""Veritas streaming pipeline — PySpark port of feature_engineering.py.

Translates all 133 features from pandas column operations to PySpark
column expressions.  Shannon entropy (for history, JA3, SNI strings) is
computed via a scalar ``pandas_udf``.
"""

import math
from typing import Iterator

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

from streaming.config import (
    CONN_STATE_VALUES,
    LOG_TRANSFORM_COLS,
    PROTO_VALUES,
    SERVICE_TOP_N,
    SSL_CIPHER_FAMILIES,
    SSL_CURVE_VALUES,
    SSL_NEXT_PROTO_VALUES,
    SSL_VERSION_ORDINAL,
)


# ── Scalar pandas UDF for Shannon entropy ─────────────────────────────────────

@pandas_udf(FloatType())
def shannon_entropy_udf(series: pd.Series) -> pd.Series:
    """Compute Shannon entropy for each string in a pandas Series."""
    def _ent(val):
        if not isinstance(val, str) or len(val) == 0:
            return np.float32(np.nan)
        counts: dict[str, int] = {}
        for ch in val:
            counts[ch] = counts.get(ch, 0) + 1
        length = len(val)
        entropy = 0.0
        for count in counts.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return np.float32(entropy)

    return series.map(_ent)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: DataFrame, feature_names: list[str]) -> DataFrame:
    """Transform a joined streaming DataFrame into the 133-feature vector.

    The output ``select`` enforces the exact column order from the trained
    model's ``feature_names.pkl`` so that every row is compatible with
    ``model.predict_proba()``.

    Parameters
    ----------
    df : DataFrame
        The joined stream (conn + ssl/x509 + http_agg + dns_agg +
        concurrency).
    feature_names : list[str]
        The ordered feature names from ``feature_names.pkl``.
    """
    feat = df

    # ── 1. Conn direct numerics ───────────────────────────────────────────
    for col_name in [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "ip_proto",
    ]:
        if col_name in feat.columns:
            feat = feat.withColumn(col_name, F.col(col_name).cast("float"))

    # ── 2. Conn derived features ──────────────────────────────────────────
    feat = (
        feat
        .withColumn("total_bytes",
                     F.coalesce(F.col("orig_bytes"), F.lit(0)).cast("float")
                     + F.coalesce(F.col("resp_bytes"), F.lit(0)).cast("float"))
        .withColumn("total_pkts",
                     F.coalesce(F.col("orig_pkts"), F.lit(0)).cast("float")
                     + F.coalesce(F.col("resp_pkts"), F.lit(0)).cast("float"))
        .withColumn("bytes_ratio",
                     F.coalesce(F.col("orig_bytes"), F.lit(0)).cast("float")
                     / (F.coalesce(F.col("resp_bytes"), F.lit(0)).cast("float") + 1))
        .withColumn("pkts_ratio",
                     F.coalesce(F.col("orig_pkts"), F.lit(0)).cast("float")
                     / (F.coalesce(F.col("resp_pkts"), F.lit(0)).cast("float") + 1))
        .withColumn("avg_pkt_size_orig",
                     F.coalesce(F.col("orig_bytes"), F.lit(0)).cast("float")
                     / (F.coalesce(F.col("orig_pkts"), F.lit(0)).cast("float") + 1))
        .withColumn("avg_pkt_size_resp",
                     F.coalesce(F.col("resp_bytes"), F.lit(0)).cast("float")
                     / (F.coalesce(F.col("resp_pkts"), F.lit(0)).cast("float") + 1))
        .withColumn("overhead_orig",
                     (F.coalesce(F.col("orig_ip_bytes"), F.lit(0)).cast("float")
                      - F.coalesce(F.col("orig_bytes"), F.lit(0)).cast("float"))
                     / (F.coalesce(F.col("orig_pkts"), F.lit(0)).cast("float") + 1))
        .withColumn("overhead_resp",
                     (F.coalesce(F.col("resp_ip_bytes"), F.lit(0)).cast("float")
                      - F.coalesce(F.col("resp_bytes"), F.lit(0)).cast("float"))
                     / (F.coalesce(F.col("resp_pkts"), F.lit(0)).cast("float") + 1))
    )

    # ── 3. Log transforms for skewed features ─────────────────────────────
    for col_name in LOG_TRANSFORM_COLS:
        if col_name in feat.columns:
            feat = feat.withColumn(
                f"{col_name}_log",
                F.log1p(F.greatest(F.col(col_name), F.lit(0))).cast("float"),
            )

    # ── 4. Proto one-hot ──────────────────────────────────────────────────
    if "proto" in feat.columns:
        for val in PROTO_VALUES:
            feat = feat.withColumn(
                f"proto_{val}",
                F.when(F.col("proto") == val, 1).otherwise(0).cast("tinyint"),
            )

    # ── 5. Service top-N + other ──────────────────────────────────────────
    if "service" in feat.columns:
        for svc in SERVICE_TOP_N:
            feat = feat.withColumn(
                f"service_{svc}",
                F.when(F.col("service") == svc, 1).otherwise(0).cast("tinyint"),
            )
        feat = feat.withColumn(
            "service_other",
            F.when(
                (~F.col("service").isin(SERVICE_TOP_N)) & F.col("service").isNotNull(),
                1,
            ).otherwise(0).cast("tinyint"),
        )

    # ── 6. conn_state one-hot ─────────────────────────────────────────────
    if "conn_state" in feat.columns:
        for val in CONN_STATE_VALUES:
            feat = feat.withColumn(
                f"conn_state_{val}",
                F.when(F.col("conn_state") == val, 1).otherwise(0).cast("tinyint"),
            )

    # ── 7. local_orig / local_resp ────────────────────────────────────────
    for col_name in ("local_orig", "local_resp"):
        if col_name in feat.columns:
            feat = feat.withColumn(
                col_name,
                F.when(F.col(col_name) == "T", 1).otherwise(0).cast("tinyint"),
            )

    # ── 8. History field features ─────────────────────────────────────────
    if "history" in feat.columns:
        hist_col = F.coalesce(F.col("history"), F.lit(""))
        feat = feat.withColumn("_history_filled", hist_col)
        feat = feat.withColumn("history_len", F.length("_history_filled").cast("float"))
        for ch in ["S", "s", "F", "f", "R", "r", "D", "d"]:
            feat = feat.withColumn(
                f"history_has_{ch}",
                F.when(F.col("_history_filled").contains(ch), 1).otherwise(0).cast("tinyint"),
            )
        feat = feat.withColumn("history_entropy", shannon_entropy_udf(F.col("_history_filled")))
        feat = feat.drop("_history_filled")

    # ── 9. SSL features ───────────────────────────────────────────────────
    if "ssl_version" in feat.columns:
        # Map version string → ordinal integer.
        version_expr = F.lit(None).cast("float")
        for ver_str, ver_int in SSL_VERSION_ORDINAL.items():
            version_expr = F.when(F.col("ssl_version") == ver_str, float(ver_int)).otherwise(version_expr)
        feat = feat.withColumn("ssl_version", version_expr)

    if "ssl_cipher" in feat.columns:
        for fam in SSL_CIPHER_FAMILIES:
            feat = feat.withColumn(
                f"ssl_cipher_{fam}",
                F.when(F.col("ssl_cipher").startswith(fam), 1).otherwise(0).cast("tinyint"),
            )

    if "ssl_curve" in feat.columns:
        for val in SSL_CURVE_VALUES:
            feat = feat.withColumn(
                f"ssl_curve_{val}",
                F.when(F.col("ssl_curve") == val, 1).otherwise(0).cast("tinyint"),
            )

    for bool_col, out_col in [
        ("ssl_resumed", "ssl_resumed"),
        ("ssl_established", "ssl_established"),
        ("ssl_sni_matches_cert", "ssl_sni_match"),
    ]:
        if bool_col in feat.columns:
            feat = feat.withColumn(
                out_col,
                F.when(F.col(bool_col) == "T", 1).otherwise(0).cast("tinyint"),
            )
            if out_col != bool_col:
                feat = feat.drop(bool_col)

    if "ssl_next_protocol" in feat.columns:
        for val in SSL_NEXT_PROTO_VALUES:
            feat = feat.withColumn(
                f"ssl_next_proto_{val}",
                F.when(F.col("ssl_next_protocol") == val, 1).otherwise(0).cast("tinyint"),
            )

    if "ssl_ssl_history" in feat.columns:
        ssl_hist = F.coalesce(F.col("ssl_ssl_history"), F.lit(""))
        feat = feat.withColumn("_ssl_hist_filled", ssl_hist)
        feat = feat.withColumn("ssl_history_len", F.length("_ssl_hist_filled").cast("float"))
        feat = feat.withColumn("ssl_history_entropy", shannon_entropy_udf(F.col("_ssl_hist_filled")))
        feat = feat.drop("_ssl_hist_filled")

    if "ssl_ja3" in feat.columns:
        feat = feat.withColumn("ssl_ja3_entropy", shannon_entropy_udf(F.col("ssl_ja3")))

    if "ssl_ja3s" in feat.columns:
        feat = feat.withColumn("ssl_ja3s_entropy", shannon_entropy_udf(F.col("ssl_ja3s")))

    if "ssl_server_name" in feat.columns:
        sni = F.coalesce(F.col("ssl_server_name"), F.lit(""))
        feat = feat.withColumn("_sni_filled", sni)
        feat = feat.withColumn("ssl_sni_len", F.length("_sni_filled").cast("float"))
        feat = feat.withColumn("ssl_sni_entropy", shannon_entropy_udf(F.col("_sni_filled")))
        feat = feat.drop("_sni_filled")

    # ── 10. x509 certificate features ─────────────────────────────────────
    for col_name in [c for c in feat.columns if c.startswith("x509_")]:
        feat = feat.withColumn(col_name, F.col(col_name).cast("float"))

    # ── 11. Join indicators (already tinyint from stream_joins) ───────────
    # has_ssl, has_http, has_dns — no transform needed.

    # ── 12. DNS aggregated features ───────────────────────────────────────
    for col_name in [c for c in feat.columns if c.startswith("dns_")]:
        feat = feat.withColumn(col_name, F.col(col_name).cast("float"))

    # ── 13. HTTP aggregated features ──────────────────────────────────────
    for col_name in [c for c in feat.columns if c.startswith("http_")]:
        feat = feat.withColumn(col_name, F.col(col_name).cast("float"))

    # ── 14. Flow concurrency features ─────────────────────────────────────
    for col_name in ["src_flow_count", "unique_dest_count", "unique_dest_port_count", "dest_port_entropy"]:
        if col_name in feat.columns:
            feat = feat.withColumn(col_name, F.col(col_name).cast("float"))

    # ── 15. Fill NaN with -1 sentinel ─────────────────────────────────────
    # Build the final select using the exact feature order from the model.
    # Any missing feature column is filled with -1 (literal).
    select_exprs = []
    for name in feature_names:
        if name in feat.columns:
            select_exprs.append(
                F.coalesce(F.col(name).cast("float"), F.lit(-1.0)).alias(name)
            )
        else:
            select_exprs.append(F.lit(-1.0).cast("float").alias(name))

    # Keep uid and passthrough columns for the downstream inference output.
    passthrough = ["uid", "event_time"]
    for pt in passthrough:
        if pt in feat.columns:
            select_exprs.insert(0, F.col(pt))

    # Also preserve network identifiers for the detection output.
    for id_col in ["id_orig_h", "id_resp_h"]:
        if id_col in feat.columns:
            select_exprs.insert(0, F.col(id_col))

    return feat.select(*select_exprs)
