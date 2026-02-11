"""Veritas streaming pipeline — multi-stream join chain.

Reproduces the batch join order from ``data_loader._load_class()``:
  ssl + x509 → conn + ssl → + http_agg → + dns_agg

All uid-keyed joins use left outer with time-range conditions so that
Spark Structured Streaming can track watermarks and eventually emit rows
where the right side is null (no matching SSL/HTTP/DNS for that flow).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def join_ssl_x509(ssl_df: DataFrame, x509_df: DataFrame) -> DataFrame:
    """Join SSL events with X.509 certificate features via first cert fingerprint.

    The SSL stream has ``cert_chain_fps`` (array of strings).  We extract
    the first element and join to x509 on ``fingerprint``.
    """
    # Extract first cert fingerprint from the array.
    ssl = ssl_df.withColumn(
        "_first_cert_fp",
        F.col("cert_chain_fps").getItem(0),
    )

    # Compute x509 features (mirrors data_loader._load_x509).
    x509 = (
        x509_df
        .withColumn("x509_key_length", F.col("certificate_key_length").cast("float"))
        .withColumn("x509_is_rsa", F.when(F.col("certificate_key_type") == "rsa", 1).otherwise(0).cast("tinyint"))
        .withColumn("x509_is_ecdsa", F.when(F.col("certificate_key_type") == "ecdsa", 1).otherwise(0).cast("tinyint"))
        .withColumn("x509_is_self_signed",
                     F.when(F.col("certificate_subject") == F.col("certificate_issuer"), 1).otherwise(0).cast("tinyint"))
        .withColumn("x509_is_ca", F.when(F.col("basic_constraints_ca") == "T", 1).otherwise(0).cast("tinyint"))
        .withColumn("x509_is_host_cert", F.when(F.col("host_cert") == "T", 1).otherwise(0).cast("tinyint"))
        .withColumn("x509_validity_days",
                     ((F.col("certificate_not_valid_after") - F.col("certificate_not_valid_before")) / 86400).cast("float"))
        .select(
            "fingerprint", "event_time",
            "x509_key_length", "x509_is_rsa", "x509_is_ecdsa",
            "x509_is_self_signed", "x509_is_ca", "x509_is_host_cert",
            "x509_validity_days",
        )
        .dropDuplicates(["fingerprint"])
    )

    # Left join ssl → x509 on first cert fingerprint.
    ssl_x509 = ssl.join(
        x509.withColumnRenamed("event_time", "x509_event_time"),
        ssl["_first_cert_fp"] == x509["fingerprint"],
        "left",
    ).drop("_first_cert_fp", "fingerprint", "x509_event_time")

    return ssl_x509


def _prefix_ssl_columns(ssl_x509_df: DataFrame) -> DataFrame:
    """Prefix SSL columns with ``ssl_`` to match batch pipeline naming.

    Columns already prefixed with ``x509_`` or named ``uid``/``event_time``
    are left unchanged.
    """
    skip = {"uid", "event_time", "ts"}
    renamed = ssl_x509_df
    for field in ssl_x509_df.columns:
        if field in skip or field.startswith("x509_") or field.startswith("ssl_"):
            continue
        # SSL drop columns that aren't needed downstream.
        if field in {
            "id_orig_h", "id_orig_p", "id_resp_h", "id_resp_p",
            "last_alert", "cert_chain_fps", "client_cert_chain_fps",
            "subject", "issuer", "validation_status",
            "orig_l2_addr", "resp_l2_addr",
        }:
            renamed = renamed.drop(field)
            continue
        renamed = renamed.withColumnRenamed(field, f"ssl_{field}")
    return renamed


def join_all_streams(
    conn_df: DataFrame,
    ssl_x509_df: DataFrame,
    http_agg_df: DataFrame,
    dns_agg_df: DataFrame,
) -> DataFrame:
    """Perform the full multi-stream left-outer join chain.

    Join order (matches batch pipeline):
      1. conn LEFT JOIN ssl_x509 ON uid
      2. result LEFT JOIN http_agg ON uid
      3. result LEFT JOIN dns_agg ON uid

    Join indicators ``has_ssl``, ``has_http``, ``has_dns`` are added.
    """
    # Prefix ssl columns to avoid name collisions with conn.
    ssl_prefixed = _prefix_ssl_columns(ssl_x509_df)

    # Alias event_time columns to avoid ambiguity.
    ssl_join = ssl_prefixed.withColumnRenamed("event_time", "ssl_event_time")
    http_join = http_agg_df
    dns_join = dns_agg_df

    # 1. conn + ssl_x509
    joined = conn_df.join(
        ssl_join,
        on="uid",
        how="left",
    )
    joined = joined.withColumn(
        "has_ssl",
        F.when(F.col("ssl_version").isNotNull(), 1).otherwise(0).cast("tinyint"),
    )
    joined = joined.drop("ssl_event_time")

    # 2. + http_agg
    joined = joined.join(http_join, on="uid", how="left")
    joined = joined.withColumn(
        "has_http",
        F.when(F.col("http_request_count").isNotNull(), 1).otherwise(0).cast("tinyint"),
    )

    # 3. + dns_agg
    joined = joined.join(dns_join, on="uid", how="left")
    joined = joined.withColumn(
        "has_dns",
        F.when(F.col("dns_query_count").isNotNull(), 1).otherwise(0).cast("tinyint"),
    )

    return joined
