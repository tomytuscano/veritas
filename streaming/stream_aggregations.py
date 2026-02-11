"""Veritas streaming pipeline — HTTP and DNS per-uid aggregation.

Ports the aggregation logic from ``data_loader._load_http()`` and
``data_loader._load_dns()`` into PySpark column expressions so they work
inside Spark Structured Streaming.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from streaming.config import DNS_QTYPE_VALUES, HTTP_METHOD_VALUES


# ── HTTP aggregation ──────────────────────────────────────────────────────────

def aggregate_http(http_df: DataFrame) -> DataFrame:
    """Aggregate raw HTTP stream events per ``uid``.

    Produces the same output columns as ``data_loader._load_http()``:
      http_request_count, http_avg_req_body_len, http_total_req_body_len,
      http_avg_resp_body_len, http_total_resp_body_len,
      http_status_2xx_ratio … 5xx_ratio, http_avg_ua_len,
      http_unique_hosts, http_method_{GET,POST,…}
    """
    df = http_df

    # Cast numerics.
    df = (
        df
        .withColumn("request_body_len", F.col("request_body_len").cast("double"))
        .withColumn("response_body_len", F.col("response_body_len").cast("double"))
        .withColumn("status_code", F.col("status_code").cast("double"))
    )

    # Status-code group flags.
    df = (
        df
        .withColumn("_s2xx", F.when((F.col("status_code") >= 200) & (F.col("status_code") < 300), 1.0).otherwise(0.0))
        .withColumn("_s3xx", F.when((F.col("status_code") >= 300) & (F.col("status_code") < 400), 1.0).otherwise(0.0))
        .withColumn("_s4xx", F.when((F.col("status_code") >= 400) & (F.col("status_code") < 500), 1.0).otherwise(0.0))
        .withColumn("_s5xx", F.when(F.col("status_code") >= 500, 1.0).otherwise(0.0))
    )

    # Method indicator columns.
    for m in HTTP_METHOD_VALUES:
        df = df.withColumn(f"_method_{m}", F.when(F.col("method") == m, 1.0).otherwise(0.0))

    # User-agent length.
    df = df.withColumn("_ua_len", F.coalesce(F.length(F.col("user_agent")).cast("double"), F.lit(0.0)))

    # Build aggregation expressions.
    agg_exprs = [
        F.count("request_body_len").alias("http_request_count"),
        F.avg("request_body_len").alias("http_avg_req_body_len"),
        F.sum("request_body_len").alias("http_total_req_body_len"),
        F.avg("response_body_len").alias("http_avg_resp_body_len"),
        F.sum("response_body_len").alias("http_total_resp_body_len"),
        F.avg("_s2xx").alias("http_status_2xx_ratio"),
        F.avg("_s3xx").alias("http_status_3xx_ratio"),
        F.avg("_s4xx").alias("http_status_4xx_ratio"),
        F.avg("_s5xx").alias("http_status_5xx_ratio"),
        F.avg("_ua_len").alias("http_avg_ua_len"),
        F.countDistinct("host").alias("http_unique_hosts"),
    ]
    for m in HTTP_METHOD_VALUES:
        agg_exprs.append(F.sum(f"_method_{m}").alias(f"http_method_{m}"))

    return df.groupBy("uid").agg(*agg_exprs)


# ── DNS aggregation ──────────────────────────────────────────────────────────

def aggregate_dns(dns_df: DataFrame) -> DataFrame:
    """Aggregate raw DNS stream events per ``uid``.

    Produces the same output columns as ``data_loader._load_dns()``:
      dns_query_count, dns_rtt_mean, dns_rtt_std,
      dns_nxdomain_ratio, dns_rejected, dns_has_answer,
      dns_avg_ttl, dns_unique_queries, dns_qtype_{A,AAAA,…}
    """
    df = dns_df

    df = df.withColumn("rtt", F.col("rtt").cast("double"))

    df = df.withColumn("_is_nxdomain", F.when(F.col("rcode_name") == "NXDOMAIN", 1.0).otherwise(0.0))
    df = df.withColumn("_rejected", F.when(F.col("rejected") == "T", 1.0).otherwise(0.0))
    df = df.withColumn("_has_answer", F.when(F.size(F.col("answers")) > 0, 1.0).otherwise(0.0))

    # First TTL from array.
    df = df.withColumn("_first_ttl", F.col("TTLs").getItem(0).cast("double"))

    # Query-type indicator columns.
    for qt in DNS_QTYPE_VALUES:
        df = df.withColumn(f"_qtype_{qt}", F.when(F.col("qtype_name") == qt, 1.0).otherwise(0.0))

    agg_exprs = [
        F.count("rtt").alias("dns_query_count"),
        F.avg("rtt").alias("dns_rtt_mean"),
        F.stddev("rtt").alias("dns_rtt_std"),
        F.avg("_is_nxdomain").alias("dns_nxdomain_ratio"),
        F.max("_rejected").alias("dns_rejected"),
        F.avg("_has_answer").alias("dns_has_answer"),
        F.avg("_first_ttl").alias("dns_avg_ttl"),
        F.countDistinct("query").alias("dns_unique_queries"),
    ]
    for qt in DNS_QTYPE_VALUES:
        agg_exprs.append(F.sum(f"_qtype_{qt}").alias(f"dns_qtype_{qt}"))

    return df.groupBy("uid").agg(*agg_exprs)
