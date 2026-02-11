"""Veritas streaming pipeline — main entry point.

Wires the full streaming DAG:
  ingest → pre-aggregate → join → concurrency → features → inference
  → write detections to Kafka.

Usage::

    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        streaming/stream_pipeline.py
"""

import logging
import sys
from pathlib import Path

from pyspark.sql import functions as F

# Ensure project root is on the path for imports.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from streaming.config import (
    CHECKPOINT_DIR,
    KAFKA_BROKERS,
    TOPIC_CONN,
    TOPIC_DETECTIONS,
    TOPIC_DNS,
    TOPIC_HTTP,
    TOPIC_SSL,
    TOPIC_X509,
    TRIGGER_INTERVAL,
    WATERMARK_CONN,
    WATERMARK_DNS,
    WATERMARK_HTTP,
    WATERMARK_SSL,
    WATERMARK_X509,
)
from streaming.schemas import (
    CONN_SCHEMA,
    DNS_SCHEMA,
    HTTP_SCHEMA,
    SSL_SCHEMA,
    X509_SCHEMA,
)
from streaming.spark_session import create_spark_session
from streaming.stream_aggregations import aggregate_dns, aggregate_http
from streaming.stream_concurrency import compute_concurrency
from streaming.stream_features import engineer_features
from streaming.stream_inference import (
    load_and_broadcast_model,
    run_inference,
)
from streaming.stream_ingest import read_kafka_stream
from streaming.stream_joins import join_all_streams, join_ssl_x509

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("veritas.streaming")


def main():
    logger.info("Starting Veritas streaming pipeline")

    # ── 1. SparkSession ───────────────────────────────────────────────────
    spark = create_spark_session()
    logger.info("SparkSession created: %s", spark.sparkContext.applicationId)

    # ── 2. Load & broadcast model ─────────────────────────────────────────
    bc_model, feature_names, bc_class_labels = load_and_broadcast_model(spark)

    # ── 3. Ingest from Kafka topics ───────────────────────────────────────
    conn_stream = read_kafka_stream(spark, TOPIC_CONN, CONN_SCHEMA, WATERMARK_CONN)
    ssl_stream = read_kafka_stream(spark, TOPIC_SSL, SSL_SCHEMA, WATERMARK_SSL)
    x509_stream = read_kafka_stream(spark, TOPIC_X509, X509_SCHEMA, WATERMARK_X509)
    http_stream = read_kafka_stream(spark, TOPIC_HTTP, HTTP_SCHEMA, WATERMARK_HTTP)
    dns_stream = read_kafka_stream(spark, TOPIC_DNS, DNS_SCHEMA, WATERMARK_DNS)

    logger.info("Kafka streams attached to 5 topics")

    # ── 4. Pre-aggregate HTTP and DNS per uid ─────────────────────────────
    http_agg = aggregate_http(http_stream)
    dns_agg = aggregate_dns(dns_stream)

    # ── 5. Join ssl + x509 ────────────────────────────────────────────────
    ssl_x509 = join_ssl_x509(ssl_stream, x509_stream)

    # ── 6. Compute concurrency features from conn ─────────────────────────
    concurrency = compute_concurrency(conn_stream)

    # ── 7. Multi-stream join ──────────────────────────────────────────────
    joined = join_all_streams(conn_stream, ssl_x509, http_agg, dns_agg)

    # Merge concurrency features back onto flows.
    joined = joined.join(concurrency, on="uid", how="left")

    # ── 8. Feature engineering (133 features) ─────────────────────────────
    features = engineer_features(joined, feature_names)

    # ── 9. Inference ──────────────────────────────────────────────────────
    detections = run_inference(features, bc_model, feature_names, bc_class_labels)

    # ── 10. Write detections to Kafka ─────────────────────────────────────
    query = (
        detections
        .selectExpr(
            "uid AS key",
            "to_json(struct(*)) AS value",
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("topic", TOPIC_DETECTIONS)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/detections")
        .trigger(processingTime=TRIGGER_INTERVAL)
        .outputMode("append")
        .start()
    )

    logger.info("Streaming query started — writing detections to %s", TOPIC_DETECTIONS)
    query.awaitTermination()


if __name__ == "__main__":
    main()
