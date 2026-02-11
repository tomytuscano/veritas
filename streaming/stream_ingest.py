"""Veritas streaming pipeline — Kafka topic ingestion."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, from_unixtime, to_timestamp
from pyspark.sql.types import StructType

from streaming.config import KAFKA_BROKERS


def read_kafka_stream(
    spark: SparkSession,
    topic: str,
    schema: StructType,
    watermark_delay: str,
    brokers: str = KAFKA_BROKERS,
) -> DataFrame:
    """Read a Kafka topic, parse JSON values, and apply a watermark.

    The Zeek ``ts`` field (epoch seconds) is converted to a Spark timestamp
    column named ``event_time`` and used as the watermark column.
    """
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", brokers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    # Rename dotted Zeek field names to underscores for Spark compatibility.
    for field in schema.fields:
        if "." in field.name:
            safe_name = field.name.replace(".", "_")
            parsed = parsed.withColumnRenamed(field.name, safe_name)

    # Convert epoch ``ts`` to a proper timestamp for watermarking.
    if "ts" in [f.name for f in schema.fields]:
        parsed = (
            parsed
            .withColumn("event_time", to_timestamp(from_unixtime(col("ts"))))
            .withWatermark("event_time", watermark_delay)
        )

    return parsed
