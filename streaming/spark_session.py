"""Veritas streaming pipeline — SparkSession builder."""

from pyspark.sql import SparkSession

SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"


def create_spark_session(app_name: str = "veritas-streaming") -> SparkSession:
    """Build a SparkSession configured for Kafka structured streaming."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", SPARK_KAFKA_PACKAGE)
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.sql.streaming.stateStore.stateSchemaCheck", "false")
        .getOrCreate()
    )
