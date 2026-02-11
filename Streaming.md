# Streaming Pipeline: Real-Time Traffic Classification

**Architecture:** Zeek sensors publish live network logs to Apache Kafka, Spark Structured Streaming joins and engineers the same 133 features as the batch pipeline, and the trained LightGBM model classifies flows in near-real-time. Detections are written to a Kafka output topic for downstream consumers (SIEM, dashboards, alerting).

```
Zeek Sensor → zeek-kafka plugin (JSON, lz4) → Kafka (5 input topics)
    → Spark Structured Streaming (join, feature eng, inference)
    → Kafka Output Topic (veritas.detections)
```

------

## 1. Kafka Topic Design

All uid-keyed topics use 12 partitions for co-partitioned stream-stream joins.

| Topic | Key | Watermark | Purpose |
| :--- | :--- | :--- | :--- |
| `veritas.conn` | uid | 5 min | Connection flows (driving stream) |
| `veritas.ssl` | uid | 5 min | SSL/TLS handshakes |
| `veritas.x509` | fingerprint | 24 hours | Certificates (long-lived, reused across flows) |
| `veritas.http` | uid | 5 min | HTTP requests |
| `veritas.dns` | uid | 5 min | DNS queries |
| `veritas.detections` | uid | — | **Output:** `{uid, ts, orig_h, resp_h, label, confidence, is_confident}` |

**Why 24 hours for x509?** TLS certificates are long-lived artifacts. A single certificate may be referenced by thousands of SSL handshakes over hours or days. The extended watermark ensures that certificate features are available when the corresponding SSL flow arrives, even if the x509 log event was published much earlier.

------

## 2. Streaming Strategy

### Micro-Batch Trigger

The pipeline uses **30-second micro-batches** (`processingTime="30 seconds"`). This balances latency against throughput — smaller batches reduce time-to-detection but increase per-batch overhead from Spark's scheduling and checkpointing.

### Watermarks & Late Data

Watermarks tell Spark how long to hold state waiting for late-arriving events before closing a window:

- **5 minutes** on conn/ssl/http/dns — accommodates typical network jitter and Zeek processing delay.
- **24 hours** on x509 — certificates are published once and referenced many times.

Once a watermark advances past a window boundary, Spark can emit null-side rows for left outer joins (e.g., a conn flow with no matching SSL record).

### Join Order

The streaming join chain mirrors the batch pipeline exactly:

1. **ssl + x509** — Join on `cert_chain_fps[0] = fingerprint` to attach certificate features to SSL handshakes.
2. **conn + ssl** — Left join on `uid` to enrich connection flows with SSL/TLS metadata.
3. **+ http_agg** — Left join pre-aggregated HTTP features on `uid`.
4. **+ dns_agg** — Left join pre-aggregated DNS features on `uid`.

All joins are **left outer** — a connection flow that has no corresponding SSL, HTTP, or DNS records still passes through with null values (filled to -1 during feature engineering).

### Concurrency Features

In the batch pipeline, concurrency features (`src_flow_count`, `unique_dest_count`, `unique_dest_port_count`, `dest_port_entropy`) are computed over the entire dataset grouped by source IP. In streaming, this is approximated with a **10-minute sliding window** with a **5-minute slide**:

- For each (source IP, window), Spark computes flow counts, unique destination counts, and port entropy.
- Results are joined back to individual flows by source IP and window membership.
- A flow appearing in multiple overlapping windows is deduplicated to retain only one set of concurrency features.

This window-based approximation means concurrency features reflect recent traffic patterns rather than all-time statistics, which is actually more appropriate for detecting real-time behavioral anomalies.

------

## 3. Module Architecture

```
streaming/
├── config.py               # Kafka brokers, watermarks, windows, trigger interval
├── schemas.py              # PySpark StructType for each Zeek log type
├── spark_session.py        # SparkSession builder with Kafka connector
├── stream_ingest.py        # Generic Kafka reader: parse JSON, watermark
├── stream_aggregations.py  # HTTP/DNS per-uid aggregation (PySpark groupBy)
├── stream_joins.py         # Multi-stream join chain (conn ← ssl ← x509 ← http ← dns)
├── stream_concurrency.py   # Windowed source-IP concurrency features
├── stream_features.py      # 133-feature port to PySpark column expressions
├── stream_inference.py     # Broadcast LightGBM model + mapInPandas UDF
├── stream_pipeline.py      # Main entry point: wires the full DAG
├── zeek/
│   └── veritas_kafka.zeek  # Zeek-kafka plugin routing config
└── docker/
    └── docker-compose.yml  # Local Kafka + Spark dev stack
```

### Data Flow Through the Pipeline

```
read_kafka_stream()         ← 5 calls, one per topic
       │
       ├── aggregate_http() ← groupBy(uid).agg(...)
       ├── aggregate_dns()  ← groupBy(uid).agg(...)
       ├── join_ssl_x509()  ← cert fingerprint join
       └── compute_concurrency() ← sliding window
       │
  join_all_streams()        ← conn + ssl_x509 + http_agg + dns_agg + concurrency
       │
  engineer_features()       ← 133 PySpark column expressions
       │
  run_inference()           ← mapInPandas with broadcast LightGBM
       │
  writeStream → Kafka       ← veritas.detections topic
```

------

## 4. Feature Engineering (Streaming Port)

The streaming `stream_features.py` produces the **exact same 133 features** as the batch `feature_engineering.py`. The key difference is the API: pandas operations become PySpark column expressions.

| Feature Category | Count | Batch API | Streaming API |
| :--- | :--- | :--- | :--- |
| Conn numerics & derived | 17 | `pd.to_numeric()`, arithmetic | `F.col().cast("float")`, arithmetic |
| Log transforms | 8 | `np.log1p()` | `F.log1p()` |
| Proto/service/conn_state one-hot | 26 | `==` comparison | `F.when().otherwise()` |
| History flags & entropy | 11 | `str.contains()`, custom `_shannon_entropy()` | `F.contains()`, `@pandas_udf` |
| SSL/TLS features | 19 | `str.startswith()`, `map()`, custom entropy | `F.startswith()`, chained `F.when()`, `@pandas_udf` |
| x509 certificate features | 7 | `pd.to_numeric()` | `F.col().cast("float")` |
| DNS/HTTP aggregated | 29 | Pass-through numeric cast | Pass-through numeric cast |
| Join indicators | 3 | `notna().astype(int8)` | `F.when().isNotNull()` |
| Concurrency | 4 | Pass-through numeric cast | Pass-through numeric cast |
| Log-transform columns | 8 | `np.log1p()` | `F.log1p(F.greatest())` |

### Shannon Entropy in Streaming

The batch pipeline computes Shannon entropy for string fields (TCP history, JA3 hash, JA3S hash, SNI, SSL history) using a pure-Python `_shannon_entropy()` function applied via `Series.map()`.

In streaming, this is implemented as a **scalar Pandas UDF** (`@pandas_udf(FloatType())`). Spark sends string columns to the UDF in Arrow-backed pandas Series batches, avoiding per-row Python overhead. The entropy calculation itself is identical:

$$H = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

### Column Order Enforcement

The trained model expects features in the exact order saved in `feature_names.pkl`. The streaming feature module builds its final `select()` from this list — any column present in the DataFrame is coalesced with a -1 fallback; any missing column is emitted as a -1 literal. This ensures perfect alignment regardless of which optional log sources (SSL, HTTP, DNS) are present for a given flow.

------

## 5. Model Inference

The inference strategy is designed to minimize serialization overhead on the Spark cluster:

1. **Load once**: `joblib.load()` reads `lgbm_model.pkl`, `feature_names.pkl`, and `class_labels.pkl` from the `models/` directory.
2. **Broadcast**: The model and class labels are broadcast to all executors via `SparkContext.broadcast()`. This ships the model exactly once per executor JVM, not once per partition.
3. **Score via mapInPandas**: Each micro-batch partition is converted to a pandas DataFrame, the feature matrix is extracted in the correct column order, and `model.predict_proba()` is called in a single vectorized operation.
4. **Confidence thresholding**: Predictions exceeding 90% confidence are flagged `is_confident=true`.

### Detection Output Schema

Each row written to `veritas.detections`:

```json
{
  "uid": "CYqnKg3YsWedEb8gR4",
  "ts": "2026-02-11T14:32:07.000Z",
  "orig_h": "192.168.1.100",
  "resp_h": "104.16.249.249",
  "label": "vpn",
  "confidence": 0.9734,
  "is_confident": true
}
```

------

## 6. Latency Profile

| Stage | Expected Latency | Cause |
| :--- | :--- | :--- |
| Zeek → Kafka | < 1 second | zeek-kafka plugin writes with 100ms buffering |
| Kafka → Spark ingest | ~30 seconds | Micro-batch trigger interval |
| Watermark hold (joins) | ~5 minutes | Waiting for late-arriving SSL/HTTP/DNS events |
| Feature engineering | < 5 seconds | PySpark column expressions per micro-batch |
| Model inference | < 2 seconds | Vectorized `predict_proba()` per partition |
| **End-to-end** | **~5–6 minutes** | Dominated by watermark delay |

The 5-minute watermark is the primary latency contributor. For flows that have all their associated log events (SSL, HTTP, DNS) arrive within the same micro-batch, the effective latency drops to ~30–60 seconds. Flows with no associated events (e.g., plain UDP connections) also resolve quickly since the left outer join emits them once the watermark advances.

------

## 7. Setup & Deployment

### Prerequisites

- Python 3.12 with `pyspark>=3.5.0` and `kafka-python>=2.0.0`
- Apache Kafka cluster (or local dev stack via Docker)
- Apache Spark 3.5+ with the Kafka connector JAR
- Trained Veritas model artifacts in `models/` (from the batch pipeline)
- Zeek sensor with the [zeek-kafka](https://github.com/SeisoLLC/zeek-kafka) plugin

### Local Development

```bash
# Start Kafka + Spark dev stack
cd streaming/docker
docker compose up -d

# Wait for services to be healthy
docker compose ps

# Submit the streaming pipeline
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    streaming/stream_pipeline.py
```

The Docker Compose stack provides:
- **Kafka** (KRaft mode, no ZooKeeper) on port 9092 (internal) / 9094 (external)
- **Spark Master** with Web UI on port 8080
- **Spark Worker** (4 cores, 4GB RAM)
- **Kafka UI** on port 8081 for topic inspection

### Zeek Sensor Configuration

Deploy `streaming/zeek/veritas_kafka.zeek` to the Zeek sensor:

```bash
zeek -i eth0 streaming/zeek/veritas_kafka.zeek
```

This configures the zeek-kafka plugin to route `Conn::LOG`, `SSL::LOG`, `X509::LOG`, `HTTP::LOG`, and `DNS::LOG` to their respective Kafka topics with lz4 compression.

### Production Deployment

For production, adjust the following in `streaming/config.py`:

| Parameter | Dev Default | Production Recommendation |
| :--- | :--- | :--- |
| `KAFKA_BROKERS` | `localhost:9092` | Multi-broker cluster address |
| `TRIGGER_INTERVAL` | `30 seconds` | 10–30 seconds depending on volume |
| `CONCURRENCY_WINDOW` | `10 minutes` | Tune based on traffic patterns |
| `CHECKPOINT_DIR` | Local `checkpoints/` | HDFS/S3 path for fault tolerance |

------

## 8. Monitoring & Troubleshooting

### Spark Streaming UI

The Spark Web UI (port 4040 on the driver) provides:
- **Streaming tab**: Input rate, processing time, batch duration per micro-batch.
- **State store size**: Monitor for unbounded growth (indicates watermark misconfiguration).
- **Task failures**: Individual executor/task errors.

### Common Issues

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| No detections output | Watermark hasn't advanced | Wait 5+ minutes; ensure conn events are flowing |
| All features are -1 | Schema mismatch | Verify Zeek JSON field names match `schemas.py` |
| OOM on executor | State store too large | Reduce watermark delays or increase executor memory |
| Slow micro-batches | Too many shuffle partitions | Tune `spark.sql.shuffle.partitions` (default 12) |
| Model not found | Missing `models/*.pkl` | Run the batch pipeline first to train and export the model |

### Consuming Detections

Read from the `veritas.detections` topic using any Kafka consumer:

```bash
# CLI consumer
kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic veritas.detections \
    --from-beginning

# Python consumer
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "veritas.detections",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)
for msg in consumer:
    detection = msg.value
    if detection["is_confident"]:
        print(f"[{detection['label']}] {detection['orig_h']} → {detection['resp_h']} "
              f"(confidence: {detection['confidence']:.1%})")
```

------

## 9. Batch vs. Streaming Comparison

| Aspect | Batch Pipeline | Streaming Pipeline |
| :--- | :--- | :--- |
| Input | Zeek CSV files on disk | Kafka topics (JSON) |
| Processing | Single-pass pandas | Spark Structured Streaming |
| Concurrency features | Full-dataset groupBy | 10-min sliding window |
| Join strategy | In-memory pandas merge | Watermark-based stream-stream join |
| Latency | Minutes to hours (depends on dataset) | ~5–6 minutes (watermark-bounded) |
| Model | Same `lgbm_model.pkl` | Same `lgbm_model.pkl` (broadcast) |
| Features | 133 (pandas) | 133 (PySpark, identical logic) |
| Output | Text report + PNG charts | Kafka topic (JSON per detection) |
| Use case | Training, evaluation, research | Production real-time monitoring |
