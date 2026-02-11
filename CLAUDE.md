# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Veritas is an AI traffic classifier that identifies encrypted/obfuscated network traffic (VPNs, proxies, BitTorrent) using flow-based behavioral analysis on Zeek network logs. It uses CUDA-accelerated LightGBM gradient boosting with 133 engineered features. Achieves 99.84% accuracy on 514k test flows.

## Running the Pipeline

```bash
# Activate virtual environment
source vertitas_env/bin/activate

# Full pipeline (load → engineer features → train → evaluate)
python main.py

# With subsampling for faster GridSearchCV tuning
python main.py --subsample 50000

# Custom seed and output directory
python main.py --seed 123 --output-dir /tmp/models
```

There is no test suite, linter, or CI/CD configuration.

## Architecture

The pipeline is a linear sequence of stages, each in its own module:

1. **`main.py`** — CLI entry point, orchestrates the pipeline stages
2. **`config.py`** — All constants centralized here: paths, traffic classes, hyperparameter grid, categorical value enums, column lists for transforms
3. **`data_loader.py`** — Loads Zeek CSVs from `zeek_logs/{normal,vpn,proxy,bittorrent}/`, performs multi-table joins (conn ← ssl ← x509, conn ← http, conn ← dns), computes concurrency features, then drops network identifiers (IPs/ports) to prevent leakage
4. **`feature_engineering.py`** — Transforms joined DataFrames into 133 features: log transforms, one-hot encoding (protocol/service/conn_state), TCP history flags, SSL/TLS features (version ordinal, cipher families, JA3/SNI entropy), x509 cert features, DNS/HTTP aggregations, join indicator flags. NaNs filled with -1 (tree-friendly sentinel).
5. **`training.py`** — GridSearchCV with StratifiedKFold (5 folds, f1_macro scoring), trains LightGBM with `device="cuda"`, exports pickled artifacts to `models/`
6. **`evaluation.py`** — Generates classification report, confusion matrix PNG, feature importance chart, and confidence threshold analysis (90% threshold)
7. **`inference.py`** — Loads trained model artifacts and runs predictions with confidence scoring on new Zeek logs
8. **`utils.py`** — Logging setup and `@timer()` context manager

## Key Design Decisions

- **NaN → -1 sentinel**: Missing values are filled with -1 so tree-based models can isolate absence as a meaningful split
- **Identifiers dropped after concurrency**: `data_loader.py` computes src-IP flow stats (flow count, unique dests, port entropy) before dropping IPs/ports
- **CUDA-accelerated training**: LightGBM uses `device="cuda"` for GPU tree building (requires lightgbm built with `USE_CUDA=ON` and GCC 12 as CUDA host compiler)
- **Nested parallelism avoided**: LightGBM uses `n_jobs=-1`, GridSearchCV uses `n_jobs=1`
- **float32 casting**: Numeric columns explicitly cast for memory efficiency
- **Reproducibility**: Seed=42 fixed everywhere, stratified splits in both train/test and CV

## Data Layout

```
zeek_logs/{normal,vpn,proxy,bittorrent}/
  conn.csv   — main flow table, joined on uid
  ssl.csv    — SSL/TLS handshakes (deduplicated on uid)
  x509.csv   — certificates (joined via cert fingerprint from ssl)
  http.csv   — aggregated per uid
  dns.csv    — aggregated per uid
```

## Inference

```bash
# Run inference on Zeek logs using the trained model
python inference.py
```

`inference.py` provides `load_model()` and `predict()` for use in other code. Predictions include per-flow labels, confidence scores, and a 90% threshold flag.

## Streaming Pipeline (Real-time)

Real-time classification via Kafka + Spark Structured Streaming. Zeek publishes live logs to Kafka topics, Spark joins and engineers features, the trained LightGBM model classifies flows, and detections are written to `veritas.detections`.

```bash
# Start local Kafka + Spark dev stack
cd streaming/docker && docker compose up -d

# Submit the streaming pipeline
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    streaming/stream_pipeline.py
```

### Streaming Architecture

```
streaming/
  config.py               — Kafka brokers, watermarks, window sizes, trigger interval
  schemas.py              — PySpark StructType schemas for each Zeek log type
  spark_session.py        — SparkSession builder with Kafka package
  stream_ingest.py        — Read from Kafka topics, parse JSON, apply watermarks
  stream_aggregations.py  — HTTP/DNS per-uid aggregation (PySpark port of data_loader)
  stream_joins.py         — Multi-stream join: conn ← ssl ← x509, ← http, ← dns
  stream_concurrency.py   — Windowed source-IP concurrency features (10-min sliding)
  stream_features.py      — PySpark port of feature_engineering.py (133 features)
  stream_inference.py     — Broadcast LightGBM model + mapInPandas UDF
  stream_pipeline.py      — Main entry point: wires the full streaming DAG
  zeek/veritas_kafka.zeek — Zeek-kafka plugin config (routes logs to topics)
  docker/docker-compose.yml — Local Kafka + Spark for development
```

### Kafka Topics

| Topic | Key | Purpose |
|---|---|---|
| `veritas.conn` | uid | Connection flows (driving stream) |
| `veritas.ssl` | uid | SSL/TLS handshakes |
| `veritas.x509` | fingerprint | Certificates (24h retention) |
| `veritas.http` | uid | HTTP requests |
| `veritas.dns` | uid | DNS queries |
| `veritas.detections` | uid | Output: label, confidence, is_confident |

### Streaming Design Decisions

- **Watermarks**: 5 min on conn/ssl/http/dns, 24h on x509 (certs are reused)
- **Trigger**: 30-second micro-batches
- **Concurrency**: 10-min sliding window (5-min slide) grouped by source IP
- **Left outer joins**: Spark waits for watermark before emitting null-side rows (~5 min latency)
- **Model broadcast**: LightGBM loaded once, broadcast to all executors, scored via `mapInPandas`

## Dependencies

Python 3.12 with: pandas, scikit-learn, joblib, matplotlib, numpy, lightgbm, pyspark, kafka-python (see `requirements.txt`).

Streaming also requires the Spark Kafka connector JAR: `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0` (auto-fetched via `--packages`).

LightGBM must be built from source with CUDA support:
```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
  pip install --no-binary lightgbm lightgbm \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
```
