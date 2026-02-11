"""Veritas streaming pipeline — configuration constants."""

import sys
from pathlib import Path

# Allow imports from the project root (batch pipeline modules).
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export batch config constants used by the streaming feature-engineering layer.
from config import (  # noqa: E402, F401
    CLASSES,
    CONCURRENCY_COLS,
    CONN_STATE_VALUES,
    DNS_QTYPE_VALUES,
    HTTP_METHOD_VALUES,
    LOG_TRANSFORM_COLS,
    OUTPUT_DIR,
    PROTO_VALUES,
    SERVICE_TOP_N,
    SSL_CIPHER_FAMILIES,
    SSL_CURVE_VALUES,
    SSL_NEXT_PROTO_VALUES,
    SSL_VERSION_ORDINAL,
    CONFIDENCE_THRESHOLD,
)

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BROKERS = "localhost:9092"

TOPIC_CONN = "veritas.conn"
TOPIC_SSL = "veritas.ssl"
TOPIC_X509 = "veritas.x509"
TOPIC_HTTP = "veritas.http"
TOPIC_DNS = "veritas.dns"
TOPIC_DETECTIONS = "veritas.detections"

KAFKA_PARTITIONS = 12

# ── Watermarks ────────────────────────────────────────────────────────────────
WATERMARK_CONN = "5 minutes"
WATERMARK_SSL = "5 minutes"
WATERMARK_HTTP = "5 minutes"
WATERMARK_DNS = "5 minutes"
WATERMARK_X509 = "24 hours"  # Certs are long-lived and reused across flows.

# ── Trigger / micro-batch interval ────────────────────────────────────────────
TRIGGER_INTERVAL = "30 seconds"

# ── Concurrency window (sliding) ─────────────────────────────────────────────
CONCURRENCY_WINDOW = "10 minutes"
CONCURRENCY_SLIDE = "5 minutes"

# ── Checkpointing ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = str(_project_root / "checkpoints")

# ── Model artifacts ───────────────────────────────────────────────────────────
MODEL_DIR = OUTPUT_DIR
