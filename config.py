"""Veritas classifier pipeline — configuration constants."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR / "zeek_logs"
OUTPUT_DIR = PROJECT_DIR / "models"

# ── Traffic classes ────────────────────────────────────────────────────────────
CLASSES = ["normal", "vpn", "proxy", "bittorrent"]

# ── Reproducibility / split ────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CONFIDENCE_THRESHOLD = 0.90

# ── Columns to DROP (identifiers — leak-prone and useless for generalisation) ─
CONN_DROP_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "tunnel_parents", "orig_l2_addr", "resp_l2_addr",
]

SSL_DROP_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "last_alert", "cert_chain_fps", "client_cert_chain_fps",
    "subject", "issuer", "validation_status",
    "orig_l2_addr", "resp_l2_addr",
]

DNS_DROP_COLS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "trans_id", "query", "qclass", "qclass_name",
    "answers", "TTLs", "orig_l2_addr", "resp_l2_addr",
]

# ── Conn numeric columns (cast to float32) ────────────────────────────────────
CONN_NUMERIC_COLS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "ip_proto",
]

# ── Log-transform columns (skewed distributions) ──────────────────────────────
LOG_TRANSFORM_COLS = [
    "duration", "orig_bytes", "resp_bytes", "total_bytes",
    "orig_ip_bytes", "resp_ip_bytes", "orig_pkts", "resp_pkts",
]

# ── Hardcoded categorical value sets (ensures alignment across runs) ──────────
CONN_STATE_VALUES = [
    "S0", "S1", "S2", "S3", "SF", "REJ", "RSTO", "RSTR",
    "RSTOS0", "RSTRH", "SH", "SHR", "OTH",
]

PROTO_VALUES = ["tcp", "udp", "icmp"]

SERVICE_TOP_N = [
    "dns", "http", "ssl", "dhcp", "ntp", "ssh", "irc", "ftp", "smtp",
]

SSL_VERSION_ORDINAL = {
    "SSLv2": 0, "SSLv3": 1,
    "TLSv10": 2, "TLSv11": 3, "TLSv12": 4, "TLSv13": 5,
}

SSL_CIPHER_FAMILIES = [
    "TLS_AES", "TLS_CHACHA20", "TLS_ECDHE_RSA", "TLS_ECDHE_ECDSA",
    "TLS_RSA", "TLS_DHE",
]

SSL_CURVE_VALUES = ["x25519", "secp256r1", "secp384r1", "secp521r1"]

SSL_NEXT_PROTO_VALUES = ["h2", "http/1.1", "h3"]

DNS_QTYPE_VALUES = ["A", "AAAA", "PTR", "CNAME", "MX", "TXT", "SRV"]

# ── HTTP categorical values ───────────────────────────────────────────────────
HTTP_METHOD_VALUES = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "CONNECT"]

# ── Concurrency feature columns (computed in data_loader) ─────────────────────
CONCURRENCY_COLS = [
    "src_flow_count", "unique_dest_count",
    "unique_dest_port_count", "dest_port_entropy",
]

# ── LightGBM GridSearchCV parameter grid ──────────────────────────────────────
LGBM_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [10, 20, -1],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 50],
}
