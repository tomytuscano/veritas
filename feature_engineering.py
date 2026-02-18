"""Veritas classifier pipeline — feature engineering."""

import logging
import math

import numpy as np
import pandas as pd

from config import (
    CONCURRENCY_COLS,
    CONN_STATE_VALUES,
    LOG_TRANSFORM_COLS,
    PROTO_VALUES,
    PROXY_PORTS,
    SERVICE_TOP_N,
    SSL_CIPHER_FAMILIES,
    SSL_CURVE_VALUES,
    SSL_NEXT_PROTO_VALUES,
    SSL_VERSION_ORDINAL,
    VPN_PORTS,
)

logger = logging.getLogger("veritas")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _shannon_entropy(s: pd.Series) -> pd.Series:
    """Compute Shannon entropy for each string in a Series."""
    def _ent(val):
        if not isinstance(val, str) or len(val) == 0:
            return np.nan
        counts = {}
        for ch in val:
            counts[ch] = counts.get(ch, 0) + 1
        length = len(val)
        entropy = 0.0
        for count in counts.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    return s.map(_ent, na_action="ignore").astype(np.float32)


def _bool_col(series: pd.Series) -> pd.Series:
    """Convert T/F string column to 1/0 int8."""
    return series.map({"T": 1, "F": 0}).fillna(0).astype(np.int8)


# ── Main transform ────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw joined DataFrame into numeric feature matrix + label array.

    Returns (X DataFrame with all numeric features, list of feature names).
    The label column is preserved in df but NOT in returned X.
    """
    # Collect all feature columns in a dict, then concat once at the end
    cols: dict[str, pd.Series] = {}

    # ── 1. Conn direct numerics ────────────────────────────────────────────
    for col in [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "ip_proto",
    ]:
        if col in df.columns:
            cols[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # ── 2. Conn derived features ───────────────────────────────────────────
    orig_bytes = cols["orig_bytes"].fillna(0)
    resp_bytes = cols["resp_bytes"].fillna(0)
    orig_pkts = cols["orig_pkts"].fillna(0)
    resp_pkts = cols["resp_pkts"].fillna(0)
    orig_ip_bytes = cols["orig_ip_bytes"].fillna(0)
    resp_ip_bytes = cols["resp_ip_bytes"].fillna(0)

    cols["total_bytes"] = (orig_bytes + resp_bytes).astype(np.float32)
    cols["total_pkts"] = (orig_pkts + resp_pkts).astype(np.float32)
    cols["bytes_ratio"] = (orig_bytes / (resp_bytes + 1)).astype(np.float32)
    cols["pkts_ratio"] = (orig_pkts / (resp_pkts + 1)).astype(np.float32)
    cols["avg_pkt_size_orig"] = (orig_bytes / (orig_pkts + 1)).astype(np.float32)
    cols["avg_pkt_size_resp"] = (resp_bytes / (resp_pkts + 1)).astype(np.float32)
    cols["overhead_orig"] = ((orig_ip_bytes - orig_bytes) / (orig_pkts + 1)).astype(np.float32)
    cols["overhead_resp"] = ((resp_ip_bytes - resp_bytes) / (resp_pkts + 1)).astype(np.float32)

    # Bidirectionality and throughput
    cols["is_bidirectional"] = ((orig_bytes > 0) & (resp_bytes > 0)).astype(np.int8)
    cols["payload_asymmetry"] = (
        (orig_bytes - resp_bytes).abs() / (orig_bytes + resp_bytes + 1)
    ).astype(np.float32)
    if "duration" in cols:
        duration_safe = cols["duration"].clip(lower=0.001)
        cols["bytes_per_second"] = (
            cols["total_bytes"] / duration_safe
        ).astype(np.float32)
        cols["pkts_per_second"] = (
            cols["total_pkts"] / duration_safe
        ).astype(np.float32)

    # ── 3. Log transforms for skewed features ──────────────────────────────
    for col in LOG_TRANSFORM_COLS:
        if col in cols:
            cols[f"{col}_log"] = np.log1p(cols[col].clip(lower=0)).astype(np.float32)

    # ── 4. Destination port features ──────────────────────────────────────
    if "resp_port" in df.columns:
        port = pd.to_numeric(df["resp_port"], errors="coerce")
        cols["resp_port"] = port.astype(np.float32)
        cols["is_vpn_port"] = port.isin(VPN_PORTS).astype(np.int8)
        cols["is_proxy_port"] = port.isin(PROXY_PORTS).astype(np.int8)
        cols["is_web_port"] = port.isin({80, 443, 8443}).astype(np.int8)
        cols["is_ephemeral_port"] = (port > 32768).astype(np.int8)
        cols["is_registered_port"] = ((port >= 1024) & (port <= 32768)).astype(np.int8)
        cols["is_well_known_port"] = (port < 1024).astype(np.int8)

    # ── 5. Proto one-hot ───────────────────────────────────────────────────
    if "proto" in df.columns:
        for val in PROTO_VALUES:
            cols[f"proto_{val}"] = (df["proto"] == val).astype(np.int8)

    # ── 6. Service top-N + other ───────────────────────────────────────────
    if "service" in df.columns:
        for svc in SERVICE_TOP_N:
            cols[f"service_{svc}"] = (df["service"] == svc).astype(np.int8)
        cols["service_other"] = (
            (~df["service"].isin(SERVICE_TOP_N) & df["service"].notna())
            .astype(np.int8)
        )

    # ── 6. conn_state one-hot ──────────────────────────────────────────────
    if "conn_state" in df.columns:
        for val in CONN_STATE_VALUES:
            cols[f"conn_state_{val}"] = (df["conn_state"] == val).astype(np.int8)

    # ── 7. local_orig / local_resp ─────────────────────────────────────────
    if "local_orig" in df.columns:
        cols["local_orig"] = _bool_col(df["local_orig"])
    if "local_resp" in df.columns:
        cols["local_resp"] = _bool_col(df["local_resp"])

    # ── 8. History field features ──────────────────────────────────────────
    if "history" in df.columns:
        hist = df["history"].fillna("")
        cols["history_len"] = hist.str.len().astype(np.float32)
        for ch in ["S", "s", "F", "f", "R", "r", "D", "d"]:
            cols[f"history_has_{ch}"] = hist.str.contains(ch, regex=False).astype(np.int8)
        cols["history_entropy"] = _shannon_entropy(hist)

    # ── 9. SSL features ───────────────────────────────────────────────────
    if "ssl_version" in df.columns:
        cols["ssl_version"] = df["ssl_version"].map(SSL_VERSION_ORDINAL).astype(np.float32)

    if "ssl_cipher" in df.columns:
        for fam in SSL_CIPHER_FAMILIES:
            cols[f"ssl_cipher_{fam}"] = (
                df["ssl_cipher"].str.startswith(fam, na=False).astype(np.int8)
            )

    if "ssl_curve" in df.columns:
        for val in SSL_CURVE_VALUES:
            cols[f"ssl_curve_{val}"] = (df["ssl_curve"] == val).astype(np.int8)

    if "ssl_resumed" in df.columns:
        cols["ssl_resumed"] = _bool_col(df["ssl_resumed"])
    if "ssl_established" in df.columns:
        cols["ssl_established"] = _bool_col(df["ssl_established"])
    if "ssl_sni_matches_cert" in df.columns:
        cols["ssl_sni_match"] = _bool_col(df["ssl_sni_matches_cert"])

    if "ssl_next_protocol" in df.columns:
        for val in SSL_NEXT_PROTO_VALUES:
            cols[f"ssl_next_proto_{val}"] = (
                df["ssl_next_protocol"] == val
            ).astype(np.int8)

    if "ssl_ssl_history" in df.columns:
        ssl_hist = df["ssl_ssl_history"].fillna("")
        cols["ssl_history_len"] = ssl_hist.str.len().astype(np.float32)
        cols["ssl_history_entropy"] = _shannon_entropy(ssl_hist)

    if "ssl_ja3" in df.columns:
        cols["ssl_ja3_entropy"] = _shannon_entropy(df["ssl_ja3"])
    if "ssl_ja3s" in df.columns:
        cols["ssl_ja3s_entropy"] = _shannon_entropy(df["ssl_ja3s"])

    if "ssl_server_name" in df.columns:
        sni = df["ssl_server_name"].fillna("")
        cols["ssl_sni_len"] = sni.str.len().astype(np.float32)
        cols["ssl_sni_entropy"] = _shannon_entropy(sni)

    # ── 10. x509 certificate features ──────────────────────────────────────
    x509_cols = [c for c in df.columns if c.startswith("x509_")]
    for col in x509_cols:
        cols[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # ── 11. Join indicators ────────────────────────────────────────────────
    for flag in ["has_ssl", "has_dns", "has_http"]:
        if flag in df.columns:
            cols[flag] = df[flag].astype(np.int8)

    # ── 12. DNS aggregated features ────────────────────────────────────────
    dns_cols = [c for c in df.columns if c.startswith("dns_")]
    for col in dns_cols:
        cols[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # ── 13. HTTP aggregated features ───────────────────────────────────────
    http_cols = [c for c in df.columns if c.startswith("http_")]
    for col in http_cols:
        cols[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # ── 14. Flow concurrency features ──────────────────────────────────────
    for col in CONCURRENCY_COLS:
        if col in df.columns:
            cols[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    # Build DataFrame from all columns at once (avoids fragmentation)
    feat = pd.DataFrame(cols)

    # ── 15. Fill remaining NaN with -1 (tree-friendly sentinel) ────────────
    feat.fillna(-1, inplace=True)

    feature_names = list(feat.columns)
    logger.info("Engineered %d features", len(feature_names))
    return feat, feature_names
