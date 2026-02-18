"""Veritas classifier pipeline — data loading and joining."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CONN_DROP_COLS,
    CONN_NUMERIC_COLS,
    DNS_QTYPE_VALUES,
    HTTP_METHOD_VALUES,
    LOGS_DIR,
    SSL_DROP_COLS,
)

logger = logging.getLogger("veritas")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _port_entropy_per_source(df: pd.DataFrame) -> pd.Series:
    """Compute destination port entropy per source IP (vectorised)."""
    pair_counts = (
        df.groupby(["id.orig_h", "id.resp_p"])
        .size()
        .reset_index(name="_cnt")
    )
    totals = pair_counts.groupby("id.orig_h")["_cnt"].transform("sum")
    probs = pair_counts["_cnt"] / totals
    pair_counts["_ent"] = -probs * np.log2(probs)
    return pair_counts.groupby("id.orig_h")["_ent"].sum()


# ── CSV loaders ────────────────────────────────────────────────────────────────

def _load_ssl(path: Path) -> pd.DataFrame:
    """Load combined ssl.csv, extract first cert fingerprint, dedup per [uid, class]."""
    df = pd.read_csv(path, na_values=["-"], dtype=str, low_memory=False)

    if "cert_chain_fps" in df.columns:
        df["_first_cert_fp"] = df["cert_chain_fps"].str.split(",").str[0]

    drop = [c for c in SSL_DROP_COLS if c in df.columns and c not in ("uid", "class")]
    df.drop(columns=drop, inplace=True)
    df.drop_duplicates(subset=["uid", "class"], keep="first", inplace=True)

    df.rename(
        columns={
            c: f"ssl_{c}"
            for c in df.columns
            if c not in ("uid", "class", "_first_cert_fp")
        },
        inplace=True,
    )
    return df


def _load_x509(path: Path) -> pd.DataFrame:
    """Load combined x509.csv and compute per-certificate features."""
    df = pd.read_csv(path, na_values=["-"], dtype=str, low_memory=False)

    result = pd.DataFrame()
    result["fingerprint"] = df["fingerprint"]
    result["x509_key_length"] = pd.to_numeric(
        df["certificate.key_length"], errors="coerce"
    ).astype(np.float32)
    result["x509_is_rsa"] = (df["certificate.key_type"] == "rsa").astype(np.int8)
    result["x509_is_ecdsa"] = (df["certificate.key_type"] == "ecdsa").astype(np.int8)
    result["x509_is_self_signed"] = (
        df["certificate.subject"] == df["certificate.issuer"]
    ).astype(np.int8)
    result["x509_is_ca"] = (df["basic_constraints.ca"] == "T").astype(np.int8)
    result["x509_is_host_cert"] = (df["host_cert"] == "T").astype(np.int8)

    not_before = pd.to_numeric(df["certificate.not_valid_before"], errors="coerce")
    not_after = pd.to_numeric(df["certificate.not_valid_after"], errors="coerce")
    result["x509_validity_days"] = ((not_after - not_before) / 86400).astype(np.float32)

    result.drop_duplicates(subset="fingerprint", keep="first", inplace=True)
    return result


def _load_http(path: Path) -> pd.DataFrame:
    """Load combined http.csv, aggregate per [uid, class]."""
    df = pd.read_csv(path, na_values=["-"], dtype=str, low_memory=False)

    df["request_body_len"] = pd.to_numeric(df["request_body_len"], errors="coerce")
    df["response_body_len"] = pd.to_numeric(df["response_body_len"], errors="coerce")
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce")

    df["_s2xx"] = ((df["status_code"] >= 200) & (df["status_code"] < 300)).astype(np.float32)
    df["_s3xx"] = ((df["status_code"] >= 300) & (df["status_code"] < 400)).astype(np.float32)
    df["_s4xx"] = ((df["status_code"] >= 400) & (df["status_code"] < 500)).astype(np.float32)
    df["_s5xx"] = (df["status_code"] >= 500).astype(np.float32)

    for m in HTTP_METHOD_VALUES:
        df[f"_method_{m}"] = (df["method"] == m).astype(np.float32)

    df["_ua_len"] = df["user_agent"].str.len().fillna(0).astype(np.float32)

    agg_dict = {
        "request_body_len": ["count", "mean", "sum"],
        "response_body_len": ["mean", "sum"],
        "_s2xx": "mean",
        "_s3xx": "mean",
        "_s4xx": "mean",
        "_s5xx": "mean",
        "_ua_len": "mean",
        "host": "nunique",
    }
    for m in HTTP_METHOD_VALUES:
        agg_dict[f"_method_{m}"] = "sum"

    grouped = df.groupby(["uid", "class"]).agg(agg_dict)

    grouped.columns = [
        "http_request_count",
        "http_avg_req_body_len",
        "http_total_req_body_len",
        "http_avg_resp_body_len",
        "http_total_resp_body_len",
        "http_status_2xx_ratio",
        "http_status_3xx_ratio",
        "http_status_4xx_ratio",
        "http_status_5xx_ratio",
        "http_avg_ua_len",
        "http_unique_hosts",
    ] + [f"http_method_{m}" for m in HTTP_METHOD_VALUES]

    grouped.reset_index(inplace=True)
    return grouped


def _load_dns(path: Path) -> pd.DataFrame:
    """Load combined dns.csv, aggregate per [uid, class]."""
    df = pd.read_csv(path, na_values=["-"], dtype=str, low_memory=False)

    df["rtt"] = pd.to_numeric(df["rtt"], errors="coerce")

    df["_is_nxdomain"] = (df["rcode_name"] == "NXDOMAIN").astype(np.float32)
    df["_rejected"] = (df["rejected"] == "T").astype(np.float32)
    df["_has_answer"] = df["answers"].notna().astype(np.float32) if "answers" in df.columns else 0.0

    if "TTLs" in df.columns:
        df["_first_ttl"] = (
            df["TTLs"].str.split(",").str[0].pipe(pd.to_numeric, errors="coerce")
        )
    else:
        df["_first_ttl"] = np.nan

    for qt in DNS_QTYPE_VALUES:
        df[f"_qtype_{qt}"] = (df["qtype_name"] == qt).astype(np.float32)

    agg_dict = {
        "rtt": ["count", "mean", "std"],
        "_is_nxdomain": "mean",
        "_rejected": "max",
        "_has_answer": "mean",
        "_first_ttl": "mean",
        "query": "nunique",
    }
    for qt in DNS_QTYPE_VALUES:
        agg_dict[f"_qtype_{qt}"] = "sum"

    grouped = df.groupby(["uid", "class"]).agg(agg_dict)
    grouped.columns = [
        "dns_query_count", "dns_rtt_mean", "dns_rtt_std",
        "dns_nxdomain_ratio", "dns_rejected", "dns_has_answer",
        "dns_avg_ttl", "dns_unique_queries",
    ] + [f"dns_qtype_{qt}" for qt in DNS_QTYPE_VALUES]

    grouped.reset_index(inplace=True)
    return grouped


# ── Concurrency features ──────────────────────────────────────────────────────

def _compute_concurrency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-source-IP flow concurrency features.

    Expects id.orig_h, id.resp_h, id.resp_p to be present.
    Adds new columns and returns the DataFrame (identifiers still present).
    """
    src_col = "id.orig_h"

    src_counts = df[src_col].value_counts()
    df["src_flow_count"] = df[src_col].map(src_counts).astype(np.float32)

    df["unique_dest_count"] = (
        df.groupby(src_col)["id.resp_h"].transform("nunique").astype(np.float32)
    )

    df["unique_dest_port_count"] = (
        df.groupby(src_col)["id.resp_p"].transform("nunique").astype(np.float32)
    )

    port_ent = _port_entropy_per_source(df)
    df["dest_port_entropy"] = df[src_col].map(port_ent).astype(np.float32)

    return df


# ── Main loader ────────────────────────────────────────────────────────────────

def load_all_classes(logs_dir: Path = LOGS_DIR) -> pd.DataFrame:
    """Load combined Zeek CSVs from logs/, join, and return a labelled DataFrame."""
    logger.info("Loading combined Zeek logs from %s", logs_dir)

    # 1. Load conn (has 'class' column from combine step)
    conn = pd.read_csv(
        logs_dir / "conn.csv", na_values=["-"], dtype=str, low_memory=False
    )
    logger.info("  conn.csv: %d rows", len(conn))

    # 2. Concurrency features BEFORE dropping identifiers
    conn = _compute_concurrency(conn)

    # 3. Extract resp_port as a feature before dropping identifiers
    if "id.resp_p" in conn.columns:
        conn["resp_port"] = pd.to_numeric(conn["id.resp_p"], errors="coerce").astype(np.float32)

    # 4. Drop identifiers — keep uid and class for joins
    drop = [c for c in CONN_DROP_COLS if c in conn.columns and c not in ("uid", "class")]
    conn.drop(columns=drop, inplace=True)

    # 5. Cast conn numerics
    for col in CONN_NUMERIC_COLS:
        if col in conn.columns:
            conn[col] = pd.to_numeric(conn[col], errors="coerce").astype(np.float32)

    # 5. SSL + x509
    ssl_path = logs_dir / "ssl.csv"
    if ssl_path.exists():
        ssl = _load_ssl(ssl_path)

        x509_path = logs_dir / "x509.csv"
        if x509_path.exists() and "_first_cert_fp" in ssl.columns:
            x509 = _load_x509(x509_path)
            ssl = ssl.merge(
                x509, left_on="_first_cert_fp", right_on="fingerprint", how="left"
            )
            ssl.drop(columns=["_first_cert_fp", "fingerprint"], errors="ignore", inplace=True)
        else:
            ssl.drop(columns=["_first_cert_fp"], errors="ignore", inplace=True)

        conn = conn.merge(ssl, on=["uid", "class"], how="left")
        conn["has_ssl"] = conn["ssl_version"].notna().astype(np.int8)
    else:
        conn["has_ssl"] = np.int8(0)

    # 6. HTTP
    http_path = logs_dir / "http.csv"
    if http_path.exists():
        http_agg = _load_http(http_path)
        conn = conn.merge(http_agg, on=["uid", "class"], how="left")
        conn["has_http"] = conn["http_request_count"].notna().astype(np.int8)
    else:
        conn["has_http"] = np.int8(0)

    # 7. DNS
    dns_path = logs_dir / "dns.csv"
    if dns_path.exists():
        dns_agg = _load_dns(dns_path)
        conn = conn.merge(dns_agg, on=["uid", "class"], how="left")
        conn["has_dns"] = conn["dns_query_count"].notna().astype(np.int8)
    else:
        conn["has_dns"] = np.int8(0)

    # 8. Drop uid, rename class → label
    conn.drop(columns=["uid"], inplace=True)
    conn.rename(columns={"class": "label"}, inplace=True)

    conn = conn.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Total dataset: %d rows, %d columns", len(conn), len(conn.columns))
    return conn
