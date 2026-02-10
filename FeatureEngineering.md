# Feature Engineering Architecture: Traffic Classification

**Pipeline Overview:** The system engineers **95 distinct features** derived from three core Zeek log sources (`conn`, `ssl`, `dns`). These features are designed to capture flow dynamics, cryptographic fingerprints, and protocol behavior.

### 1. Flow Volume & Shape (17 Features)

This group forms the backbone of the classification model, quantifying the "physicality" of the connection.

- **Direct Metrics:** `duration`, `orig/resp_bytes`, `orig/resp_pkts`, `orig/resp_ip_bytes`, `missed_bytes`, `ip_proto`.

- **Derived Metrics:** `total_bytes`, `total_pkts`, `bytes_ratio`, `pkts_ratio`, `avg_pkt_size_orig/resp`.

- **Overhead Metrics:** Calculated as:

  Overhead = (IP\_Bytes - Payload\_Bytes)/Packets

  *High overhead values indicate significant encapsulation (e.g., VPN headers).*

**Behavioral Signatures:**

| Traffic Class | Duration | Packet Size Variance | Byte Ratio | Protocol |

| :--- | :--- | :--- | :--- | :--- |

| **BitTorrent** | Variable | High (Bursty) | Asymmetric | Mostly UDP (17) |

| **VPN** | Long | Low (Uniform MTU padding) | ~1.0 (Bidirectional) | UDP/TCP |

| **Proxy** | Variable | Dips (Small control pkts) | Highly Asymmetric | TCP |

| **Normal** | Short-Med | High Variance | Variable | TCP/UDP |

### 2. Protocol & Service State (26 Features)

Encodes the negotiation state and service type using one-hot encoding.

- **Protocol:** `tcp`, `udp`, `icmp`.
- **Service:** `dns`, `http`, `ssl`, `dhcp`, `ntp`, `ssh`, `irc`, `ftp`, `smtp`, `other`.
- **Connection State:** 13 Zeek states (e.g., `S0`, `S1`, `SF`, `REJ`).

**Detection Logic:**

- **BitTorrent:** Generates massive `S0` counts (connection attempts to dead peers/trackers).
- **VPN:** Clusters heavily on `SF` (clean connection completion) combined with `service=ssl`.
- **Proxy:** Exhibits unusual service mixes and non-standard state transitions.

### 3. Connection History & Entropy (11 Features)

Analyzes the sequence of TCP flags (`S`=SYN, `F`=FIN, `D`=Data, `R`=Reset) to determine the complexity of the conversation.

- **Features:** `history_len`, `history_entropy`, and boolean flags for specific characters (e.g., `has_S`, `has_R`).

**Entropy Analysis (`history_entropy`):**

This metric quantifies the complexity of the flag sequence.

- **Low Entropy:** Simple scans or floods (e.g., `S`, `Sr`).
- **High Entropy:** Sustained, encrypted tunnels with complex bidirectional data exchange.
- **Signatures:**
  - *BitTorrent:* Short history, frequent `D` (data) and `R` (resets).
  - *Normal:* Standard "ShADadfF" (Handshake $\rightarrow$ Data $\rightarrow$ Teardown).

### 4. SSL/TLS Fingerprinting (19 Features)

Crucial for differentiating encrypted tunnels (VPNs) from standard HTTPS web traffic.

- **Categorical:** `ssl_version` (Ordinal 0-5), `cipher_family`, `curve`, `next_protocol`.
- **Entropy Metrics:** `ssl_ja3_entropy`, `ssl_ja3s_entropy`, `ssl_sni_entropy`, `ssl_history_entropy`.
- **Validation:** `ssl_sni_match` (Does SNI match the Cert?), `ssl_resumed`.

**The "Why" - Encrypted Traffic Differentiation:**

- **JA3/JA3S:** VPN clients (WireGuard, OpenVPN) use distinct TLS stacks, producing JA3 hashes different from standard browsers (Chrome/Firefox).
- **SNI Entropy:** VPNs often use random/high-entropy server names or lack SNI entirely, whereas normal traffic targets recognizable domains (e.g., `google.com`).
- **Cipher Selection:** VPNs prefer high-performance suites like `CHACHA20` (WireGuard) or `AES-GCM` (OpenVPN).

### 5. DNS Aggregation (14 Features)

Summarizes DNS intent associated with the connection IP.

- **Metrics:** `query_count`, `rtt_mean/std`, `unique_queries`, `nxdomain_ratio`, `avg_ttl`.
- **Query Types:** Distribution of `A`, `AAAA`, `PTR`, `CNAME`, `MX`, `TXT`, `SRV`.

**Behavioral Signatures:**

- **BitTorrent:** High `nxdomain_ratio` (churning tracker domains) and low TTLs.
- **VPN:** `has_dns=0` (Resolution occurs inside the tunnel).
- **Proxy:** Sparse local DNS footprint; queries are offloaded to the proxy resolver.

### 6. Join Indicators & Handling

- **Signal Features:** `has_ssl`, `has_dns`.
  - *VPN Strong Signal:* `has_ssl=1` AND `has_dns=0` (Encrypted tunnel, no local DNS).
  - *Torrent Strong Signal:* `has_ssl=0` AND `has_dns=1` (Plaintext UDP with tracker lookups).
- **NaN Strategy:** All missing values are imputed with `-1`. This allows tree-based models (e.g., Random Forest) to isolate "missing" data as a distinct informative branch (e.g., `ssl_ja3_entropy <= -0.5` effectively splits on "No SSL Present").