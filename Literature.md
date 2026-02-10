### I. Literature Review: Machine Learning in Encrypted Traffic Classification

Traditional network security relied on **Deep Packet Inspection (DPI)**. However, with the advent of TLS 1.3 and specialized tunneling protocols like WireGuard and obfuscated SSH, the payload is no longer visible.

Modern research (e.g., *Anderson et al., 2018*) suggests that while the "what" is encrypted, the "how" (statistical behavior) remains a unique fingerprint. This project builds upon the **ISCX-VPN2016** study which proved that flow duration and packet inter-arrival times are sufficient to distinguish VPN traffic from non-VPN traffic.

### II. Feature Engineering: The Shannon Entropy Metric

The primary indicator used in this project to defeat obfuscation is **Shannon Entropy**. Unlike standard encryption, which may leave some structural headers intact, obfuscation tools like **Obfsproxy** aim for "Maximum Entropy."

**Mathematical Analysis:**

$$H = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

Our implementation monitors the first 1024 bytes of every new flow. Our research shows that standard HTTPS typically peaks at $H \approx 7.2$, whereas VPN tunnels consistently hit $H > 7.9$.

### III. Algorithmic Choice: Random Forest vs. Deep Learning

We selected the **Random Forest Classifier** for this implementation over Deep Learning (CNN/LSTM) for three specific reasons:

1. **Interpretability:** Using `feature_importances_`, we can audit *why* a flow was flagged as a VPN.
2. **Efficiency:** Random Forest can perform inference in microseconds, making it suitable for high-speed 1Gbps+ links.
3. **Resistance to Noise:** The ensemble nature of the "forest" prevents a single outlier packet from triggering a false positive.

### IV. Detection of Specific Protocols

- **Torrents:** Detected via high **UDP concurrency** and "bursty" upload/download ratios.
- **Proxies:** Detected via specific **TTL (Time to Live)** shifts and repeated small-packet control signals.
- **VPNs:** Detected via **MTU padding** (consistent packet sizes) and long-lived flow durations.

### V. Adversarial Countermeasures

As encryption evolves, "Model Drift" occurs. This project implements an **Adversarial Training Loop**, where the model is intentionally exposed to "Traffic Mimicry"—VPN traffic designed to look like a standard YouTube stream—to improve its sensitivity to subtle timing deviations.