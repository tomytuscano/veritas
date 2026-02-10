## Project Overview: AI Traffic Classifier

The goal of this system is to identify encrypted and obfuscated network traffic that traditional Deep Packet Inspection (DPI) fails to catch. Instead of looking at the *content* of packets, the model analyzes the *behavior* and *shape* of the traffic.

### Core Technologies

- **NFStream:** A Python framework for fast, large-scale network traffic analysis and feature extraction.
- **Scikit-Learn:** Used for training the **Random Forest** classifier and hyperparameter tuning.
- **Shannon Entropy:** A mathematical feature used to distinguish between standard encryption and advanced obfuscation.

------

## Phase 1: Data Strategy & Acquisition

To prevent the model from simply "memorizing" IP addresses (overfitting), the system focuses on **Flow-Based Statistics**.

1. **Labeling:** Traffic is captured into PCAP files and organized into folders (`/vpn`, `/torrent`, `/normal`).
2. **Conversion:** The `NFStreamer` engine converts raw packets into bidirectional flows, extracting over 80 features.
3. **Feature Exclusion:** Identifiers like Source/Destination IP and Ports are removed to ensure the model learns protocol behavior rather than network topology.

------

## Phase 2: Feature Engineering (Entropy & Timing)

Encryption makes payloads look random. We implemented **Shannon Entropy** to quantify this randomness.

- **Formula:** $H = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$
- **Application:** High entropy ($H > 7.9$) typically indicates a VPN or Proxy tunnel, while lower entropy ($H < 5.0$) suggests unencrypted web traffic.
- **Timing:** The model utilizes **Standard Deviation of Packet Inter-Arrival Time (IAT)** to detect the "heartbeat" patterns of automated tunnels versus sporadic human browsing.

------

## Phase 3: Model Training & Optimization

We utilized a **Random Forest Classifier** because of its ability to handle non-linear relationships and provide feature importance metrics.

### Optimization Workflow

1. **GridSearchCV:** Used to automatically find the best `max_depth` and `n_estimators`.
2. **Feature Alignment:** Saved feature names into a `.pkl` file to ensure the live detection script exactly matches the training input structure.
3. **Validation:** Employed **K-Fold Cross-Validation** to ensure the model generalizes across different devices and networks.

------

## Phase 4: Production Deployment

The final system is deployed as a **Background Daemon** (Systemd service) that monitors live interfaces.

### The Inference Logic

- **Sniffing:** Live traffic is captured by an `NFPlugin`.
- **Confidence Thresholding:** The model only logs an alert if the prediction probability exceeds **90%**, reducing false positives on standard HTTPS traffic.
- **Logging:** Detections are written to `network_detections.log` with details on Source/Destination IPs and the detected application.

------

## Maintenance & Evolution

- **Adversarial Testing:** Use tools like `Obfsproxy` or `Shadowsocks` to test if the model can still identify hidden tunnels.
- **Model Retraining:** Periodically update the dataset as VPN protocols evolve to avoid "Model Drift."