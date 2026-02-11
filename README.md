# Veritas — AI Traffic Classifier

Veritas identifies encrypted and obfuscated network traffic (VPNs, proxies, BitTorrent) using flow-based behavioral analysis. Instead of Deep Packet Inspection, it analyzes traffic shape and behavior from Zeek network logs using a CUDA-accelerated LightGBM classifier.

## Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| BitTorrent | 99.99% | 99.99% | 99.99% | 145,426 |
| Normal | 99.99% | 99.99% | 99.99% | 125,573 |
| Proxy | 99.52% | 99.86% | 99.69% | 130,319 |
| VPN | 99.84% | 99.44% | 99.64% | 113,574 |
| **Overall** | **99.84%** | **99.84%** | **99.84%** | **514,892** |

- 5-fold cross-validation F1 macro: 99.82%
- 99.68% of predictions made at >= 90% confidence with 100% accuracy

## How It Works

1. **Data Loading** — Zeek CSV logs (conn, ssl, x509, http, dns) are joined on flow UID across 4 traffic classes
2. **Feature Engineering** — 133 features extracted: flow metrics, protocol/service encoding, TCP history flags, SSL/TLS fingerprinting (JA3, cipher families, SNI entropy), x509 certificates, DNS/HTTP aggregations, concurrency statistics
3. **Training** — LightGBM with GridSearchCV (48 param combos x 5 folds), GPU-accelerated via CUDA
4. **Inference** — Confidence-scored predictions with a 90% threshold to minimize false positives

## Setup

```bash
# Create and activate virtual environment
python3.12 -m venv vertitas_env
source vertitas_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install LightGBM with CUDA support (requires NVIDIA GPU + CUDA toolkit + GCC 12)
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
  pip install --no-binary lightgbm lightgbm \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
```

## Usage

### Training
```bash
# Full pipeline
python main.py

# With subsampling for faster tuning
python main.py --subsample 50000
```

### Inference
```bash
python inference.py
```

Or use programmatically:
```python
from inference import load_model, predict

model, feature_names, class_labels = load_model()
results = predict(model, X, class_labels)
# Each result: {"label": "vpn", "confidence": 0.97, "is_confident": True}
```

## Project Structure

```
main.py                  # Pipeline orchestration
config.py                # All constants and hyperparameter grid
data_loader.py           # Zeek CSV loading and multi-table joins
feature_engineering.py   # 133 feature extraction
training.py              # CUDA-accelerated LightGBM + GridSearchCV
evaluation.py            # Reports, confusion matrix, feature importance
inference.py             # Model loading and prediction
models/                  # Trained artifacts and evaluation outputs
```

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA toolkit
- Zeek network logs organized by traffic class
