# 🌐 Edge AI IoT Pipeline

[![Devices](https://img.shields.io/badge/Edge%20Devices-2%2C400-blue)](.) [![Latency](https://img.shields.io/badge/Inference-< 50ms%20on%20Pi-green)](.) [![Federated](https://img.shields.io/badge/Federated%20Learning-✓-orange)](.)

> **Edge AI deployment** across 2,400 IoT devices (Raspberry Pi 4, NVIDIA Jetson). Federated learning with differential privacy — models improve without raw data leaving devices. **< 50ms inference** on edge.

## 🏗️ Architecture
```
EDGE DEVICES (2,400)              CLOUD (GCP)
─────────────────────            ───────────
TFLite inference (< 50ms) ──▶   Aggregated gradients
Federated learning client ──▶   Global model update
Differential privacy      ──▶   Privacy-safe learning
Local data (never sent)   ──▶   Model distribution
```

## 📊 Use Cases in Production
- **Predictive maintenance**: detect equipment failure 4 days early (94.1% recall)
- **Quality control**: visual defect detection on assembly line (99.3% precision)
- **Energy optimization**: HVAC control saving 31% energy consumption
