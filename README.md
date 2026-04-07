<p align="center">
  <h1 align="center">⚙️ Bearing Fault Detection System</h1>
  <p align="center">
    <strong>Production-Ready Deep Learning for Predictive Maintenance</strong>
  </p>
  <p align="center">
    <em>Real-time bearing health diagnosis using 1D-CNN on the CWRU dataset with explainable AI, a REST API, and a live monitoring dashboard.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset — CWRU Bearing Data](#-dataset--cwru-bearing-data)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Architecture — VibrationCNN](#-model-architecture--vibrationcnn)
- [Training Pipeline](#-training-pipeline)
- [Explainability — Grad-CAM](#-explainability--grad-cam)
- [API & Dashboard](#-api--dashboard)
- [Notebooks](#-notebooks)
- [Deployment](#-deployment)
- [Technical Highlights](#-technical-highlights)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🔍 Overview

Bearing failures account for **over 40% of all rotating machinery breakdowns** in industrial settings, leading to costly unplanned downtime and safety hazards. This project presents a **complete, end-to-end deep learning system** for real-time bearing fault detection and diagnosis.

The system processes raw vibration signals from accelerometers, classifies them into **10 distinct health states** (1 normal + 9 fault conditions), and provides **explainable predictions** with Grad-CAM activation maps — enabling maintenance engineers to trust and act on AI-driven recommendations.

### What Makes This Project Stand Out

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning** | Custom 1D-CNN designed for raw vibration signal processing |
| 🔒 **Leak-Proof Evaluation** | Hybrid split strategy (file-based + temporal) eliminates data leakage |
| 🔍 **Explainable AI** | Grad-CAM visualizations map model decisions to physical fault signatures |
| ⚡ **Real-Time Ready** | Sub-100ms inference with ONNX Runtime optimization |
| 🌐 **Production API** | FastAPI REST endpoint with input validation, batch prediction, and health monitoring |
| 📊 **Live Dashboard** | Streamlit-based monitoring with signal analysis, frequency spectrum, and alert system |
| 📓 **Fully Documented** | 6 detailed Jupyter notebooks covering every step from EDA to deployment |

---

## 🏆 Key Results

<table>
<tr>
<td>

| Metric | Value |
|--------|------:|
| **Test Accuracy** | **>95%** |
| **False Negative Rate** | **<1%** |
| **Inference Latency** | **<10 ms** |
| **Model Size** | **~0.8 MB** |
| **Parameters** | **~210K** |

</td>
<td>

| Target | Status |
|--------|--------|
| Accuracy >95% | ✅ Achieved |
| Latency <100 ms | ✅ Achieved |
| Explainability | ✅ Grad-CAM |
| REST API | ✅ FastAPI |
| Edge-ready | ✅ ONNX + INT8 |

</td>
</tr>
</table>

> **Safety-Critical Performance**: The system achieves a False Negative Rate below 1%, meaning it misses fewer than 1 in 100 actual faults — critical for industrial predictive maintenance where a missed fault can cause catastrophic equipment failure.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    BEARING FAULT DETECTION SYSTEM                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│   │  Vibration   │    │   Preprocessing  │    │   1D-CNN Model   │      │
│   │   Sensor     │───▶│  • Bandpass Filt │───▶│  • 3 Conv Blocks │      │
│   │  (12 kHz)    │    │  • Windowing     │    │  • GAP + FC      │      │
│   └─────────────┘    │  • Z-Score Norm   │    │  • 10-class out  │      │
│                       └──────────────────┘    └───────┬──────────┘      │
│                                                       │                  │
│                          ┌────────────────────────────┼────────┐        │
│                          │                            │        │        │
│                          ▼                            ▼        ▼        │
│                   ┌─────────────┐           ┌──────────┐ ┌──────────┐  │
│                   │  Grad-CAM   │           │ FastAPI  │ │Streamlit │  │
│                   │  XAI Engine │           │ REST API │ │Dashboard │  │
│                   └─────────────┘           └──────────┘ └──────────┘  │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Data Flow**: Raw vibration → Bandpass filter (10–5000 Hz) → 2048-sample windows (50% overlap) → Z-score normalization → 1D-CNN → Fault class + confidence + alert level

---

## 📂 Dataset — CWRU Bearing Data

This project uses the [Case Western Reserve University (CWRU) Bearing Dataset](https://engineering.case.edu/bearingdatacenter/download-data-file), the gold-standard benchmark for bearing fault diagnosis.

**Bearing**: 6205-2RS JEM SKF deep groove ball bearing @ 1797 RPM

<table>
<tr><th>Class ID</th><th>Condition</th><th>Fault Size</th><th>Fault Frequency</th></tr>
<tr><td>0</td><td>✅ Normal</td><td>—</td><td>—</td></tr>
<tr><td>1</td><td>🔴 Inner Race</td><td>0.007"</td><td>BPFI ≈ 162 Hz</td></tr>
<tr><td>2</td><td>🔴 Inner Race</td><td>0.014"</td><td>BPFI ≈ 162 Hz</td></tr>
<tr><td>3</td><td>🔴 Inner Race</td><td>0.021"</td><td>BPFI ≈ 162 Hz</td></tr>
<tr><td>4</td><td>🔵 Outer Race</td><td>0.007"</td><td>BPFO ≈ 107 Hz</td></tr>
<tr><td>5</td><td>🔵 Outer Race</td><td>0.014"</td><td>BPFO ≈ 107 Hz</td></tr>
<tr><td>6</td><td>🔵 Outer Race</td><td>0.021"</td><td>BPFO ≈ 107 Hz</td></tr>
<tr><td>7</td><td>🟡 Ball</td><td>0.007"</td><td>BSF ≈ 140 Hz</td></tr>
<tr><td>8</td><td>🟡 Ball</td><td>0.014"</td><td>BSF ≈ 140 Hz</td></tr>
<tr><td>9</td><td>🟡 Ball</td><td>0.021"</td><td>BSF ≈ 140 Hz</td></tr>
</table>

- **40 `.mat` files** — 4 per class, drive-end accelerometer signals
- **12 kHz** sampling rate (48 kHz downsampled for files ≥ 169)
- ~10 seconds per recording → thousands of 2048-sample windows

---

## 📁 Project Structure

```
bearing-fault-detection/
│
├── config/
│   └── train_config.yaml         # Hyperparameters & experiment config
│
├── data/
│   └── cwru/                     # CWRU .mat files (97.mat, 105.mat, ...)
│
├── models/
│   ├── best_model.pth            # Best checkpoint (by validation acc)
│   └── final_model.pth           # Final epoch checkpoint
│
├── experiments/
│   └── training_results.png      # Loss/accuracy curves
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Dataset EDA
│   ├── 02_preprocessing_pipeline.ipynb    # Signal processing deep-dive
│   ├── 03_model_training.ipynb            # Architecture & training
│   ├── 04_evaluation_analysis.ipynb       # Metrics & confusion matrices
│   ├── 05_explainability_gradcam.ipynb    # Grad-CAM & interpretability
│   └── 06_deployment_optimization.ipynb   # ONNX export & benchmarks
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── data_loader.py        # CWRUDataLoader — .mat file parsing
│   │   └── preprocessing.py      # Filtering, normalization, splitting
│   ├── models/
│   │   └── vibration_cnn.py      # VibrationCNN architecture
│   ├── training/
│   │   ├── train.py              # Training loop + BearingDataset
│   │   └── evaluate.py           # ModelEvaluator + metrics
│   ├── interpretation/
│   │   └── gradcam.py            # GradCAM1D + filter visualization
│   ├── api/
│   │   └── app.py                # FastAPI REST endpoint
│   └── dashboard/
│       └── app.py                # Streamlit monitoring dashboard
│
├── main.py                       # One-command training entry point
├── requirements.txt              # Dependencies
├── setup.py                      # Package installation
└── README.md                     # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (optional, CPU works fine)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/krishbansal-2205/bearing-fault-detection.git
cd bearing-fault-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in editable mode
pip install -e .
```

### Download the Dataset

Download the CWRU `.mat` files from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) and place them in `data/cwru/`:

```
data/cwru/
├── 97.mat   # Normal (0 HP)
├── 98.mat   # Normal (1 HP)
├── 105.mat  # Inner Race 0.007" 
├── ...
└── 237.mat  # Outer Race 0.021"
```

---

## 💻 Usage

### Train the Model

```bash
# Train with default config (hybrid split, 50 epochs)
python main.py
```

Configuration is managed via `config/train_config.yaml`:

```yaml
data:
  window_size: 2048
  overlap: 0.5
  split_method: 'hybrid'       # 'time_based', 'file_based', or 'hybrid'
  batch_size: 64

model:
  num_classes: 10
  dropout_rate: 0.5

training:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.01
  early_stopping_patience: 10
  augment_train: true
```

### Launch the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then test with:
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@vibration_signal.csv" \
  -F "sampling_rate=12000"

# Health check
curl http://localhost:8000/health
```

### Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🧠 Model Architecture — VibrationCNN

A lightweight **1D Convolutional Neural Network** designed specifically for raw vibration signal classification:

```
Input: [batch, 1, 2048]  — single-channel vibration signal (0.17 seconds at 12 kHz)

 ┌─────────────────────────────────────────────────────────────────┐
 │  CONVOLUTIONAL FEATURE EXTRACTOR                                │
 │                                                                  │
 │  Block 1: Conv1d(1→32, k=64, s=8) → BN → ReLU → Drop(0.2)    │
 │           → MaxPool(4)                     Output: [B, 32, 32]  │
 │                                                                  │
 │  Block 2: Conv1d(32→64, k=32, s=1) → BN → ReLU → Drop(0.3)   │
 │           → MaxPool(4)                     Output: [B, 64, 8]   │
 │                                                                  │
 │  Block 3: Conv1d(64→128, k=16, s=1) → BN → ReLU → Drop(0.4)  │
 │           → AdaptiveAvgPool(1)             Output: [B, 128, 1]  │
 └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CLASSIFIER                                                      │
 │  Flatten → Linear(128→64) → ReLU → Dropout(0.5)                │
 │         → Linear(64→10)                    Output: [B, 10]      │
 └─────────────────────────────────────────────────────────────────┘
```

### Design Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| **1D-CNN** over 2D | Processes raw time-series directly | No spectrogram computation needed, lower latency |
| **Large first kernel (k=64)** | Captures ~5.3 ms of signal | Spans multiple fault impulse cycles |
| **Progressive dropout** (0.2→0.5) | Stronger regularization deeper | Prevents overfitting on small dataset |
| **Global Average Pooling** | Translation invariance | Fault impulses can appear at any position |
| **AdamW + Cosine LR** | Fast, stable convergence | Label smoothing (0.1) prevents overconfidence |
| **~210K parameters** | Lightweight model | Sub-10ms inference, <1 MB size |

---

## 🔧 Training Pipeline

### Preprocessing Flow

```
Raw .mat file → Downsample (48→12 kHz if needed)
             → Bandpass filter (10–5000 Hz, Butterworth 4th order, zero-phase)
             → Data split (hybrid: file-based + temporal)
             → Windowing (2048 samples, 50% overlap)
             → Per-window Z-score normalization
             → PyTorch Tensor [B, 1, 2048]
             → Online augmentation (noise + random scaling)
```

### Data Splitting Strategies

The system supports three split methods to prevent **data leakage** — a common pitfall in vibration-based ML:

| Strategy | How It Works | Leakage Risk |
|----------|-------------|:------------:|
| **File-based** | Stratified split by recording file | ✅ Low |
| **Time-based** | First 70% → train, last 30% → test (per signal) | ✅ Low |
| **Hybrid** ⭐ | File split + temporal within files | ✅✅ Lowest |

> ⚠️ **Why not random window split?** Adjacent windows from the same recording share >90% of their samples. Random splitting would leak training data into the test set, resulting in inflated accuracy (99%+) that doesn't generalize.

### Training Features

- **Label Smoothing** (ε=0.1) — prevents overconfident predictions
- **Gradient Clipping** (max_norm=1.0) — stabilizes training with varying signal amplitudes
- **Cosine Annealing LR** — smooth learning rate decay
- **Early Stopping** (patience=10) — prevents overfitting
- **Online Data Augmentation** — Gaussian noise injection + random amplitude scaling

---

## 🔍 Explainability — Grad-CAM

The system includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** adapted for 1D signals, providing visual explanations of model predictions:

```
Grad-CAM(x) = ReLU( Σ_k  α_k · A^k )

where α_k = GAP(∂y^c / ∂A^k)  — gradient-weighted importance of each channel
```

### What It Shows

- **Red regions** = temporal segments most important for the prediction
- **Fault signals** → periodic high-activation spikes aligned with fault impulse frequencies
- **Normal signals** → uniform, low activation (no dominant fault features)

### Validation Against Domain Knowledge

| Fault Type | Expected Pattern | Model Behavior |
|------------|-----------------|----------------|
| Inner Race | Periodic impulses at BPFI (~162 Hz) | ✅ Activations match BPFI periodicity |
| Outer Race | Regular impacts at BPFO (~107 Hz) | ✅ Activations match BPFO spacing |
| Ball | Random-interval impulses at BSF (~140 Hz) | ✅ Scattered activations consistent with BSF |
| Normal | No dominant pattern | ✅ Low, uniform activation |

---

## 🌐 API & Dashboard

### FastAPI REST API (`src/api/app.py`)

Production-ready endpoint with:

- **`POST /predict`** — Single signal fault classification
- **`POST /predict/batch`** — Batch prediction
- **`GET /health`** — Service health check
- **`GET /classes`** — Supported fault classes
- **`GET /docs`** — Interactive Swagger documentation

**Response example:**
```json
{
  "predicted_class": "Inner_Race_007",
  "predicted_class_id": 1,
  "confidence": 0.987,
  "alert_level": "RED - Critical Fault Detected",
  "top_3_predictions": [
    {"class": "Inner_Race_007", "class_id": 1, "confidence": 0.987},
    {"class": "Inner_Race_014", "class_id": 2, "confidence": 0.008},
    {"class": "Normal", "class_id": 0, "confidence": 0.003}
  ],
  "processing_time_ms": 4.23,
  "timestamp": "2026-04-07T18:30:00"
}
```

### Streamlit Dashboard (`src/dashboard/app.py`)

Real-time monitoring interface featuring:

- 📊 **Signal analysis** — time-domain waveform visualization
- 📈 **Frequency spectrum** — PSD with fault frequency markers (BPFI, BPFO, BSF)
- 🔍 **Fault diagnosis** — live classification with confidence gauge and alert levels
- 📋 **Feature extraction** — RMS, kurtosis, crest factor, radar chart
- 📜 **Prediction history** — session-based tracking and trends
- 🎯 **Demo mode** — synthetic signal generation for all 10 fault types

---

## 📓 Notebooks

Six comprehensive Jupyter notebooks provide a **complete walkthrough** of the project:

| # | Notebook | Focus |
|---|----------|-------|
| 01 | **Data Exploration** | Dataset loading, signal visualization, FFT/PSD analysis, spectrograms, statistical features, class distributions |
| 02 | **Preprocessing Pipeline** | Bandpass filter design, normalization comparison, windowing, split strategies, data leakage verification |
| 03 | **Model Training** | Architecture deep-dive, receptive field math, training curves, learning rate schedule, latency benchmarks |
| 04 | **Evaluation Analysis** | Classification report, confusion matrices, ROC/AUC curves, safety metrics (FNR/FPR), t-SNE of learned features |
| 05 | **Explainability** | Grad-CAM heatmaps, severity progression, learned filter visualization, saliency maps, domain validation |
| 06 | **Deployment** | ONNX export, PyTorch vs ONNX benchmark, INT8 quantization, noise robustness, deployment checklist |

---

## 🚢 Deployment

### ONNX Runtime (Recommended)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/bearing_fault_detector.onnx")
signal = np.random.randn(1, 1, 2048).astype(np.float32)  # preprocessed signal
result = session.run(None, {"vibration_signal": signal})
logits = result[0]  # [1, 10]
```

### Docker (API)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && pip install -e .
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Model Formats Available

| Format | Size | Use Case |
|--------|------|----------|
| PyTorch (`.pth`) | ~0.8 MB | Training, fine-tuning |
| ONNX (`.onnx`) | ~0.8 MB | Cross-platform production inference |
| INT8 Quantized | ~0.3 MB | Edge/embedded devices |

---

## ⚡ Technical Highlights

<details>
<summary><strong>🔒 Data Leakage Prevention</strong></summary>

Unlike many CWRU-based projects that randomly shuffle windows (achieving misleading 99%+ accuracy), this project implements a **hybrid split** strategy:
1. Files are split per-class (stratified) — no file appears in both train and test
2. Train files use the **early 70%** of each recording
3. Test files use the **late 30%** of each recording

This simulates real-world deployment: training on early operational data, testing on later/degraded data.
</details>

<details>
<summary><strong>🧱 Modular Architecture</strong></summary>

The codebase is cleanly separated into independent modules:
- `src/data/` — Data loading and preprocessing (filter, normalize, window, split)
- `src/models/` — Neural network architecture
- `src/training/` — Training loop, evaluation, early stopping
- `src/interpretation/` — Grad-CAM XAI engine
- `src/api/` — Production REST API
- `src/dashboard/` — Real-time monitoring UI

Each module has its own `__init__.py` with clean public exports.
</details>

<details>
<summary><strong>📊 Comprehensive Evaluation</strong></summary>

Beyond standard accuracy, the system tracks:
- **Per-class precision/recall/F1** — identifies weak classes
- **False Negative Rate** — safety-critical: faults that go undetected
- **False Positive Rate** — operational: unnecessary maintenance alerts
- **Prediction confidence** — calibration analysis
- **ROC/AUC** — per-class discrimination ability
- **t-SNE embeddings** — learned feature separability
</details>

<details>
<summary><strong>🌐 Production-Ready API</strong></summary>

The FastAPI endpoint includes:
- CORS middleware for frontend integration
- Request/response validation with Pydantic models
- Request logging middleware
- ONNX Runtime support with PyTorch fallback
- Batch prediction endpoint
- Proper error handling and HTTP status codes
- Auto-generated Swagger/OpenAPI documentation
</details>

---

## 🔮 Future Work

- [ ] Multi-sensor fusion (drive-end + fan-end accelerometers)
- [ ] Remaining Useful Life (RUL) prediction
- [ ] Transfer learning for different bearing types
- [ ] TensorRT optimization for NVIDIA Jetson edge deployment
- [ ] Continuous learning pipeline with MLflow tracking
- [ ] Attention mechanism integration (Transformer-based)
- [ ] Real-time streaming with Apache Kafka integration

---

## 📚 References

1. **Smith, W.A. & Randall, R.B.** (2015). Rolling element bearing diagnostics using the Case Western Reserve University data. *Mechanical Systems and Signal Processing*, 64-65, 100-131.

2. **Selvaraju, R.R. et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.

3. **CWRU Bearing Data Center** — [https://engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)

4. **Zhang, W. et al.** (2017). A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals. *Sensors*, 17(2), 425.

---

