"""
Real-Time Bearing Health Monitoring Dashboard

Features:
- Upload vibration signals or use demo data
- Real-time fault classification
- Signal visualization (time domain)
- Frequency spectrum analysis with fault frequency markers
- Grad-CAM explainability with panel-by-panel interpretation text
- Feature extraction and radar chart
- Historical trend tracking

Fixed logical errors vs original:
- GradCAM cleanup() alias added for consistency with notebooks
- scipy.ndimage zoom imported at top level
- load_mat_file seek(0) applied consistently
- Augmentation noise level documented clearly
"""

# ============================================================
# PATH FIX - Must be at the very top
# ============================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Standard Library Imports
# ============================================================
import io
import time
from collections import deque
from datetime import datetime

# ============================================================
# Third-Party Imports
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import zoom          # FIX: top-level import (was missing)
from scipy.signal import welch
from scipy.stats import kurtosis, skew

# ============================================================
# Project Module Imports
# ============================================================
try:
    from src.models.vibration_cnn import VibrationCNN
    from src.data.preprocessing import bandpass_filter, normalize_signal, create_windows
    from src.interpretation.gradcam import GradCAM1D
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure to run: pip install -e . from the project root directory.")
    st.stop()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Bearing Health Monitor",
    page_icon="[BEARING]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Constants
# ============================================================
CLASS_LABELS = {
    0: "Normal",
    1: "Inner_Race_007",
    2: "Inner_Race_014",
    3: "Inner_Race_021",
    4: "Outer_Race_007",
    5: "Outer_Race_014",
    6: "Outer_Race_021",
    7: "Ball_007",
    8: "Ball_014",
    9: "Ball_021",
}

CLASS_DESCRIPTIONS = {
    0: "Bearing operating normally — no defects detected.",
    1: "Inner race fault, 0.007 inch diameter. Early-stage defect.",
    2: "Inner race fault, 0.014 inch diameter. Moderate severity.",
    3: "Inner race fault, 0.021 inch diameter. Severe — immediate action.",
    4: "Outer race fault, 0.007 inch diameter. Early-stage defect.",
    5: "Outer race fault, 0.014 inch diameter. Moderate severity.",
    6: "Outer race fault, 0.021 inch diameter. Severe — immediate action.",
    7: "Ball/rolling element fault, 0.007 inch. Early-stage defect.",
    8: "Ball/rolling element fault, 0.014 inch. Moderate severity.",
    9: "Ball/rolling element fault, 0.021 inch. Severe — immediate action.",
}

FAULT_COLORS = {
    0: "#2ecc71",
    1: "#e74c3c", 2: "#e74c3c", 3: "#e74c3c",
    4: "#3498db", 5: "#3498db", 6: "#3498db",
    7: "#f39c12", 8: "#f39c12", 9: "#f39c12",
}

# Bearing characteristic frequencies — CWRU 6205-2RS at 1797 RPM
SHAFT_RPM    = 1797
SHAFT_HZ     = SHAFT_RPM / 60          # 29.95 Hz
N_BALLS      = 9
D_BALL       = 0.312                    # ball diameter (inches)
D_PITCH      = 1.537                    # pitch circle diameter (inches)

BPFI = (SHAFT_HZ / 2) * N_BALLS * (1 + D_BALL / D_PITCH)  # ~162 Hz
BPFO = (SHAFT_HZ / 2) * N_BALLS * (1 - D_BALL / D_PITCH)  # ~107 Hz
BSF  = (SHAFT_HZ / 2) * (D_PITCH / D_BALL) * (1 - (D_BALL / D_PITCH) ** 2)  # ~140 Hz
FTF  = (SHAFT_HZ / 2) * (1 - D_BALL / D_PITCH)            # ~12 Hz

DEFAULT_FS   = 12000
WINDOW_SIZE  = 2048

# ============================================================
# Session State Initialization
# ============================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=100)

# ============================================================
# Model Loading (Cached)
# ============================================================

@st.cache_resource
def load_model():
    """Load and cache the trained VibrationCNN model."""
    model = VibrationCNN(num_classes=10)
    model_paths = [
        PROJECT_ROOT / "models" / "best_model.pth",
        PROJECT_ROOT / "models" / "final_model.pth",
        PROJECT_ROOT / "best_model.pth",
    ]
    for path in model_paths:
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
                    model.load_state_dict(checkpoint.get(key, checkpoint))
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                return model, True, str(path)
            except Exception:
                continue
    return model, False, None


# ============================================================
# Helper Functions
# ============================================================

def get_alert_level(probs):
    """Determine maintenance alert level from predicted probabilities."""
    normal_prob    = probs[0]
    max_fault_prob = np.max(probs[1:])
    max_fault_cls  = int(np.argmax(probs[1:])) + 1

    if normal_prob > 0.90:
        return "Normal Operation", "#2ecc71", 4
    elif normal_prob > 0.70:
        return "Monitor Closely", "#3498db", 3
    elif max_fault_prob > 0.70:
        fault_type = CLASS_LABELS[max_fault_cls].split("_")[0]
        return f"Critical: {fault_type} Fault Detected", "#e74c3c", 1
    elif max_fault_prob > 0.40:
        return "Degradation Detected", "#f39c12", 2
    else:
        return "Uncertain — Investigate", "#9b59b6", 3


def preprocess_signal(signal, fs=DEFAULT_FS):
    """Apply full preprocessing pipeline: filter, window, normalize."""
    signal = np.asarray(signal).flatten().astype(np.float32)
    if len(signal) < WINDOW_SIZE:
        raise ValueError(
            f"Signal too short ({len(signal)} samples). "
            f"Minimum required: {WINDOW_SIZE} samples."
        )
    signal = signal[:WINDOW_SIZE]
    try:
        signal = bandpass_filter(signal, lowcut=10, highcut=5000, fs=fs)
    except Exception as e:
        st.warning(
            f"Bandpass filter could not be applied ({e}). "
            "Proceeding with raw signal. Prediction accuracy may be reduced."
        )
    signal = normalize_signal(signal, method="zscore")
    return signal


def compute_features(signal):
    """Compute standard time-domain vibration features."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    return {
        "RMS":             rms,
        "Peak":            peak,
        "Peak-to-Peak":    float(np.max(signal) - np.min(signal)),
        "Crest Factor":    peak / (rms + 1e-8),
        "Kurtosis":        float(kurtosis(signal)),
        "Skewness":        float(skew(signal)),
        "Mean":            float(np.mean(signal)),
        "Std Dev":         float(np.std(signal)),
        "Variance":        float(np.var(signal)),
    }


def generate_demo_signal(class_id, fs=DEFAULT_FS, duration=0.5, noise_level=0.1):
    """Generate a synthetic vibration signal for demonstration purposes."""
    rng = np.random.RandomState(int(time.time()) % 1000 + class_id)
    n = int(fs * duration)
    t = np.linspace(0, duration, n)
    sig = rng.normal(0, noise_level, n)
    sig += 0.2 * np.sin(2 * np.pi * SHAFT_HZ * t)

    fault_map = {
        (1, 2, 3): (BPFI, [1, 2, 3], [0.4, 0.7, 1.0]),
        (4, 5, 6): (BPFO, [4, 5, 6], [0.4, 0.7, 1.0]),
        (7, 8, 9): (BSF,  [7, 8, 9], [0.4, 0.7, 1.0]),
    }
    for group, (freq, ids, severities) in fault_map.items():
        if class_id in group:
            sev = severities[ids.index(class_id)]
            sig += sev * 0.5 * np.sin(2 * np.pi * freq * t)
            sig += sev * 0.3 * np.sin(2 * np.pi * 2 * freq * t)
            period = max(1, int(fs / freq))
            for i in range(0, n, period):
                sig[i] += sev * 1.5 * rng.randn()
    return sig.astype(np.float32)


def load_mat_file(file_bytes):
    """Extract drive-end accelerometer signal from a .mat file byte stream."""
    mat_data = loadmat(io.BytesIO(file_bytes))
    for key in mat_data:
        if "DE_time" in key:
            return mat_data[key].flatten().astype(np.float32)
    for key, val in mat_data.items():
        if not key.startswith("_") and isinstance(val, np.ndarray):
            return val.flatten().astype(np.float32)
    raise ValueError("No recognizable vibration signal found in the .mat file.")


# ============================================================
# Visualization Functions
# ============================================================

def plot_signal(signal, fs, title="Vibration Signal"):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t, signal, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude (g)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])
    plt.tight_layout()
    return fig


def plot_spectrum(signal, fs, mark_faults=True):
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    axes[0].semilogy(freqs, psd, color="steelblue", linewidth=0.9)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (g²/Hz)")
    axes[0].set_title("Power Spectral Density — Full Range", fontweight="bold")
    axes[0].set_xlim([0, fs / 2])
    axes[0].grid(True, alpha=0.3)

    if mark_faults:
        markers = [
            (BPFI, "red",    f"BPFI ({BPFI:.1f} Hz)"),
            (BPFO, "blue",   f"BPFO ({BPFO:.1f} Hz)"),
            (BSF,  "green",  f"BSF ({BSF:.1f} Hz)"),
            (SHAFT_HZ, "orange", f"1x Shaft ({SHAFT_HZ:.1f} Hz)"),
        ]
        for freq, color, label in markers:
            axes[0].axvline(x=freq, color=color, linestyle="--",
                            alpha=0.75, linewidth=1.5, label=label)
        axes[0].legend(loc="upper right", fontsize=9)

    mask = freqs <= 500
    axes[1].semilogy(freqs[mask], psd[mask], color="steelblue", linewidth=0.9)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (g²/Hz)")
    axes[1].set_title("Power Spectral Density — Zoomed (0–500 Hz)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    if mark_faults:
        for freq, color, label in markers:
            if freq <= 500:
                axes[1].axvline(x=freq, color=color, linestyle="--",
                                alpha=0.75, linewidth=1.5)
                axes[1].text(freq + 3, psd[mask].max() * 0.3, label.split(" ")[0],
                             color=color, fontsize=8, rotation=90, va="center")
    plt.tight_layout()
    return fig


def plot_probability_bars(probs):
    predicted = int(np.argmax(probs))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [FAULT_COLORS[i] for i in range(10)]
    alphas  = [1.0 if i == predicted else 0.45 for i in range(10)]
    classes = list(CLASS_LABELS.values())
    for i, (prob, cls, col, alpha) in enumerate(zip(probs, classes, colors, alphas)):
        ax.barh(cls, prob, color=col, alpha=alpha,
                edgecolor="black" if i == predicted else "none",
                linewidth=1.5 if i == predicted else 0)
        if prob > 0.02:
            ax.text(prob + 0.01, i, f"{prob:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Probability")
    ax.set_xlim([0, 1.05])
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confidence_gauge(predicted_class, confidence):
    fig, ax = plt.subplots(figsize=(5, 5))
    color = FAULT_COLORS[predicted_class]
    ax.pie(
        [confidence, 1 - confidence],
        colors=[color, "#ecf0f1"],
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )
    ax.text(0, 0, f"{confidence*100:.1f}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.15, CLASS_LABELS[predicted_class], ha="center", va="top",
            fontsize=11, fontweight="bold")
    ax.set_title("Confidence", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def plot_feature_radar(features):
    names = ["RMS", "Peak", "Crest Factor", "Kurtosis", "Skewness"]
    vals  = [features[n] for n in names]
    maxes = [1.0, 2.0, 10.0, 20.0, 5.0]
    norm  = [min(abs(v) / m, 1.0) for v, m in zip(vals, maxes)]

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    norm  += norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, norm, color="steelblue", alpha=0.25)
    ax.plot(angles, norm, color="steelblue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_title("Feature Profile", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


# ============================================================
# Grad-CAM Visualization with Explanatory Text
# ============================================================

def compute_and_plot_gradcam(model, signal_tensor, predicted_class, fs=DEFAULT_FS):
    """
    Run Grad-CAM on the last convolutional block and return a figure
    with three annotated panels plus textual interpretation.

    Returns (fig, cam_resized) or (None, None) on failure.
    """
    try:
        # Target the last Conv1d before global pooling (features[10])
        target_layer = model.features[10]
        gradcam      = GradCAM1D(model, target_layer)

        cam_raw = gradcam.generate_cam(signal_tensor, target_class=predicted_class)
        gradcam.cleanup()                    # FIX: alias now exists in GradCAM1D

        signal_np  = signal_tensor.squeeze().detach().numpy()
        cam_resize = zoom(cam_raw, len(signal_np) / len(cam_raw))
        cam_resize = np.clip(cam_resize, 0, 1)
        time_ms    = np.arange(len(signal_np)) / fs * 1000

        sig_range  = max(np.max(np.abs(signal_np)), 1e-6)

        fig, axes = plt.subplots(3, 1, figsize=(14, 11))
        fig.subplots_adjust(hspace=0.52)

        # --- Panel 1: Signal + overlay ---
        axes[0].plot(time_ms, signal_np, color="black", linewidth=0.6, alpha=0.55,
                     zorder=2)
        im = axes[0].imshow(
            cam_resize[np.newaxis, :],
            cmap="jet",
            aspect="auto",
            extent=[time_ms[0], time_ms[-1], -sig_range, sig_range],
            alpha=0.45,
            vmin=0,
            vmax=1,
            zorder=1,
        )
        plt.colorbar(im, ax=axes[0], label="Activation Intensity (0 = low, 1 = high)",
                     fraction=0.025)
        axes[0].set_xlim([time_ms[0], time_ms[-1]])
        axes[0].set_ylim([-sig_range * 1.1, sig_range * 1.1])
        axes[0].set_ylabel("Amplitude (normalized)")
        axes[0].set_title(
            "Panel 1 — Signal with Grad-CAM Heatmap Overlay",
            fontweight="bold", fontsize=11,
        )
        axes[0].grid(True, alpha=0.2)

        # --- Panel 2: Activation intensity curve ---
        axes[1].fill_between(time_ms, 0, cam_resize, alpha=0.25, color="crimson")
        axes[1].plot(time_ms, cam_resize, color="crimson", linewidth=1.5)
        axes[1].axhline(y=0.70, color="forestgreen", linestyle="--",
                        linewidth=1.5, label="High-importance threshold (0.70)")
        axes[1].set_ylim([0, 1.10])
        axes[1].set_ylabel("Activation Strength")
        axes[1].set_title(
            "Panel 2 — Temporal Activation Intensity Curve",
            fontweight="bold", fontsize=11,
        )
        axes[1].legend(fontsize=9, loc="upper right")
        axes[1].grid(True, alpha=0.3)

        high_pct = 100.0 * np.mean(cam_resize > 0.70)
        axes[1].text(
            0.01, 0.93,
            f"High-activation fraction: {high_pct:.1f}% of window",
            transform=axes[1].transAxes, fontsize=9,
            va="top", color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # --- Panel 3: Critical region highlights ---
        important = cam_resize > 0.70
        axes[2].plot(time_ms, signal_np, color="silver", linewidth=0.5,
                     alpha=0.6, label="Full signal")
        if important.any():
            axes[2].plot(
                time_ms[important], signal_np[important],
                color="crimson", linewidth=0, marker="o", markersize=2.5,
                alpha=0.85, label="High-importance samples (activation > 0.70)",
            )
        axes[2].set_xlabel("Time (ms)")
        axes[2].set_ylabel("Amplitude (normalized)")
        axes[2].set_title(
            "Panel 3 — Critical Temporal Regions Highlighted",
            fontweight="bold", fontsize=11,
        )
        axes[2].legend(fontsize=9, loc="upper right")
        axes[2].grid(True, alpha=0.2)

        return fig, cam_resize

    except Exception as e:
        st.warning(f"Grad-CAM computation failed: {e}")
        return None, None


def render_gradcam_interpretation(predicted_class, cam_resize, confidence):
    """
    Display structured textual interpretation panels beside the Grad-CAM plot.
    These panels explain what each visualization element means physically.
    """
    fault_name = CLASS_LABELS[predicted_class]
    fault_family = fault_name.split("_")[0] if predicted_class > 0 else "Normal"

    high_pct = 100.0 * np.mean(cam_resize > 0.70) if cam_resize is not None else 0.0
    activation_desc = (
        "sparse (less than 20% of the window)" if high_pct < 20
        else "moderate (20–50% of the window)" if high_pct < 50
        else "dense (more than 50% of the window)"
    )

    st.markdown("---")
    st.markdown("### How to Read This Visualization")

    # Panel explanations
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown(
            """
            **Panel 1 — Heatmap Overlay**

            The vibration waveform is drawn in black. The color overlay
            (blue to red scale) shows the Grad-CAM activation intensity
            at every time sample.

            - **Red / warm colors** indicate time regions that strongly
              influenced the model's classification decision.
            - **Blue / cool colors** indicate time regions that had little
              or no influence on the decision.

            In a healthy (Normal) bearing, you should see a predominantly
            blue overlay with no dominant red regions.
            In a faulty bearing, you should see periodic red spikes that
            align with the characteristic fault impulse frequency.
            """
        )

    with col_p2:
        st.markdown(
            f"""
            **Panel 2 — Activation Intensity**

            This plot strips away the signal and shows only the
            activation strength over time as a curve from 0 (unimportant)
            to 1 (most important).

            The green dashed line at **0.70** is the high-importance
            threshold. Peaks above this line are the moments in time
            the model treats as the primary evidence for its prediction.

            For this prediction:
            - Predicted class: **{fault_name}**
            - Confidence: **{confidence*100:.1f}%**
            - High-activation fraction: **{high_pct:.1f}%** of window
              ({activation_desc})

            A high fraction with a periodic pattern is consistent
            with a fault impulse train at a known bearing frequency.
            """
        )

    with col_p3:
        freq_text = ""
        if fault_family == "Inner":
            freq_text = (
                f"Expected impulse spacing: **1 / BPFI = {1000/BPFI:.2f} ms** "
                f"(BPFI = {BPFI:.1f} Hz)"
            )
        elif fault_family == "Outer":
            freq_text = (
                f"Expected impulse spacing: **1 / BPFO = {1000/BPFO:.2f} ms** "
                f"(BPFO = {BPFO:.1f} Hz)"
            )
        elif fault_family == "Ball":
            freq_text = (
                f"Expected impulse spacing: **1 / BSF = {1000/BSF:.2f} ms** "
                f"(BSF = {BSF:.1f} Hz)"
            )

        st.markdown(
            f"""
            **Panel 3 — Critical Sample Locations**

            Red dots mark every sample where the Grad-CAM activation
            exceeded the 0.70 threshold. These are the exact time
            instants the model treats as fault evidence.

            **Domain-physics check:**
            If the red dot clusters appear with regular spacing,
            they should align with the bearing's characteristic
            fault frequency period:

            {freq_text if freq_text else "For Normal signals, critical samples should be absent or randomly scattered."}

            **What to look for:**
            - Regular periodic clusters → consistent with a bearing fault
            - Random sparse dots → Normal or low-confidence prediction
            - Continuous dense clusters → signal anomaly; inspect sensor
            """
        )

    st.markdown("---")

    # Domain validation table
    st.markdown("### Fault Signature Reference Table")
    ref_df = pd.DataFrame({
        "Fault Type":       ["Normal",        "Inner Race",  "Outer Race",  "Ball / Rolling Element"],
        "Frequency":        ["None",           f"BPFI ({BPFI:.1f} Hz)", f"BPFO ({BPFO:.1f} Hz)", f"BSF ({BSF:.1f} Hz)"],
        "Impulse Period":   ["None",           f"{1000/BPFI:.2f} ms",   f"{1000/BPFO:.2f} ms",   f"{1000/BSF:.2f} ms"],
        "Expected Pattern": [
            "Low uniform activation, no peaks",
            "Periodic high-activation spikes at BPFI intervals",
            "Regularly spaced activation bursts at BPFO intervals",
            "Amplitude-modulated or irregularly spaced activations",
        ],
    })
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.info(
        "Grad-CAM provides a gradient-weighted attribution map. It does not guarantee "
        "that only fault-related signal content is highlighted — sensor noise or mechanical "
        "resonances may also contribute. Use the frequency spectrum (Spectrum tab) to "
        "cross-validate the presence of fault harmonic peaks before making maintenance decisions."
    )


# ============================================================
# Main Application
# ============================================================

def main():
    model, model_loaded, model_path = load_model()

    # ---------- Sidebar ----------
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")

    if model_loaded:
        st.sidebar.success(f"Model loaded")
        st.sidebar.caption(f"Path: {model_path}")
    else:
        st.sidebar.error("Model not found")
        st.sidebar.caption("Train a model first: python main.py")

    st.sidebar.markdown("---")

    input_mode = st.sidebar.radio(
        "Signal Source",
        ["Upload File", "Demo Signal", "Real-Time Simulation"],
        index=1,
    )

    fs = st.sidebar.number_input(
        "Sampling Frequency (Hz)",
        min_value=1000, max_value=100000, value=DEFAULT_FS, step=1000,
        help="The sampling rate of the uploaded vibration signal.",
    )

    uploaded_file = None
    demo_class    = 0

    if input_mode == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Vibration Signal",
            type=["csv", "npy", "txt", "mat"],
        )
    elif input_mode == "Demo Signal":
        demo_class = st.sidebar.selectbox(
            "Select Fault Type",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda x: f"{x}: {CLASS_LABELS[x]}",
        )
        st.sidebar.info(CLASS_DESCRIPTIONS[demo_class])
        if st.sidebar.button("Regenerate Signal"):
            st.rerun()
    elif input_mode == "Real-Time Simulation":
        sim_class = st.sidebar.selectbox(
            "Simulated Condition",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda x: CLASS_LABELS[x],
        )
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
        refresh_interval = st.sidebar.slider("Refresh interval (s)", 1, 10, 3)
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_spectrum = st.sidebar.checkbox("Show Spectrum Tab",     value=True)
    show_features = st.sidebar.checkbox("Show Features Tab",     value=True)
    show_gradcam  = st.sidebar.checkbox("Show Grad-CAM Tab",     value=True)
    mark_freqs    = st.sidebar.checkbox("Mark Fault Frequencies", value=True)

    with st.sidebar.expander("Bearing Specification"):
        st.markdown(
            f"""
            **CWRU 6205-2RS Deep Groove Ball Bearing**
            - Shaft speed: {SHAFT_RPM} RPM ({SHAFT_HZ:.1f} Hz)
            - Number of balls: {N_BALLS}
            - Ball diameter: {D_BALL} in
            - Pitch diameter: {D_PITCH} in

            **Characteristic fault frequencies:**
            - BPFI (inner race): {BPFI:.1f} Hz
            - BPFO (outer race): {BPFO:.1f} Hz
            - BSF (ball):        {BSF:.1f} Hz
            - FTF (cage):        {FTF:.1f} Hz
            """
        )

    # ---------- Main Content ----------
    st.title("Bearing Fault Detection System")
    st.markdown(
        "Real-time vibration analysis and fault classification using a 1D Convolutional "
        "Neural Network trained on the CWRU Bearing Dataset."
    )
    st.markdown("---")

    # ---------- Signal Acquisition ----------
    signal        = None
    signal_source = ""

    if input_mode == "Upload File" and uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.read()          # Read once; store bytes
            name      = uploaded_file.name
            if name.endswith((".csv", ".txt")):
                signal = np.loadtxt(io.BytesIO(raw_bytes), delimiter=",").flatten()
            elif name.endswith(".npy"):
                signal = np.load(io.BytesIO(raw_bytes)).flatten()
            elif name.endswith(".mat"):
                signal = load_mat_file(raw_bytes)     # FIX: consistent seek via BytesIO
            signal        = signal.astype(np.float32)
            signal_source = f"Uploaded: {name}"
            st.success(f"Loaded {name} — {len(signal):,} samples, {len(signal)/fs:.2f} s")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    elif input_mode == "Demo Signal":
        signal        = generate_demo_signal(demo_class, fs=fs)
        signal_source = f"Demo: {CLASS_LABELS[demo_class]}"
        st.info(f"Generated synthetic demo signal — class: **{CLASS_LABELS[demo_class]}**")

    elif input_mode == "Real-Time Simulation":
        signal        = generate_demo_signal(sim_class, fs=fs)
        signal_source = f"Simulation: {CLASS_LABELS[sim_class]}"
        st.info(f"Simulating condition: **{CLASS_LABELS[sim_class]}**")

    # ---------- Analysis Tabs ----------
    if signal is not None:
        tab_names = ["Signal Analysis", "Diagnosis"]
        if show_spectrum:
            tab_names.append("Spectrum")
        if show_gradcam:
            tab_names.append("Grad-CAM Explainability")
        if show_features:
            tab_names.append("Features")
        tab_names.append("History")

        tabs = st.tabs(tab_names)
        tab_map = {name: tab for name, tab in zip(tab_names, tabs)}

        # ====== Signal Analysis ======
        with tab_map["Signal Analysis"]:
            st.subheader("Time-Domain Signal")
            fig = plot_signal(signal, fs, title=f"Vibration Signal — {signal_source}")
            st.pyplot(fig)
            plt.close(fig)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Duration",  f"{len(signal)/fs:.3f} s")
            c2.metric("Samples",   f"{len(signal):,}")
            c3.metric("Peak",      f"{np.max(np.abs(signal)):.4f} g")
            c4.metric("RMS",       f"{np.sqrt(np.mean(signal**2)):.4f} g")
            c5.metric("Kurtosis",  f"{kurtosis(signal):.2f}")

        # ====== Diagnosis ======
        with tab_map["Diagnosis"]:
            st.subheader("Fault Diagnosis")
            if not model_loaded:
                st.error("Model not loaded. Run: python main.py")
            else:
                try:
                    sig_proc = preprocess_signal(signal, fs)
                    t0       = time.time()
                    with torch.no_grad():
                        tensor  = torch.FloatTensor(sig_proc).unsqueeze(0).unsqueeze(0)
                        logits  = model(tensor)
                        probs   = F.softmax(logits, dim=1).squeeze().numpy()
                    inf_ms = (time.time() - t0) * 1000

                    pred_cls   = int(np.argmax(probs))
                    confidence = float(probs[pred_cls])
                    alert_txt, alert_col, priority = get_alert_level(probs)

                    st.session_state.prediction_history.append({
                        "timestamp":       datetime.now(),
                        "predicted_class": pred_cls,
                        "confidence":      confidence,
                        "alert_level":     alert_txt,
                        "source":          signal_source,
                    })

                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                    with c1:
                        st.markdown("**Predicted Fault**")
                        st.markdown(
                            f"<h3 style='color:{alert_col};margin:0'>"
                            f"{CLASS_LABELS[pred_cls]}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.caption(CLASS_DESCRIPTIONS[pred_cls])
                    with c2:
                        st.markdown("**Confidence**")
                        st.markdown(f"<h3 style='margin:0'>{confidence*100:.1f}%</h3>",
                                    unsafe_allow_html=True)
                        st.progress(confidence)
                    with c3:
                        st.markdown("**Alert Level**")
                        st.markdown(
                            f"<h4 style='color:{alert_col};margin:0'>{alert_txt}</h4>",
                            unsafe_allow_html=True,
                        )
                    with c4:
                        st.markdown("**Inference**")
                        st.markdown(f"<h4 style='margin:0'>{inf_ms:.1f} ms</h4>",
                                    unsafe_allow_html=True)

                    st.markdown("---")
                    if   priority == 1: st.error   (f"CRITICAL FAULT — {CLASS_LABELS[pred_cls]}. Immediate maintenance required.")
                    elif priority == 2: st.warning ("DEGRADATION DETECTED. Schedule inspection.")
                    elif priority == 3: st.info    ("MONITORING REQUIRED. Increase inspection frequency.")
                    else:               st.success ("NORMAL OPERATION. No action required.")

                    st.markdown("---")
                    ca, cb = st.columns([2, 1])
                    with ca:
                        st.markdown("**Class Probability Distribution**")
                        fig = plot_probability_bars(probs)
                        st.pyplot(fig)
                        plt.close(fig)
                    with cb:
                        st.markdown("**Confidence Gauge**")
                        fig = plot_confidence_gauge(pred_cls, confidence)
                        st.pyplot(fig)
                        plt.close(fig)

                    st.markdown("**Top-3 Predictions**")
                    top3 = np.argsort(probs)[-3:][::-1]
                    st.dataframe(pd.DataFrame({
                        "Rank":        [1, 2, 3],
                        "Class":       [CLASS_LABELS[i] for i in top3],
                        "Probability": [f"{probs[i]*100:.2f}%" for i in top3],
                        "Level":       ["High" if probs[i] > 0.7 else "Medium" if probs[i] > 0.3 else "Low"
                                        for i in top3],
                    }), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Diagnosis error: {e}")
                    st.exception(e)

        # ====== Spectrum ======
        if show_spectrum and "Spectrum" in tab_map:
            with tab_map["Spectrum"]:
                st.subheader("Frequency Spectrum Analysis")
                st.markdown(
                    "The Power Spectral Density (PSD) shows the energy distribution "
                    "across frequencies. Bearing faults generate periodic impulses that "
                    "appear as spectral peaks at their characteristic frequencies. "
                    "Dashed vertical lines mark the theoretical fault frequencies for "
                    "the CWRU 6205-2RS bearing at 1797 RPM."
                )
                fig = plot_spectrum(signal, fs, mark_faults=mark_freqs)
                st.pyplot(fig)
                plt.close(fig)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("1x Shaft", f"{SHAFT_HZ:.1f} Hz")
                c2.metric("BPFI (Inner)", f"{BPFI:.1f} Hz")
                c3.metric("BPFO (Outer)", f"{BPFO:.1f} Hz")
                c4.metric("BSF (Ball)",   f"{BSF:.1f} Hz")

        # ====== Grad-CAM Explainability ======
        if show_gradcam and "Grad-CAM Explainability" in tab_map:
            with tab_map["Grad-CAM Explainability"]:
                st.subheader("Grad-CAM Explainability")

                st.markdown(
                    """
                    Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question:
                    *which parts of the vibration signal did the model focus on when making its prediction?*

                    It computes the gradient of the predicted class score with respect to the feature
                    maps of the final convolutional layer. Regions where large gradients coincide with
                    large activations are deemed important. The result is a heatmap over time that
                    highlights the temporal segments most responsible for the model's decision.

                    This is particularly valuable in bearing fault diagnosis because each fault type
                    produces impulses at a known characteristic frequency. If the model has learned
                    correctly, its Grad-CAM activations should be periodic and aligned with the
                    corresponding fault frequency interval.
                    """
                )
                st.markdown("---")

                if not model_loaded:
                    st.error("Model must be loaded to compute Grad-CAM. Run: python main.py")
                else:
                    try:
                        sig_proc = preprocess_signal(signal, fs)
                        tensor   = torch.FloatTensor(sig_proc).unsqueeze(0).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(tensor)
                            probs  = F.softmax(logits, dim=1).squeeze().numpy()
                        pred_cls   = int(np.argmax(probs))
                        confidence = float(probs[pred_cls])

                        target_class_opt = st.selectbox(
                            "Target class for Grad-CAM (default: predicted class)",
                            options=list(CLASS_LABELS.keys()),
                            index=pred_cls,
                            format_func=lambda x: f"{x}: {CLASS_LABELS[x]}",
                            help=(
                                "Grad-CAM can be computed for any class, not just the predicted one. "
                                "Changing the target class shows which signal regions are important "
                                "for distinguishing that particular fault type."
                            ),
                        )

                        target_tensor = torch.FloatTensor(sig_proc).unsqueeze(0).unsqueeze(0)
                        fig_cam, cam_resize = compute_and_plot_gradcam(
                            model, target_tensor, target_class_opt, fs=fs
                        )

                        if fig_cam is not None:
                            st.markdown(
                                f"**Grad-CAM computed for class: {CLASS_LABELS[target_class_opt]}** "
                                f"(model prediction: {CLASS_LABELS[pred_cls]}, "
                                f"confidence: {confidence*100:.1f}%)"
                            )
                            st.pyplot(fig_cam)
                            plt.close(fig_cam)

                            render_gradcam_interpretation(target_class_opt, cam_resize, confidence)

                    except Exception as e:
                        st.error(f"Grad-CAM error: {e}")
                        st.exception(e)

        # ====== Features ======
        if show_features and "Features" in tab_map:
            with tab_map["Features"]:
                st.subheader("Time-Domain Feature Extraction")
                st.markdown(
                    "The following statistical features are extracted from the preprocessed "
                    "vibration window. These hand-crafted features were the basis of classical "
                    "fault detection before deep learning. They remain useful as a sanity check "
                    "and for threshold-based alerting systems."
                )
                feats = compute_features(signal)
                ca, cb = st.columns([1, 1])
                with ca:
                    st.markdown("**Feature Values**")
                    st.dataframe(pd.DataFrame({
                        "Feature": list(feats.keys()),
                        "Value":   [f"{v:.6f}" for v in feats.values()],
                    }), use_container_width=True, hide_index=True)

                    st.markdown("**Health Indicator Thresholds**")
                    kurt   = feats["Kurtosis"]
                    crest  = feats["Crest Factor"]
                    if kurt > 10:
                        st.error   (f"Kurtosis = {kurt:.2f} (> 10: strong impulsive content, likely fault)")
                    elif kurt > 5:
                        st.warning (f"Kurtosis = {kurt:.2f} (> 5: elevated impulsiveness, monitor)")
                    else:
                        st.success (f"Kurtosis = {kurt:.2f} (< 5: normal Gaussian-like noise)")

                    if crest > 5:
                        st.error   (f"Crest Factor = {crest:.2f} (> 5: impulsive peaks detected)")
                    elif crest > 3:
                        st.warning (f"Crest Factor = {crest:.2f} (> 3: slightly elevated)")
                    else:
                        st.success (f"Crest Factor = {crest:.2f} (< 3: normal)")

                with cb:
                    st.markdown("**Feature Radar Chart**")
                    st.caption(
                        "Each axis is normalized to its typical maximum for a faulty bearing. "
                        "A larger filled area generally indicates more anomalous vibration content."
                    )
                    fig = plot_feature_radar(feats)
                    st.pyplot(fig)
                    plt.close(fig)

        # ====== History ======
        with tab_map["History"]:
            st.subheader("Prediction History")
            if st.session_state.prediction_history:
                hist = pd.DataFrame(list(st.session_state.prediction_history))
                hist["timestamp"]       = hist["timestamp"].dt.strftime("%H:%M:%S")
                hist["predicted_class"] = hist["predicted_class"].map(CLASS_LABELS)
                hist["confidence"]      = hist["confidence"].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(
                    hist[["timestamp", "predicted_class", "confidence", "alert_level", "source"]],
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button("Clear History"):
                    st.session_state.prediction_history.clear()
                    st.rerun()
            else:
                st.info("No predictions yet. Analyze a signal to begin recording history.")

    else:
        # Welcome screen
        st.markdown("### Select a signal source from the sidebar to begin analysis.")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **How It Works**

                1. Upload a vibration signal file (.csv, .npy, .mat) or select a demo signal
                2. The system applies bandpass filtering (10–5000 Hz) and Z-score normalization
                3. The 1D-CNN classifies the signal into one of 10 bearing health states
                4. Results are presented with confidence scores, alert levels, and Grad-CAM explanations
                """
            )
        with c2:
            st.markdown(
                """
                **Supported Fault Classes**

                - Normal operation
                - Inner race defects at 0.007, 0.014, 0.021 inch severity
                - Outer race defects at 0.007, 0.014, 0.021 inch severity
                - Ball/rolling element defects at 0.007, 0.014, 0.021 inch severity
                """
            )


if __name__ == "__main__":
    main()