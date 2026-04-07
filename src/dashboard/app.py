"""
Real-Time Bearing Health Monitoring Dashboard

Features:
- Upload vibration signals or use demo data
- Real-time fault classification
- Signal visualization (time domain)
- Frequency spectrum analysis
- Feature extraction and display
- Grad-CAM visualization
- Historical trend tracking
"""

# ============================================================
# PATH FIX - Must be at the very top
# ============================================================
from collections import deque
from datetime import datetime
import time
import io
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import streamlit as st
import sys
from pathlib import Path

# Get project root (3 levels up: dashboard -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Imports
# ============================================================

# Import project modules
try:
    from src.models.vibration_cnn import VibrationCNN
    from src.data.preprocessing import bandpass_filter, normalize_signal, create_windows
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure to run: pip install -e . from project root")
    st.stop()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Bearing Health Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '# Bearing Fault Detection System\nUsing 1D-CNN with Time-Based Split'
    }
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
    9: "Ball_021"
}

CLASS_DESCRIPTIONS = {
    0: "Bearing operating normally with no defects",
    1: "Inner race fault - 0.007 inch diameter (early stage)",
    2: "Inner race fault - 0.014 inch diameter (moderate)",
    3: "Inner race fault - 0.021 inch diameter (severe)",
    4: "Outer race fault - 0.007 inch diameter (early stage)",
    5: "Outer race fault - 0.014 inch diameter (moderate)",
    6: "Outer race fault - 0.021 inch diameter (severe)",
    7: "Ball/Rolling element fault - 0.007 inch (early stage)",
    8: "Ball/Rolling element fault - 0.014 inch (moderate)",
    9: "Ball/Rolling element fault - 0.021 inch (severe)"
}

FAULT_COLORS = {
    0: "#2ecc71",  # Green - Normal
    1: "#e74c3c",  # Red - Inner race
    2: "#e74c3c",
    3: "#e74c3c",
    4: "#3498db",  # Blue - Outer race
    5: "#3498db",
    6: "#3498db",
    7: "#f39c12",  # Orange - Ball
    8: "#f39c12",
    9: "#f39c12"
}

# Bearing parameters (CWRU 6205-2RS)
SHAFT_RPM = 1797
SHAFT_SPEED_HZ = SHAFT_RPM / 60
N_BALLS = 9
D_BALL = 0.312  # inches
D_PITCH = 1.537  # inches
CONTACT_ANGLE = 0  # degrees

# Theoretical fault frequencies
BPFI = (SHAFT_SPEED_HZ / 2) * N_BALLS * (1 + (D_BALL / D_PITCH))  # ~162 Hz
BPFO = (SHAFT_SPEED_HZ / 2) * N_BALLS * (1 - (D_BALL / D_PITCH))  # ~107 Hz
BSF = (SHAFT_SPEED_HZ / 2) * (D_PITCH / D_BALL) * \
    (1 - (D_BALL / D_PITCH) ** 2)  # ~140 Hz
FTF = (SHAFT_SPEED_HZ / 2) * (1 - (D_BALL / D_PITCH))  # ~12 Hz

DEFAULT_FS = 12000  # Sampling frequency
WINDOW_SIZE = 2048

# ============================================================
# Session State Initialization
# ============================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=100)

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ============================================================
# Model Loading (Cached)
# ============================================================


@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    model = VibrationCNN(num_classes=10)

    # Try multiple model paths
    model_paths = [
        PROJECT_ROOT / 'models' / 'best_model.pth',
        PROJECT_ROOT / 'models' / 'final_model.pth',
        PROJECT_ROOT / 'best_model.pth',
    ]

    for model_path in model_paths:
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu',
                                        weights_only=False)

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                return model, True, str(model_path)
            except Exception as e:
                continue

    return model, False, None


# ============================================================
# Helper Functions
# ============================================================
def get_alert_level(probs):
    """
    Determine maintenance alert level based on prediction probabilities.

    Returns:
        alert_text: Description of alert
        alert_emoji: Emoji indicator
        alert_color: Color code
        priority: Numerical priority (1=highest)
    """
    normal_prob = probs[0]
    max_fault_prob = np.max(probs[1:])
    max_fault_class = np.argmax(probs[1:]) + 1

    if normal_prob > 0.9:
        return "Normal Operation", "✅", "#2ecc71", 4
    elif normal_prob > 0.7:
        return "Monitor Closely", "👁️", "#3498db", 3
    elif max_fault_prob > 0.7:
        fault_type = CLASS_LABELS[max_fault_class].split('_')[0]
        return f"Critical: {fault_type} Fault", "🚨", "#e74c3c", 1
    elif max_fault_prob > 0.4:
        return "Degradation Detected", "⚠️", "#f39c12", 2
    else:
        return "Uncertain - Investigate", "🔍", "#9b59b6", 3


def preprocess_signal(signal, fs=DEFAULT_FS):
    """Preprocess raw signal for model input."""
    # Ensure 1D
    signal = np.asarray(signal).flatten().astype(np.float32)

    # Check minimum length
    if len(signal) < WINDOW_SIZE:
        raise ValueError(
            f"Signal too short: {len(signal)} samples. Need at least {WINDOW_SIZE}.")

    # Take first window
    signal = signal[:WINDOW_SIZE]

    # Apply bandpass filter
    try:
        signal = bandpass_filter(signal, lowcut=10, highcut=5000, fs=fs)
    except Exception as e:
        st.warning(f"Filter warning: {e}. Using raw signal.")

    # Normalize
    signal = normalize_signal(signal, method='zscore')

    return signal


def compute_features(signal):
    """Compute time-domain features from signal."""
    features = {
        'RMS': np.sqrt(np.mean(signal ** 2)),
        'Peak': np.max(np.abs(signal)),
        'Peak-to-Peak': np.max(signal) - np.min(signal),
        'Crest Factor': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-8),
        'Kurtosis': kurtosis(signal),
        'Skewness': skew(signal),
        'Mean': np.mean(signal),
        'Std Dev': np.std(signal),
        'Variance': np.var(signal)
    }
    return features


def generate_demo_signal(class_id, fs=DEFAULT_FS, duration=0.5, noise_level=0.1):
    """Generate synthetic vibration signal for demo."""
    np.random.seed(int(time.time()) % 1000 + class_id)

    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples)

    # Base noise
    signal = np.random.normal(0, noise_level, num_samples)

    # Add shaft frequency component
    signal += 0.2 * np.sin(2 * np.pi * SHAFT_SPEED_HZ * t)

    if class_id == 0:  # Normal
        # Just noise + shaft frequency
        pass

    elif class_id in [1, 2, 3]:  # Inner race faults
        severity = [0.4, 0.7, 1.0][class_id - 1]
        # Add BPFI harmonics
        signal += severity * 0.5 * np.sin(2 * np.pi * BPFI * t)
        signal += severity * 0.3 * np.sin(2 * np.pi * 2 * BPFI * t)
        # Add impulses
        impulse_period = int(fs / BPFI)
        for i in range(0, num_samples, impulse_period):
            if i < num_samples:
                signal[i] += severity * 1.5 * np.random.randn()

    elif class_id in [4, 5, 6]:  # Outer race faults
        severity = [0.4, 0.7, 1.0][class_id - 4]
        # Add BPFO harmonics
        signal += severity * 0.5 * np.sin(2 * np.pi * BPFO * t)
        signal += severity * 0.3 * np.sin(2 * np.pi * 2 * BPFO * t)
        # Add impulses
        impulse_period = int(fs / BPFO)
        for i in range(0, num_samples, impulse_period):
            if i < num_samples:
                signal[i] += severity * 1.5 * np.random.randn()

    elif class_id in [7, 8, 9]:  # Ball faults
        severity = [0.4, 0.7, 1.0][class_id - 7]
        # Add BSF harmonics
        signal += severity * 0.5 * np.sin(2 * np.pi * BSF * t)
        signal += severity * 0.3 * np.sin(2 * np.pi * 2 * BSF * t)
        # Add impulses
        impulse_period = int(fs / BSF)
        for i in range(0, num_samples, impulse_period):
            if i < num_samples:
                signal[i] += severity * 1.5 * np.random.randn()

    return signal.astype(np.float32)


def load_mat_file(uploaded_file):
    """Load signal from .mat file."""
    mat_data = loadmat(io.BytesIO(uploaded_file.read()))

    # Find drive-end signal
    for key in mat_data.keys():
        if 'DE_time' in key:
            return mat_data[key].flatten().astype(np.float32)

    # Fallback: find first array
    for key, value in mat_data.items():
        if not key.startswith('_') and isinstance(value, np.ndarray):
            return value.flatten().astype(np.float32)

    raise ValueError("Could not find signal data in .mat file")


# ============================================================
# Visualization Functions
# ============================================================
def plot_signal(signal, fs, title="Vibration Signal"):
    """Plot time-domain signal."""
    time_axis = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, signal, linewidth=0.5, color='steelblue')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (g)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_axis[-1]])

    plt.tight_layout()
    return fig


def plot_spectrum(signal, fs, mark_faults=True):
    """Plot frequency spectrum with fault frequency markers."""
    # Compute PSD
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full spectrum
    axes[0].semilogy(freqs, psd, color='steelblue', linewidth=1)
    axes[0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[0].set_ylabel('PSD (g²/Hz)', fontsize=11)
    axes[0].set_title('Power Spectral Density', fontsize=12, fontweight='bold')
    axes[0].set_xlim([0, fs / 2])
    axes[0].grid(True, alpha=0.3)

    if mark_faults:
        axes[0].axvline(x=BPFI, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                        label=f'BPFI ({BPFI:.1f} Hz)')
        axes[0].axvline(x=BPFO, color='blue', linestyle='--', alpha=0.7, linewidth=1.5,
                        label=f'BPFO ({BPFO:.1f} Hz)')
        axes[0].axvline(x=BSF, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                        label=f'BSF ({BSF:.1f} Hz)')
        axes[0].axvline(x=SHAFT_SPEED_HZ, color='orange', linestyle='--', alpha=0.7, linewidth=1.5,
                        label=f'1× Shaft ({SHAFT_SPEED_HZ:.1f} Hz)')
        axes[0].legend(loc='upper right', fontsize=9)

    # Zoomed spectrum (0 - 500 Hz)
    mask = freqs <= 500
    axes[1].semilogy(freqs[mask], psd[mask], color='steelblue', linewidth=1)
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('PSD (g²/Hz)', fontsize=11)
    axes[1].set_title('Zoomed Spectrum (0-500 Hz)',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    if mark_faults:
        for freq, color, name in [(BPFI, 'red', 'BPFI'), (BPFO, 'blue', 'BPFO'),
                                  (BSF, 'green', 'BSF'), (SHAFT_SPEED_HZ, 'orange', '1×')]:
            if freq <= 500:
                axes[1].axvline(x=freq, color=color,
                                linestyle='--', alpha=0.7, linewidth=1.5)
                axes[1].text(freq + 5, axes[1].get_ylim()[1] * 0.5, name,
                             color=color, fontsize=9, rotation=90, va='center')

    plt.tight_layout()
    return fig


def plot_probability_bars(probs):
    """Plot horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(CLASS_LABELS.values())
    colors = [FAULT_COLORS[i] for i in range(10)]

    # Highlight predicted class
    predicted = np.argmax(probs)
    alphas = [1.0 if i == predicted else 0.5 for i in range(10)]

    bars = ax.barh(classes, probs, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    bars[predicted].set_alpha(1.0)
    bars[predicted].set_edgecolor('black')
    bars[predicted].set_linewidth(2)

    ax.set_xlabel('Probability', fontsize=11)
    ax.set_xlim([0, 1])
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, axis='x', alpha=0.3)

    # Add value labels
    for i, (prob, bar) in enumerate(zip(probs, bars)):
        if prob > 0.02:
            ax.text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_confusion_style(predicted_class, confidence):
    """Create a visual indicator of prediction confidence."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create pie chart
    sizes = [confidence, 1 - confidence]
    colors = [FAULT_COLORS[predicted_class], '#ecf0f1']

    wedges, texts = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.3, edgecolor='white'))

    # Add center text
    ax.text(0, 0, f'{confidence*100:.1f}%', ha='center', va='center',
            fontsize=28, fontweight='bold', color=FAULT_COLORS[predicted_class])

    ax.text(0, -0.15, CLASS_LABELS[predicted_class], ha='center', va='top',
            fontsize=12, fontweight='bold')

    ax.set_title('Confidence', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def plot_feature_radar(features):
    """Create radar chart of features."""
    # Select key features
    feature_names = ['RMS', 'Peak', 'Crest Factor', 'Kurtosis', 'Skewness']
    values = [features[name] for name in feature_names]

    # Normalize values for visualization
    max_vals = [1, 2, 10, 20, 5]  # Typical max values
    normalized = [min(v / m, 1) for v, m in zip(values, max_vals)]

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(feature_names),
                         endpoint=False).tolist()
    normalized += normalized[:1]  # Close the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, normalized, color='steelblue', alpha=0.25)
    ax.plot(angles, normalized, color='steelblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_title('Feature Profile', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


# ============================================================
# Main Application
# ============================================================
def main():
    # Load model
    model, model_loaded, model_path = load_model()
    st.session_state.model_loaded = model_loaded

    # ========== Sidebar ==========
    st.sidebar.title("🔧 Configuration")
    st.sidebar.markdown("---")

    # Model status
    if model_loaded:
        st.sidebar.success(f"✅ Model loaded")
        st.sidebar.caption(f"Path: {model_path}")
    else:
        st.sidebar.error("❌ Model not found")
        st.sidebar.caption("Train a model first using: python main.py")

    st.sidebar.markdown("---")

    # Input source selection
    input_mode = st.sidebar.radio(
        "📥 Signal Source",
        ["Upload File", "Demo Signal", "Real-Time Simulation"],
        index=1
    )

    # Sampling frequency
    fs = st.sidebar.number_input(
        "🔊 Sampling Frequency (Hz)",
        min_value=1000,
        max_value=100000,
        value=DEFAULT_FS,
        step=1000,
        help="Sampling frequency of the vibration signal"
    )

    st.sidebar.markdown("---")

    # Input-specific settings
    uploaded_file = None
    demo_class = 0

    if input_mode == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Vibration Signal",
            type=['csv', 'npy', 'txt', 'mat'],
            help="Upload a vibration signal file"
        )

    elif input_mode == "Demo Signal":
        demo_class = st.sidebar.selectbox(
            "🎯 Select Fault Type",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda x: f"{x}: {CLASS_LABELS[x]}",
            index=0
        )

        st.sidebar.info(CLASS_DESCRIPTIONS[demo_class])

        if st.sidebar.button("🔄 Regenerate Signal"):
            st.rerun()

    elif input_mode == "Real-Time Simulation":
        sim_class = st.sidebar.selectbox(
            "🎯 Simulated Condition",
            options=list(CLASS_LABELS.keys()),
            format_func=lambda x: CLASS_LABELS[x],
            index=0
        )

        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
        refresh_interval = st.sidebar.slider("Refresh interval (s)", 1, 10, 3)

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    st.sidebar.markdown("---")

    # Display settings
    st.sidebar.subheader("📊 Display Options")
    show_spectrum = st.sidebar.checkbox("Show Spectrum", value=True)
    show_features = st.sidebar.checkbox("Show Features", value=True)
    mark_fault_freqs = st.sidebar.checkbox(
        "Mark Fault Frequencies", value=True)

    st.sidebar.markdown("---")

    # Bearing info
    with st.sidebar.expander("ℹ️ Bearing Information"):
        st.markdown(f"""
        **CWRU 6205-2RS Bearing**
        - Shaft Speed: {SHAFT_RPM} RPM ({SHAFT_SPEED_HZ:.1f} Hz)
        - Number of Balls: {N_BALLS}
        - Ball Diameter: {D_BALL} in
        - Pitch Diameter: {D_PITCH} in
        
        **Fault Frequencies:**
        - BPFI (Inner): {BPFI:.1f} Hz
        - BPFO (Outer): {BPFO:.1f} Hz
        - BSF (Ball): {BSF:.1f} Hz
        - FTF (Cage): {FTF:.1f} Hz
        """)

    # ========== Main Content ==========
    st.title("🔧 Bearing Fault Detection System")
    st.markdown(
        "Real-time vibration analysis and fault classification using deep learning")
    st.markdown("---")

    # Get signal based on input mode
    signal = None
    signal_source = ""

    if input_mode == "Upload File" and uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                signal = np.loadtxt(io.BytesIO(
                    uploaded_file.read()), delimiter=',').flatten()
            elif uploaded_file.name.endswith('.npy'):
                signal = np.load(io.BytesIO(uploaded_file.read())).flatten()
            elif uploaded_file.name.endswith('.mat'):
                uploaded_file.seek(0)
                signal = load_mat_file(uploaded_file)

            signal = signal.astype(np.float32)
            signal_source = f"Uploaded: {uploaded_file.name}"
            st.success(
                f"✅ Loaded {uploaded_file.name} ({len(signal):,} samples, {len(signal)/fs:.2f}s)")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    elif input_mode == "Demo Signal":
        signal = generate_demo_signal(demo_class, fs=fs)
        signal_source = f"Demo: {CLASS_LABELS[demo_class]}"
        st.info(f"📊 Generated demo signal: **{CLASS_LABELS[demo_class]}**")

    elif input_mode == "Real-Time Simulation":
        signal = generate_demo_signal(sim_class, fs=fs)
        signal_source = f"Simulation: {CLASS_LABELS[sim_class]}"
        st.info(f"🔄 Simulating: **{CLASS_LABELS[sim_class]}**")

    # ========== Analysis ==========
    if signal is not None:
        # Create tabs
        tabs = st.tabs(["📊 Signal Analysis", "🔍 Diagnosis",
                       "📈 Spectrum", "📋 Features", "📜 History"])

        # ========== Tab 1: Signal Analysis ==========
        with tabs[0]:
            st.subheader("Time-Domain Signal")

            # Signal plot
            fig_signal = plot_signal(
                signal, fs, title=f"Vibration Signal ({signal_source})")
            st.pyplot(fig_signal)
            plt.close()

            # Quick stats
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Duration", f"{len(signal)/fs:.3f} s")
            col2.metric("Samples", f"{len(signal):,}")
            col3.metric("Peak", f"{np.max(np.abs(signal)):.4f} g")
            col4.metric("RMS", f"{np.sqrt(np.mean(signal**2)):.4f} g")
            col5.metric("Kurtosis", f"{kurtosis(signal):.2f}")

        # ========== Tab 2: Diagnosis ==========
        with tabs[1]:
            st.subheader("Fault Diagnosis")

            if not model_loaded:
                st.error("❌ Model not loaded. Cannot perform diagnosis.")
                st.info("Train a model using: `python main.py`")
            else:
                try:
                    # Preprocess
                    signal_processed = preprocess_signal(signal, fs)

                    # Inference
                    start_time = time.time()
                    with torch.no_grad():
                        signal_tensor = torch.FloatTensor(
                            signal_processed).unsqueeze(0).unsqueeze(0)
                        logits = model(signal_tensor)
                        probs = F.softmax(logits, dim=1).squeeze().numpy()
                    inference_time = (time.time() - start_time) * 1000

                    predicted_class = int(np.argmax(probs))
                    confidence = float(probs[predicted_class])
                    alert_text, alert_emoji, alert_color, priority = get_alert_level(
                        probs)

                    # Store in history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'alert_level': alert_text,
                        'source': signal_source
                    })

                    # Display results
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                    with col1:
                        st.markdown("### Predicted Fault")
                        st.markdown(
                            f"<h2 style='color:{alert_color}; margin:0;'>{CLASS_LABELS[predicted_class]}</h2>",
                            unsafe_allow_html=True
                        )
                        st.caption(CLASS_DESCRIPTIONS[predicted_class])

                    with col2:
                        st.markdown("### Confidence")
                        st.markdown(
                            f"<h2 style='margin:0;'>{confidence*100:.1f}%</h2>", unsafe_allow_html=True)

                        # Confidence bar
                        st.progress(confidence)

                    with col3:
                        st.markdown("### Alert Level")
                        st.markdown(
                            f"<h3 style='color:{alert_color}; margin:0;'>{alert_emoji} {alert_text}</h3>",
                            unsafe_allow_html=True
                        )

                    with col4:
                        st.markdown("### Time")
                        st.markdown(
                            f"<h3 style='margin:0;'>{inference_time:.1f} ms</h3>", unsafe_allow_html=True)

                    st.markdown("---")

                    # Alert banner
                    if priority == 1:
                        st.error(
                            f"🚨 **CRITICAL FAULT DETECTED** - {CLASS_LABELS[predicted_class]} - Immediate maintenance required!")
                    elif priority == 2:
                        st.warning(
                            f"⚠️ **DEGRADATION DETECTED** - Schedule maintenance inspection soon.")
                    elif priority == 3:
                        st.info(
                            f"👁️ **MONITORING REQUIRED** - Continue to monitor bearing condition.")
                    else:
                        st.success(
                            f"✅ **NORMAL OPERATION** - Bearing operating within normal parameters.")

                    st.markdown("---")

                    # Probability distribution
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("#### Class Probabilities")
                        fig_probs = plot_probability_bars(probs)
                        st.pyplot(fig_probs)
                        plt.close()

                    with col2:
                        st.markdown("#### Confidence Gauge")
                        fig_conf = plot_confusion_style(
                            predicted_class, confidence)
                        st.pyplot(fig_conf)
                        plt.close()

                    # Top 3 predictions table
                    st.markdown("#### Top 3 Predictions")
                    top3_idx = np.argsort(probs)[-3:][::-1]

                    top3_df = pd.DataFrame({
                        'Rank': [1, 2, 3],
                        'Class': [CLASS_LABELS[i] for i in top3_idx],
                        'Probability': [f"{probs[i]*100:.2f}%" for i in top3_idx],
                        'Confidence': ['High' if probs[i] > 0.7 else 'Medium' if probs[i] > 0.3 else 'Low'
                                       for i in top3_idx]
                    })

                    st.dataframe(
                        top3_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"❌ Error during diagnosis: {e}")
                    st.exception(e)

        # ========== Tab 3: Spectrum ==========
        with tabs[2]:
            if show_spectrum:
                st.subheader("Frequency Spectrum Analysis")

                fig_spectrum = plot_spectrum(
                    signal, fs, mark_faults=mark_fault_freqs)
                st.pyplot(fig_spectrum)
                plt.close()

                # Fault frequency info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("1× Shaft", f"{SHAFT_SPEED_HZ:.1f} Hz")
                col2.metric("BPFI (Inner)", f"{BPFI:.1f} Hz")
                col3.metric("BPFO (Outer)", f"{BPFO:.1f} Hz")
                col4.metric("BSF (Ball)", f"{BSF:.1f} Hz")
            else:
                st.info("Spectrum analysis disabled. Enable in sidebar.")

        # ========== Tab 4: Features ==========
        with tabs[3]:
            if show_features:
                st.subheader("Time-Domain Features")

                features = compute_features(signal)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("#### Feature Values")
                    features_df = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Value': [f"{v:.6f}" for v in features.values()]
                    })
                    st.dataframe(
                        features_df, use_container_width=True, hide_index=True)

                    # Health indicators
                    st.markdown("#### Health Indicators")

                    kurt = features['Kurtosis']
                    crest = features['Crest Factor']

                    if kurt > 10:
                        st.error(
                            f"🔴 High Kurtosis: {kurt:.2f} (>10 indicates fault)")
                    elif kurt > 5:
                        st.warning(f"🟡 Elevated Kurtosis: {kurt:.2f}")
                    else:
                        st.success(f"🟢 Normal Kurtosis: {kurt:.2f}")

                    if crest > 5:
                        st.error(
                            f"🔴 High Crest Factor: {crest:.2f} (impulsive)")
                    elif crest > 3:
                        st.warning(f"🟡 Elevated Crest Factor: {crest:.2f}")
                    else:
                        st.success(f"🟢 Normal Crest Factor: {crest:.2f}")

                with col2:
                    st.markdown("#### Feature Profile")
                    fig_radar = plot_feature_radar(features)
                    st.pyplot(fig_radar)
                    plt.close()
            else:
                st.info("Feature analysis disabled. Enable in sidebar.")

        # ========== Tab 5: History ==========
        with tabs[4]:
            st.subheader("Prediction History")

            if len(st.session_state.prediction_history) > 0:
                history_df = pd.DataFrame(
                    list(st.session_state.prediction_history))
                history_df['timestamp'] = history_df['timestamp'].dt.strftime(
                    '%H:%M:%S')
                history_df['predicted_class'] = history_df['predicted_class'].map(
                    CLASS_LABELS)
                history_df['confidence'] = history_df['confidence'].apply(
                    lambda x: f"{x*100:.1f}%")

                st.dataframe(
                    history_df[['timestamp', 'predicted_class',
                                'confidence', 'alert_level', 'source']],
                    use_container_width=True,
                    hide_index=True
                )

                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history.clear()
                    st.rerun()
            else:
                st.info(
                    "No predictions yet. Analyze a signal to start building history.")

    else:
        # No signal loaded - show welcome message
        st.markdown("## 👈 Get Started")
        st.markdown(
            "Select a signal source from the sidebar to begin analysis.")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 How It Works")
            st.markdown("""
            1. **Upload** a vibration signal file (CSV, NPY, MAT)
            2. **Or select** a demo signal to test the system
            3. **Automatic preprocessing**: Filtering, normalization
            4. **Deep learning inference**: 1D-CNN fault classification
            5. **Comprehensive analysis**: Time-domain, frequency, features
            """)

        with col2:
            st.markdown("### 🎯 Supported Fault Types")
            st.markdown("""
            - ✅ **Normal** operation
            - 🔴 **Inner race** defects (0.007", 0.014", 0.021")
            - 🔵 **Outer race** defects (0.007", 0.014", 0.021")
            - 🟡 **Ball** defects (0.007", 0.014", 0.021")
            """)

        st.markdown("---")

        # Quick demo button
        if st.button("🚀 Quick Demo - Normal Signal", type="primary"):
            st.session_state['demo_mode'] = True
            st.rerun()


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()
