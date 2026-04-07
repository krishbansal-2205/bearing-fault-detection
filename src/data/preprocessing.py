"""
Signal preprocessing with time-based splitting support.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple, List, Dict
from collections import defaultdict, Counter
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 10.0,
    highcut: float = 5000.0,
    fs: int = 12000,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        signal: Input signal
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)

    b, a = sp_signal.butter(order, [low, high], btype='band')
    filtered = sp_signal.filtfilt(b, a, signal)

    return filtered


def normalize_signal(
    window: np.ndarray,
    method: str = 'zscore',
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize signal window.

    Args:
        window: Signal window
        method: Normalization method ('zscore', 'minmax', 'maxabs')
        epsilon: Small constant to prevent division by zero

    Returns:
        Normalized window
    """
    if method == 'zscore':
        mean = np.mean(window)
        std = np.std(window)
        if std < epsilon:
            return window - mean
        return (window - mean) / std

    elif method == 'minmax':
        min_val = np.min(window)
        max_val = np.max(window)
        return (window - min_val) / (max_val - min_val + epsilon)

    elif method == 'maxabs':
        max_abs = np.max(np.abs(window))
        return window / (max_abs + epsilon)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_windows(
    signal: np.ndarray,
    window_size: int = 2048,
    overlap: float = 0.5
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create overlapping windows from signal.

    Args:
        signal: Input signal
        window_size: Window size in samples
        overlap: Overlap fraction (0.5 = 50%)

    Returns:
        windows: List of windows
        indices: Starting indices of each window
    """
    step = int(window_size * (1 - overlap))
    windows = []
    indices = []

    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
        indices.append(i)

    return windows, indices


def time_based_split(
    signals_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    train_time_ratio: float = 0.7,
    window_size: int = 2048,
    overlap: float = 0.5,
    fs: int = 12000,
    min_signal_length: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each signal temporally into train and test.

    Strategy:
    - Use FIRST train_time_ratio% of each signal for training
    - Use LAST (1-train_time_ratio)% for testing

    This simulates:
    - Training on early operation data
    - Testing on later/degraded operation data

    Args:
        signals_dict: Dictionary of signals
        labels_dict: Dictionary of labels
        train_time_ratio: Fraction for training (0.7 = first 70%)
        window_size: Window size
        overlap: Window overlap
        fs: Sampling frequency
        min_signal_length: Minimum signal length to include

    Returns:
        X_train, y_train, X_test, y_test
    """
    logger.info("=" * 60)
    logger.info("TIME-BASED SPLIT")
    logger.info("=" * 60)
    logger.info(f"Strategy: First {train_time_ratio*100:.0f}% → Train, "
                f"Last {(1-train_time_ratio)*100:.0f}% → Test")

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    files_used = 0
    files_skipped = 0

    for filename, signal in tqdm(signals_dict.items(), desc="Processing files"):
        label = labels_dict[filename]

        # Skip very short signals
        if len(signal) < min_signal_length:
            files_skipped += 1
            continue

        files_used += 1

        # Preprocess
        signal_filtered = bandpass_filter(signal, fs=fs)

        # Calculate temporal split point
        split_idx = int(len(signal_filtered) * train_time_ratio)

        # Split signal
        train_signal = signal_filtered[:split_idx]
        test_signal = signal_filtered[split_idx:]

        # Create windows from train portion
        train_wins, _ = create_windows(train_signal, window_size, overlap)
        for window in train_wins:
            train_windows.append(normalize_signal(window))
            train_labels.append(label)

        # Create windows from test portion
        test_wins, _ = create_windows(test_signal, window_size, overlap)
        for window in test_wins:
            test_windows.append(normalize_signal(window))
            test_labels.append(label)

    X_train = np.array(train_windows, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)
    X_test = np.array(test_windows, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int64)

    logger.info(f"Files used: {files_used}, skipped: {files_skipped}")
    logger.info(f"Train windows: {len(X_train)}")
    logger.info(f"Test windows: {len(X_test)}")
    logger.info(f"Train distribution: {dict(Counter(y_train))}")
    logger.info(f"Test distribution: {dict(Counter(y_test))}")

    return X_train, y_train, X_test, y_test


def hybrid_split(
    signals_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    file_train_ratio: float = 0.7,
    time_train_ratio: float = 0.7,
    window_size: int = 2048,
    overlap: float = 0.5,
    fs: int = 12000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hybrid approach: Split files AND use time-based split.

    Strategy:
    1. Split files into train/test sets
    2. For train files: Use early portion (first 70%)
    3. For test files: Use later portion (last 30%)

    Most realistic for production deployment.

    Args:
        signals_dict: Dictionary of signals
        labels_dict: Dictionary of labels
        file_train_ratio: Fraction of files for training
        time_train_ratio: Fraction of signal to use temporally
        window_size: Window size
        overlap: Window overlap
        fs: Sampling frequency
        seed: Random seed

    Returns:
        X_train, y_train, X_test, y_test
    """
    logger.info("=" * 60)
    logger.info("HYBRID SPLIT (File-based + Time-based)")
    logger.info("=" * 60)

    # Step 1: Split files by class (stratified)
    files_by_class = defaultdict(list)
    for filename, label in labels_dict.items():
        files_by_class[label].append(filename)

    train_files = []
    test_files = []

    np.random.seed(seed)
    for class_id, files in files_by_class.items():
        files = files.copy()
        np.random.shuffle(files)

        n_train = max(1, int(len(files) * file_train_ratio))
        train_files.extend(files[:n_train])
        test_files.extend(files[n_train:])

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Test files: {len(test_files)}")

    # Step 2: Process files with time-based windowing
    def process_files(file_list: List[str], use_early_portion: bool = True):
        """Process files with temporal selection."""
        windows = []
        labels = []

        for filename in file_list:
            signal = signals_dict[filename]
            label = labels_dict[filename]

            # Preprocess
            signal_filtered = bandpass_filter(signal, fs=fs)

            # Select temporal portion
            if use_early_portion:
                # Use first 70% (early)
                end_idx = int(len(signal_filtered) * time_train_ratio)
                signal_portion = signal_filtered[:end_idx]
            else:
                # Use last 30% (late)
                start_idx = int(len(signal_filtered) * time_train_ratio)
                signal_portion = signal_filtered[start_idx:]

            # Create windows
            wins, _ = create_windows(signal_portion, window_size, overlap)

            for window in wins:
                windows.append(normalize_signal(window))
                labels.append(label)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)

    # Train: Early portions of train files
    X_train, y_train = process_files(train_files, use_early_portion=True)

    # Test: Later portions of test files
    X_test, y_test = process_files(test_files, use_early_portion=False)

    logger.info(f"Train windows (early data): {len(X_train)}")
    logger.info(f"Test windows (late data): {len(X_test)}")
    logger.info(f"Train distribution: {dict(Counter(y_train))}")
    logger.info(f"Test distribution: {dict(Counter(y_test))}")

    return X_train, y_train, X_test, y_test


def file_based_split(
    signals_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    train_ratio: float = 0.75,
    window_size: int = 2048,
    overlap: float = 0.5,
    fs: int = 12000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Traditional file-based split (baseline for comparison).

    Args:
        signals_dict: Dictionary of signals
        labels_dict: Dictionary of labels
        train_ratio: Fraction of files for training
        window_size: Window size
        overlap: Window overlap
        fs: Sampling frequency
        seed: Random seed

    Returns:
        X_train, y_train, X_test, y_test
    """
    logger.info("=" * 60)
    logger.info("FILE-BASED SPLIT (Baseline)")
    logger.info("=" * 60)

    # Split files by class
    files_by_class = defaultdict(list)
    for filename, label in labels_dict.items():
        files_by_class[label].append(filename)

    train_files = []
    test_files = []

    np.random.seed(seed)
    for class_id, files in files_by_class.items():
        files = files.copy()
        np.random.shuffle(files)

        n_train = max(1, int(len(files) * train_ratio))
        train_files.extend(files[:n_train])
        test_files.extend(files[n_train:])

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Test files: {len(test_files)}")

    # Process files (use entire signal)
    def process_files(file_list: List[str]):
        windows = []
        labels = []

        for filename in file_list:
            signal = signals_dict[filename]
            label = labels_dict[filename]

            # Preprocess
            signal_filtered = bandpass_filter(signal, fs=fs)

            # Create windows from entire signal
            wins, _ = create_windows(signal_filtered, window_size, overlap)

            for window in wins:
                windows.append(normalize_signal(window))
                labels.append(label)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)

    X_train, y_train = process_files(train_files)
    X_test, y_test = process_files(test_files)

    logger.info(f"Train windows: {len(X_train)}")
    logger.info(f"Test windows: {len(X_test)}")

    return X_train, y_train, X_test, y_test


class PreprocessingPipeline:
    """Complete preprocessing pipeline."""

    def __init__(
        self,
        window_size: int = 2048,
        overlap: float = 0.5,
        fs: int = 12000,
        lowcut: float = 10.0,
        highcut: float = 5000.0,
        normalize_method: str = 'zscore'
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.normalize_method = normalize_method

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """Process single signal into windows."""
        # Filter
        filtered = bandpass_filter(signal, self.lowcut, self.highcut, self.fs)

        # Create windows
        windows, _ = create_windows(filtered, self.window_size, self.overlap)

        # Normalize
        normalized = np.array([
            normalize_signal(w, method=self.normalize_method)
            for w in windows
        ])

        return normalized
