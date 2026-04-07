"""
CWRU Bearing Dataset Loader with Time-Based Split Support.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CWRUDataLoader:
    """
    Load and organize CWRU bearing dataset.

    Supports:
    - Standard file-based loading
    - Time-based splitting for realistic evaluation
    - Metadata extraction
    """

    # File to class mapping
    FILE_MAPPING = {
        # Normal baseline
        '97.mat': 0, '98.mat': 0, '99.mat': 0, '100.mat': 0,

        # Inner race faults - 0.007 inch
        '105.mat': 1, '106.mat': 1, '107.mat': 1, '108.mat': 1,

        # Inner race faults - 0.014 inch
        '169.mat': 2, '170.mat': 2, '171.mat': 2, '172.mat': 2,

        # Inner race faults - 0.021 inch
        '209.mat': 3, '210.mat': 3, '211.mat': 3, '212.mat': 3,

        # Outer race faults - 0.007 inch
        '130.mat': 4, '131.mat': 4, '132.mat': 4, '133.mat': 4,

        # Outer race faults - 0.014 inch
        '197.mat': 5, '198.mat': 5, '199.mat': 5, '200.mat': 5,

        # Outer race faults - 0.021 inch
        '234.mat': 6, '235.mat': 6, '236.mat': 6, '237.mat': 6,

        # Ball faults - 0.007 inch
        '118.mat': 7, '119.mat': 7, '120.mat': 7, '121.mat': 7,

        # Ball faults - 0.014 inch
        '185.mat': 8, '186.mat': 8, '187.mat': 8, '188.mat': 8,

        # Ball faults - 0.021 inch
        '222.mat': 9, '223.mat': 9, '224.mat': 9, '225.mat': 9,
    }

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

    def __init__(self, data_dir: str = './data/cwru'):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing .mat files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _extract_signal_key(self, mat_data: Dict) -> str:
        """Find drive-end accelerometer signal in .mat file."""
        for key in mat_data.keys():
            if 'DE_time' in key:
                return key
        raise ValueError(f"No DE signal found in keys: {mat_data.keys()}")

    def load_mat_file(self, filename: str) -> Tuple[np.ndarray, Dict]:
        """
        Load a single .mat file.

        Args:
            filename: Name of .mat file

        Returns:
            signal: 1D numpy array
            metadata: Dictionary with file info
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} not found")

        # Load .mat file
        mat_data = loadmat(filepath)

        # Extract signal
        signal_key = self._extract_signal_key(mat_data)
        signal = mat_data[signal_key].flatten().astype(np.float32)

        # Determine sampling rate and downsample if needed
        file_num = int(filename.split('.')[0])
        if file_num < 169:
            fs = 12000
        else:
            fs = 48000
            # Downsample to 12 kHz
            signal = signal[::4]
            fs = 12000

        metadata = {
            'filename': filename,
            'signal_key': signal_key,
            'sampling_rate': fs,
            'num_samples': len(signal),
            'duration_sec': len(signal) / fs,
            'class_id': self.FILE_MAPPING.get(filename, -1),
            'class_name': self.CLASS_LABELS.get(self.FILE_MAPPING.get(filename, -1), 'Unknown')
        }

        return signal, metadata

    def load_all_data(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Load entire dataset.

        Returns:
            signals_dict: Dictionary mapping filename to signal
            metadata_df: DataFrame with file information
        """
        signals_dict = {}
        metadata_list = []

        logger.info("Loading CWRU dataset...")

        for filename, class_id in tqdm(self.FILE_MAPPING.items(), desc="Loading files"):
            try:
                signal, meta = self.load_mat_file(filename)
                signals_dict[filename] = signal
                metadata_list.append(meta)
            except FileNotFoundError:
                logger.warning(f"File {filename} not found, skipping...")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        metadata_df = pd.DataFrame(metadata_list)

        logger.info(f"Loaded {len(signals_dict)} files successfully")
        logger.info(
            f"Class distribution:\n{metadata_df['class_name'].value_counts()}")

        return signals_dict, metadata_df

    def get_labels_dict(self, metadata_df: pd.DataFrame) -> Dict[str, int]:
        """
        Create filename to class_id mapping.

        Args:
            metadata_df: Metadata DataFrame

        Returns:
            Dictionary mapping filename to class_id
        """
        return {row['filename']: row['class_id'] for _, row in metadata_df.iterrows()}
