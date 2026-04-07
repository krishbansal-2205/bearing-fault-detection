"""
1D CNN for bearing fault classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VibrationCNN(nn.Module):
    """
    1D Convolutional Neural Network for bearing fault classification.

    Architecture:
    - 3 convolutional blocks with increasing channels
    - Batch normalization and dropout for regularization
    - Global average pooling
    - Fully connected classifier

    Input: [batch, 1, 2048] - Single channel vibration signal
    Output: [batch, num_classes] - Class logits
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5
    ):
        """
        Initialize model.

        Args:
            num_classes: Number of fault classes
            dropout_rate: Dropout probability
        """
        super(VibrationCNN, self).__init__()

        self.num_classes = num_classes

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: High-frequency feature extraction
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=4),

            # Block 2: Mid-level patterns
            nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=4),

            # Block 3: High-level features
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 1, signal_length]

        Returns:
            Class logits [batch, num_classes]
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features before classification.

        Args:
            x: Input tensor [batch, 1, signal_length]

        Returns:
            Features [batch, 128]
        """
        features = self.features(x)
        return features.squeeze(-1)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        total_params: Total parameter count
        trainable_params: Trainable parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
