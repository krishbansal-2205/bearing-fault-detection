"""
Training pipeline with time-based split support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BearingDataset(Dataset):
    """PyTorch dataset for bearing vibration signals."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        Initialize dataset.

        Args:
            X: Signal windows [num_samples, window_size]
            y: Labels [num_samples]
            augment: Whether to apply data augmentation
        """
        self.X = torch.from_numpy(X).unsqueeze(
            1).float()  # [N, 1, window_size]
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        # Optional augmentation
        if self.augment and torch.rand(1).item() < 0.3:
            # Add noise
            noise = torch.randn_like(x) * 0.05
            x = x + noise

            # Random scaling
            scale = 0.9 + torch.rand(1).item() * 0.2
            x = x * scale

        return x, y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience: int = 10
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    Train model with early stopping.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        early_stopping_patience: Patience for early stopping

    Returns:
        train_losses, train_accs, test_losses, test_accs, best_test_acc
    """
    model = model.to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    # Tracking
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0.0
    patience_counter = 0

    logger.info(f"Training on {device}")
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # ========== Training ==========
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ========== Testing ==========
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        test_loss = running_loss / len(test_loader)
        test_acc = 100.0 * correct / total

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update scheduler
        scheduler.step()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:5.1f}% - "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:5.1f}%"
            )

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model
            import os
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict()
            }, 'models/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"Training complete. Best test accuracy: {best_test_acc:.2f}%")

    return train_losses, train_accs, test_losses, test_accs, best_test_acc
