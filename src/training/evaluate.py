"""
Model evaluation utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""

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

    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

    def evaluate(self) -> Dict:
        """
        Evaluate model and return metrics.

        Returns:
            Dictionary with all metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for signals, labels in self.test_loader:
                signals = signals.to(self.device)
                outputs = self.model(signals)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = 100.0 * np.mean(all_preds == all_labels)

        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.CLASS_LABELS.values()),
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Binary metrics (Normal vs Fault)
        binary_labels = (all_labels > 0).astype(int)
        binary_preds = (all_preds > 0).astype(int)

        tn = np.sum((binary_labels == 0) & (binary_preds == 0))
        fp = np.sum((binary_labels == 0) & (binary_preds == 1))
        fn = np.sum((binary_labels == 1) & (binary_preds == 0))
        tp = np.sum((binary_labels == 1) & (binary_preds == 1))

        fnr = 100.0 * fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = 100.0 * fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'false_negative_rate': fnr,
            'false_positive_rate': fpr,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

        return metrics

    def print_report(self, metrics: Dict):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"False Negative Rate: {metrics['false_negative_rate']:.2f}%")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.2f}%")
        print("\nPer-Class Metrics:")

        report = metrics['classification_report']
        for class_name in self.CLASS_LABELS.values():
            if class_name in report:
                cls_metrics = report[class_name]
                print(f"  {class_name:20s}: Precision={cls_metrics['precision']:.3f}, "
                      f"Recall={cls_metrics['recall']:.3f}, "
                      f"F1={cls_metrics['f1-score']:.3f}")

        print("=" * 60)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict:
    """
    Quick evaluation function.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device

    Returns:
        Metrics dictionary
    """
    evaluator = ModelEvaluator(model, test_loader, device)
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)
    return metrics
