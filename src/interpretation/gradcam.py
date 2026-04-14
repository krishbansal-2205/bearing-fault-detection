"""
Gradient-weighted Class Activation Mapping for 1D signals.

Provides visual explanations of which temporal regions in vibration signals
are most important for the model's fault predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import cv2


class GradCAM1D:
    """
    Grad-CAM for 1D convolutional networks.

    Shows which time samples in the vibration signal are most important
    for the model's prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained CNN model
            target_layer: Layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks (store handles for cleanup)
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate class activation map.

        Args:
            input_signal: Input signal tensor [1, 1, length]
            target_class: Class to visualize (None = predicted class)

        Returns:
            cam: Activation map [length] with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_signal)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # [channels, length]
        activations = self.activations[0]  # [channels, length]

        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=1, keepdim=True)  # [channels, 1]

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # [length]

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(
        self,
        input_signal: torch.Tensor,
        true_label: int,
        pred_label: int,
        save_path: str,
        fs: int = 12000,
        class_names: Optional[dict] = None
    ):
        """
        Create comprehensive visualization of Grad-CAM results.

        Args:
            input_signal: Input signal [1, 1, length]
            true_label: Ground truth class
            pred_label: Predicted class
            save_path: Path to save figure
            fs: Sampling frequency
            class_names: Dictionary mapping class ID to name
        """
        # Generate CAM
        cam = self.generate_cam(input_signal, target_class=pred_label)

        # Get signal as numpy array
        signal = input_signal.squeeze().cpu().numpy()

        # Resize CAM to match signal length
        if len(cam) != len(signal):
            cam = cv2.resize(cam, (len(signal), 1)).squeeze()

        # Create time axis
        time = np.arange(len(signal)) / fs

        # Get class names
        if class_names is None:
            class_names = {i: f"Class_{i}" for i in range(10)}

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        # 1. Original signal
        axes[0].plot(time, signal, color='black', linewidth=0.8)
        axes[0].set_title(
            f'Input Signal | True: {class_names[true_label]} | Pred: {class_names[pred_label]}',
            fontsize=12,
            fontweight='bold'
        )
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Acceleration (g)')
        axes[0].grid(True, alpha=0.3)

        # 2. CAM overlay on signal
        axes[1].plot(time, signal, color='black', linewidth=0.8, alpha=0.4)
        im = axes[1].imshow(
            cam[np.newaxis, :],
            cmap='jet',
            aspect='auto',
            extent=[time[0], time[-1], signal.min(), signal.max()],
            alpha=0.6
        )
        axes[1].set_title(
            'Grad-CAM Overlay (Red = High Importance)', fontsize=12)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Acceleration (g)')
        plt.colorbar(im, ax=axes[1], label='Activation')

        # 3. Pure activation heatmap
        axes[2].plot(time, cam, color='red', linewidth=2)
        axes[2].fill_between(time, 0, cam, alpha=0.3, color='red')
        axes[2].set_title('Activation Strength Over Time', fontsize=12)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Normalized Activation')
        axes[2].axhline(y=0.7, color='green', linestyle='--', linewidth=2,
                        label='High Importance Threshold (0.7)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

        # 4. Highlighted important regions
        threshold = 0.7
        important_mask = cam > threshold

        axes[3].plot(time, signal, color='gray', linewidth=0.8, alpha=0.3,
                     label='Full Signal')
        axes[3].plot(time[important_mask], signal[important_mask],
                     color='red', linewidth=1.5, marker='o', markersize=2,
                     label='Important Regions (Activation > 0.7)')
        axes[3].set_title(
            'Important Temporal Regions for Classification', fontsize=12)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Acceleration (g)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Grad-CAM visualization saved to {save_path}")


def visualize_learned_filters(
    model: torch.nn.Module,
    layer_name: str = 'conv1',
    save_path: str = 'learned_filters.png'
):
    """
    Visualize learned convolutional filters.

    Shows what temporal patterns the first layer detects.

    Args:
        model: Trained model
        layer_name: Name of conv layer to visualize
        save_path: Path to save figure
    """
    # Get conv layer by name
    modules_dict = dict(model.named_modules())
    layer = modules_dict.get(layer_name)
    if layer is None:
        # Fallback: find first Conv1d layer
        import torch.nn as tnn
        for name, module in model.named_modules():
            if isinstance(module, tnn.Conv1d):
                layer = module
                layer_name = name
                break
    if layer is None:
        raise ValueError(f"No conv layer found with name '{layer_name}'")

    # Get weights [out_channels, in_channels, kernel_size]
    weights = layer.weight.data.cpu().numpy()

    num_filters = weights.shape[0]
    rows = int(np.sqrt(num_filters))
    cols = (num_filters + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_filters):
        if i < num_filters:
            filter_weights = weights[i, 0, :]  # [kernel_size]
            axes[i].plot(filter_weights, linewidth=1, color='blue')
            axes[i].set_title(f'Filter {i}', fontsize=8)
            axes[i].axis('off')
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Learned Filters in {layer_name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Filter visualization saved to {save_path}")


# Example usage
if __name__ == "__main__":
    from src.models.vibration_cnn import VibrationCNN

    # Load model
    model = VibrationCNN(num_classes=10)
    checkpoint = torch.load('./models/best_model.pth', map_location='cpu',
                            weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Create dummy input
    dummy_signal = torch.randn(1, 1, 2048)

    # Initialize Grad-CAM (target last conv layer before pooling)
    # model.features[10] is Conv1d(64, 128, kernel_size=16)
    target_layer = model.features[10]
    gradcam = GradCAM1D(model, target_layer)

    # Generate visualization
    CLASS_LABELS = {
        0: "Normal", 1: "IR007", 2: "IR014", 3: "IR021",
        4: "OR007", 5: "OR014", 6: "OR021",
        7: "B007", 8: "B014", 9: "B021"
    }

    gradcam.visualize(
        input_signal=dummy_signal,
        true_label=0,
        pred_label=0,
        save_path='gradcam_example.png',
        class_names=CLASS_LABELS
    )

    # Clean up hooks
    gradcam.remove_hooks()

    # Visualize learned filters (features.0 is the first Conv1d)
    visualize_learned_filters(
        model, layer_name='features.0', save_path='filters.png')
