# main.py
"""
Main training script with time-based split.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from src.data.data_loader import CWRUDataLoader
from src.data.preprocessing import time_based_split, hybrid_split, file_based_split
from src.models.vibration_cnn import VibrationCNN, count_parameters
from src.training.train import train_model, BearingDataset
from src.training.evaluate import evaluate_model
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/train_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_results(train_losses, train_accs, test_losses, test_accs, save_path='results.png'):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(train_losses, 'b-', linewidth=2, label='Train')
    axes[0].plot(test_losses, 'r-', linewidth=2, label='Test')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, 'b-', linewidth=2, label='Train')
    axes[1].plot(test_accs, 'r-', linewidth=2, label='Test')
    axes[1].axhline(y=95, color='g', linestyle='--',
                    alpha=0.5, label='Target (95%)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Test Accuracy',
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Results plot saved to {save_path}")


def main():
    """Main training pipeline."""

    print("=" * 70)
    print("BEARING FAULT DETECTION - TIME-BASED SPLIT")
    print("=" * 70)

    # Load config
    config = load_config()

    # Set random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    Path(config['paths']['model_dir']).mkdir(exist_ok=True)
    Path(config['paths']['experiments_dir']).mkdir(exist_ok=True)

    # Load data
    logger.info("Loading CWRU dataset...")
    loader = CWRUDataLoader(data_dir=config['paths']['data_dir'])
    signals_dict, metadata = loader.load_all_data()

    if len(signals_dict) == 0:
        logger.error("No data files found!")
        return

    labels_dict = loader.get_labels_dict(metadata)

    # Split data based on method
    split_method = config['data']['split_method']
    logger.info(f"Using split method: {split_method}")

    if split_method == 'time_based':
        X_train, y_train, X_test, y_test = time_based_split(
            signals_dict, labels_dict,
            train_time_ratio=config['data']['time_based']['train_time_ratio'],
            window_size=config['data']['window_size'],
            overlap=config['data']['overlap'],
            fs=config['data']['fs'],
            min_signal_length=config['data']['time_based']['min_signal_length']
        )

    elif split_method == 'hybrid':
        X_train, y_train, X_test, y_test = hybrid_split(
            signals_dict, labels_dict,
            file_train_ratio=config['data']['hybrid']['file_train_ratio'],
            time_train_ratio=config['data']['hybrid']['time_train_ratio'],
            window_size=config['data']['window_size'],
            overlap=config['data']['overlap'],
            fs=config['data']['fs'],
            seed=seed
        )

    elif split_method == 'file_based':
        X_train, y_train, X_test, y_test = file_based_split(
            signals_dict, labels_dict,
            train_ratio=config['data']['file_based']['train_ratio'],
            window_size=config['data']['window_size'],
            overlap=config['data']['overlap'],
            fs=config['data']['fs'],
            seed=seed
        )

    else:
        raise ValueError(f"Unknown split method: {split_method}")

    # Create data loaders
    train_dataset = BearingDataset(
        X_train, y_train,
        augment=config['training']['augment_train']
    )
    test_dataset = BearingDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    # Create model
    logger.info("Creating model...")
    model = VibrationCNN(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Train model
    logger.info("Starting training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_losses, train_accs, test_losses, test_accs, best_test_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    # Plot results
    plot_results(
        train_losses, train_accs, test_losses, test_accs,
        save_path=f"{config['paths']['experiments_dir']}/training_results.png"
    )

    # Final evaluation
    logger.info("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(
        f"{config['paths']['model_dir']}/best_model.pth"))
    model.to(device)

    metrics = evaluate_model(model, test_loader, device)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Split Method: {split_method}")
    print(f"Train Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Final Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.2f}%")
    print("=" * 70)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': {
            'best_test_acc': best_test_acc,
            'final_test_acc': metrics['accuracy'],
            'fnr': metrics['false_negative_rate']
        }
    }, f"{config['paths']['model_dir']}/final_model.pth")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
