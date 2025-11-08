"""
Training script for dog stool detection model.
"""

import os
import yaml
from pathlib import Path
from typing import Dict
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import create_model, MultiTaskLoss
from dataset import DogStoolDataset, get_train_transforms, get_val_transforms
from utils import AverageMeter, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class for multi-task stool classification."""

    def __init__(self, config: Dict, device: str = None):
        """
        Args:
            config: Configuration dictionary
            device: Device to use for training
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = create_model(config)
        self.model = self.model.to(self.device)

        # Create loss function
        loss_weights = config.get('training', {}).get('loss_weights', {})
        self.criterion = MultiTaskLoss(loss_weights)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Create data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create tensorboard writer
        if config.get('logging', {}).get('use_tensorboard', True):
            log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adam')
        lr = self.config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.0001)

        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('training', {}).get('scheduler', 'cosine')
        epochs = self.config.get('training', {}).get('epochs', 100)

        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        else:
            return None

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        data_config = self.config.get('data', {})
        image_size = tuple(data_config.get('image_size', [224, 224]))
        batch_size = data_config.get('batch_size', 32)
        num_workers = data_config.get('num_workers', 4)

        label_config = self.config.get('labels', {})

        # Create datasets
        train_dataset = DogStoolDataset(
            data_dir=data_config.get('train_dir', 'data/train'),
            label_config=label_config,
            transform=get_train_transforms(image_size, self.config),
            image_size=image_size
        )

        val_dataset = DogStoolDataset(
            data_dir=data_config.get('val_dir', 'data/val'),
            label_config=label_config,
            transform=get_val_transforms(image_size),
            image_size=image_size
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        return train_loader, val_loader

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Metrics
        loss_meters = {
            'total_loss': AverageMeter(),
            'stool_type_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),
            'color_loss': AverageMeter(),
            'health_loss': AverageMeter()
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for images, labels in pbar:
            # Move to device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss
            losses = self.criterion(outputs, labels)

            # Backward pass
            losses['total_loss'].backward()
            self.optimizer.step()

            # Update metrics
            for key, meter in loss_meters.items():
                if key in losses:
                    meter.update(losses[key].item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': loss_meters['total_loss'].avg})

        # Return average losses
        return {key: meter.avg for key, meter in loss_meters.items()}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        # Metrics
        loss_meters = {
            'total_loss': AverageMeter(),
            'stool_type_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),
            'color_loss': AverageMeter(),
            'health_loss': AverageMeter()
        }

        # Accuracy metrics
        correct_counts = {task: 0 for task in self.config.get('labels', {}).keys()}
        total_counts = {task: 0 for task in self.config.get('labels', {}).keys()}

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                losses = self.criterion(outputs, labels)

                # Update loss metrics
                for key, meter in loss_meters.items():
                    if key in losses:
                        meter.update(losses[key].item(), images.size(0))

                # Compute accuracy
                for task_name, output in outputs.items():
                    if task_name in labels:
                        pred = output.argmax(dim=1)
                        correct = (pred == labels[task_name]).sum().item()
                        correct_counts[task_name] += correct
                        total_counts[task_name] += labels[task_name].size(0)

                # Update progress bar
                pbar.set_postfix({'loss': loss_meters['total_loss'].avg})

        # Compute accuracies
        accuracies = {
            f'{task}_acc': correct_counts[task] / total_counts[task]
            for task in correct_counts.keys()
            if total_counts[task] > 0
        }

        # Combine losses and accuracies
        metrics = {key: meter.avg for key, meter in loss_meters.items()}
        metrics.update(accuracies)

        return metrics

    def train(self):
        """Main training loop."""
        epochs = self.config.get('training', {}).get('epochs', 100)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 10)

        print(f"\nStarting training on {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"Model: {self.model.backbone_name}")
        print("-" * 50)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"LR: {current_lr:.6f}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")

            # Print accuracies
            for task in self.config.get('labels', {}).keys():
                acc_key = f'{task}_acc'
                if acc_key in val_metrics:
                    print(f"Val {task} Acc: {val_metrics[acc_key]:.4f}")

            # TensorBoard logging
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar('learning_rate', current_lr, epoch)

            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save regular checkpoint
            save_freq = self.config.get('checkpoint', {}).get('save_freq', 5)
            if (epoch + 1) % save_freq == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['total_loss'],
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                )

            # Save best checkpoint
            if is_best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['total_loss'],
                    self.checkpoint_dir / 'best_model.pth'
                )
                print(f"✓ Saved best model (loss: {self.best_val_loss:.4f})")

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            print("-" * 50)

        print("\nTraining completed!")
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train dog stool detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and start training
    trainer = Trainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
