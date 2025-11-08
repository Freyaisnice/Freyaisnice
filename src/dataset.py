"""
Dataset module for dog stool detection and classification.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class DogStoolDataset(Dataset):
    """
    Dataset for dog stool multi-task classification.

    Expected directory structure:
    data_dir/
        images/
            img1.jpg
            img2.jpg
            ...
        labels.json  # Contains multi-task labels

    labels.json format:
    {
        "img1.jpg": {
            "stool_type": "normal",
            "consistency": "formed",
            "color": "brown",
            "health": "healthy"
        },
        ...
    }
    """

    def __init__(
        self,
        data_dir: str,
        label_config: Dict[str, List[str]],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            data_dir: Directory containing images/ and labels.json
            label_config: Dictionary mapping task names to class labels
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_file = self.data_dir / "labels.json"
        self.label_config = label_config
        self.transform = transform
        self.image_size = image_size

        # Load labels
        if self.label_file.exists():
            with open(self.label_file, 'r', encoding='utf-8') as f:
                self.labels_dict = json.load(f)
        else:
            self.labels_dict = {}
            print(f"Warning: {self.label_file} not found. Using empty labels.")

        # Get image list
        if self.image_dir.exists():
            self.image_files = [
                f for f in os.listdir(self.image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
        else:
            self.image_files = []
            print(f"Warning: {self.image_dir} not found. Dataset is empty.")

        # Filter images that have labels
        if self.labels_dict:
            self.image_files = [
                f for f in self.image_files
                if f in self.labels_dict
            ]

        # Create label encoders
        self.label_encoders = {}
        for task_name, classes in label_config.items():
            self.label_encoders[task_name] = {
                class_name: idx for idx, class_name in enumerate(classes)
            }

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            labels: Dictionary with keys for each task, values are class indices
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name

        # Read image with OpenCV (for albumentations)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: resize and normalize
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Get labels
        labels = {}
        if img_name in self.labels_dict:
            img_labels = self.labels_dict[img_name]
            for task_name, class_name in img_labels.items():
                if task_name in self.label_encoders:
                    class_idx = self.label_encoders[task_name].get(class_name, 0)
                    labels[task_name] = torch.tensor(class_idx, dtype=torch.long)
                else:
                    labels[task_name] = torch.tensor(0, dtype=torch.long)
        else:
            # Default labels if not found
            for task_name in self.label_config.keys():
                labels[task_name] = torch.tensor(0, dtype=torch.long)

        return image, labels


def get_train_transforms(image_size: Tuple[int, int], config: Dict) -> A.Compose:
    """Get training augmentation pipeline."""
    aug_config = config.get('augmentation', {}).get('train', {})

    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_config.get('vertical_flip', 0.2)),
        A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
        A.ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.2),
            hue=aug_config.get('hue', 0.1),
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]

    return A.Compose(transforms)


def get_val_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """Get validation/test augmentation pipeline."""
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]

    return A.Compose(transforms)


def create_sample_dataset(output_dir: str):
    """
    Create a sample dataset structure for demonstration.

    Args:
        output_dir: Directory to create the sample dataset
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create sample labels.json
    sample_labels = {
        "sample_001.jpg": {
            "stool_type": "normal",
            "consistency": "formed",
            "color": "brown",
            "health": "healthy"
        },
        "sample_002.jpg": {
            "stool_type": "soft",
            "consistency": "soft_formed",
            "color": "yellow",
            "health": "monitor"
        },
        "sample_003.jpg": {
            "stool_type": "diarrhea",
            "consistency": "liquid",
            "color": "brown",
            "health": "concerning"
        }
    }

    labels_file = output_path / "labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(sample_labels, f, indent=2, ensure_ascii=False)

    print(f"Sample dataset structure created at {output_path}")
    print(f"Please add your images to {images_dir}")
    print(f"Labels template saved to {labels_file}")
