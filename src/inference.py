"""
Inference script for dog stool detection model.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import cv2
import numpy as np
from PIL import Image

from model import create_model
from dataset import get_val_transforms
from utils import load_checkpoint


class StoolDetector:
    """Dog stool detector and classifier."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = None
    ):
        """
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to use for inference
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)

        # Load checkpoint
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        # Get labels
        self.labels = self.config.get('labels', {})

        # Get transforms
        image_size = tuple(self.config.get('data', {}).get('image_size', [224, 224]))
        self.transform = get_val_transforms(image_size)

        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict(self, image_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predict stool characteristics from image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary mapping task names to class probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Convert logits to probabilities
        results = {}
        for task_name, logits in outputs.items():
            probabilities = torch.softmax(logits, dim=1)[0]
            class_names = self.labels.get(task_name, [])

            task_results = {
                class_name: prob.item()
                for class_name, prob in zip(class_names, probabilities)
            }
            results[task_name] = task_results

        return results

    def predict_top(self, image_path: str) -> Dict[str, Tuple[str, float]]:
        """
        Predict top class for each task.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary mapping task names to (top_class, confidence)
        """
        results = self.predict(image_path)

        top_predictions = {}
        for task_name, class_probs in results.items():
            top_class = max(class_probs.items(), key=lambda x: x[1])
            top_predictions[task_name] = top_class

        return top_predictions

    def analyze_image(self, image_path: str, verbose: bool = True) -> Dict:
        """
        Analyze image and provide detailed results.

        Args:
            image_path: Path to image file
            verbose: Whether to print results

        Returns:
            Dictionary containing analysis results
        """
        # Get predictions
        all_probs = self.predict(image_path)
        top_preds = self.predict_top(image_path)

        # Build result
        analysis = {
            'image_path': image_path,
            'predictions': {}
        }

        for task_name, (top_class, confidence) in top_preds.items():
            analysis['predictions'][task_name] = {
                'class': top_class,
                'confidence': confidence,
                'all_probabilities': all_probs[task_name]
            }

        # Print results
        if verbose:
            self.print_analysis(analysis)

        return analysis

    def print_analysis(self, analysis: Dict):
        """Print analysis results in a readable format."""
        print("\n" + "=" * 60)
        print(f"Analysis Results: {analysis['image_path']}")
        print("=" * 60)

        task_names_chinese = {
            'stool_type': '大便类型',
            'consistency': '大便状态',
            'color': '大便颜色',
            'health': '健康评估'
        }

        for task_name, result in analysis['predictions'].items():
            task_display = task_names_chinese.get(task_name, task_name)
            print(f"\n{task_display} ({task_name}):")
            print(f"  预测结果: {result['class']}")
            print(f"  置信度: {result['confidence']:.2%}")

            # Show top 3 probabilities
            sorted_probs = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            print("  Top 3 可能性:")
            for i, (class_name, prob) in enumerate(sorted_probs, 1):
                print(f"    {i}. {class_name}: {prob:.2%}")

        print("=" * 60 + "\n")

    def batch_predict(self, image_dir: str, output_file: str = None):
        """
        Predict on multiple images in a directory.

        Args:
            image_dir: Directory containing images
            output_file: Path to save results (JSON)
        """
        image_dir = Path(image_dir)
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]

        print(f"Found {len(image_files)} images")

        results = []
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}...")
            analysis = self.analyze_image(str(image_file), verbose=False)
            results.append(analysis)

        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Dog stool detection inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image-dir', type=str, help='Directory of images for batch prediction')
    parser.add_argument('--output', type=str, help='Output file for batch predictions')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create detector
    detector = StoolDetector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Single image prediction
    if args.image:
        detector.analyze_image(args.image)

    # Batch prediction
    elif args.image_dir:
        detector.batch_predict(args.image_dir, args.output)

    else:
        print("Please provide either --image or --image-dir")


if __name__ == '__main__':
    main()
