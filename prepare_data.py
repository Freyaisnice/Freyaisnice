#!/usr/bin/env python
"""
Script to prepare dataset structure and create sample labels.
"""

import argparse
from pathlib import Path
from src.dataset import create_sample_dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset structure')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to create')
    args = parser.parse_args()

    output_dir = Path(args.output)

    for split in args.splits:
        split_dir = output_dir / split
        print(f"\nCreating {split} dataset structure...")
        create_sample_dataset(str(split_dir))

    print("\n" + "=" * 60)
    print("Dataset structure created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your images to the 'images/' directories")
    print("2. Update the 'labels.json' files with your annotations")
    print("3. Run training: python src/train.py")
    print("\nFor annotation format, see the sample labels.json files.")


if __name__ == '__main__':
    main()
