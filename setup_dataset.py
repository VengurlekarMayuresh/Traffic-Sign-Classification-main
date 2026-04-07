"""
Setup script to prepare the GTSRB dataset for the Traffic Sign Classification project.

The GTSRB dataset can be downloaded from Kaggle:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Steps:
1. Download the dataset from Kaggle (requires Kaggle account)
2. Extract the archive
3. Place the train.csv and test.csv in the project root's data/ folder
4. Place all images in the data/ folder as well
5. Run this script to create the data loaders

Alternatively, you can manually create the data directory structure:
data/
  train.csv
  test.csv
  [all image files referenced in the CSV files]
"""

import os
import sys

def check_and_create_data_structure():
    """Check if data directory exists and guide user."""

    print("="*60)
    print("GTSRB Dataset Setup Guide")
    print("="*60)
    print()

    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data/' directory")
    else:
        print("'data/' directory already exists")

    # Check for required files
    train_csv_exists = os.path.exists('data/train.csv')
    test_csv_exists = os.path.exists('data/test.csv')

    print()
    print("Status:")
    print(f"  train.csv: {'[OK]' if train_csv_exists else '[MISSING]'}")
    print(f"  test.csv:  {'[OK]' if test_csv_exists else '[MISSING]'}")

    # Count images
    if os.path.exists('data'):
        image_extensions = ('.ppm', '.png', '.jpg', '.jpeg')
        image_files = []
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        print(f"  Images:    {len(image_files)} found")
    else:
        print(f"  Images:    Not checked (data/ missing)")

    print()
    if not (train_csv_exists and test_csv_exists):
        print("To obtain the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print("2. Download the dataset (requires Kaggle login)")
        print("3. Extract the archive")
        print("4. Copy train.csv and test.csv to this project's data/ folder")
        print("5. Copy all image files to this project's data/ folder")
        print()
        print("Note: The dataset contains over 50,000 images (~300MB)")
        return False
    else:
        print("Dataset appears to be present - ready to create data loaders!")
        return True

if __name__ == "__main__":
    check_and_create_data_structure()
