#!/bin/bash

# Exit on error
set -e

# Check for Kaggle API credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "❌ kaggle.json not found in ~/.kaggle/"
  echo "Please place your API credentials from https://www.kaggle.com/account"
  exit 1
fi

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json

# Create target directory
mkdir -p soil_classification_data
cd soil_classification_data

# Download the dataset
echo "📥 Downloading dataset..."
kaggle competitions download -c soil-classification

# Unzip the dataset
echo "📦 Extracting dataset..."
unzip -q soil-classification.zip
rm soil-classification.zip

echo "✅ Download and extraction complete!"
