"""
Step 1: Download the TMDB Movie Dataset using KaggleHub
This script downloads the dataset and copies it to our project folder.
"""

import kagglehub
import shutil
import os

def download_dataset():
    print("=" * 50)
    print("📥 Downloading TMDB Movie Dataset...")
    print("=" * 50)
    
    # Download latest version from Kaggle
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    
    print(f"\n✅ Dataset downloaded to: {path}")
    
    # List the files in the downloaded folder
    print("\n📁 Files in dataset:")
    for file in os.listdir(path):
        print(f"   - {file}")
    
    # Copy files to current directory for easier access
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            src = os.path.join(path, file)
            dst = os.path.join(data_dir, file)
            shutil.copy(src, dst)
            print(f"\n📋 Copied {file} to {data_dir}")
    
    print("\n" + "=" * 50)
    print("✅ Dataset ready! Files are in the 'data' folder.")
    print("=" * 50)
    
    return data_dir

if __name__ == "__main__":
    download_dataset()
