#!/bin/bash

# Function to install Git LFS if not installed
install_git_lfs() {
    echo "Git LFS is not installed. Installing..."

    # Check for the OS type and install Git LFS accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Ubuntu/Debian
        sudo apt update
        sudo apt install git-lfs -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # For macOS
        brew install git-lfs
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32" ]]; then
        # For Windows (using Chocolatey)
        choco install git-lfs
    else
        echo "Unsupported OS. Please install Git LFS manually."
        exit 1
    fi

    # Initialize Git LFS
    git lfs install
    echo "Git LFS installed successfully!"
}

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    install_git_lfs
fi

REPO_URL=https://huggingface.co/datasets/amirsh1376/FeelAnyForce/
TEMP_DIR=FAF_HF
FILES_TO_FETCH=dataset.z01,dataset.z02,dataset.z03,dataset.zip

GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$TEMP_DIR"


cd $TEMP_DIR
git lfs fetch --include="$FILES_TO_FETCH"

zip -s 0 dataset.zip --out merged.zip
rm dataset.z*
unzip merged.zip -d ../dataset

cd ..
cp TacForce_train_set.csv dataset/
cp TacForce_val_set.csv dataset/
cp TacForce_test_set.csv dataset/

rm -r $TEMP_DIR