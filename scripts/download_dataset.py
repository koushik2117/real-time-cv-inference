import os
import urllib.request
import tarfile
from pathlib import Path

def download_cifar10():
    """Download CIFAR-10 dataset if not present"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_path = data_dir / "cifar-10-python.tar.gz"
    
    if not file_path.exists():
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
        
        print("Extracting...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_cifar10()