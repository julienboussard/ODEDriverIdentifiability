import os
import tarfile
import shutil
import requests

def download_file(url: str, dest: str):
    """Download a file from a URL to a destination path."""
    if os.path.exists(dest):
        print(f"File '{dest}' already exists. Skipping download.")
        return
    print(f"Downloading {url} → {dest}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {dest}")

def extract_tar_gz(filepath: str, target_dir: str):
    """Extract a .tar.gz archive to a target directory."""
    print(f"Extracting {filepath} → {target_dir}")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"Extracted: {filepath}")

def main():
    base_url = "https://huggingface.co/datasets/kausable/CausalDynamics/resolve/main"
    subsets = ["climate", "simple", "coupled"]
    stages = ["inputs", "outputs"]

    base_dir = os.getcwd()
    target_base = os.path.join(base_dir, "data")
    os.makedirs(target_base, exist_ok=True)

    for stage in stages:
        stage_dir = os.path.join(base_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)

        for subset in subsets:
            filename = f"{subset}.tar.gz"
            url = f"{base_url}/{stage}/{filename}"
            dest_path = os.path.join(stage_dir, filename)

            download_file(url, dest_path)
            extract_tar_gz(dest_path, target_base)

    # Remove downloaded input/output directories
    for stage in stages:
        dir_to_remove = os.path.join(base_dir, stage)
        if os.path.isdir(dir_to_remove):
            shutil.rmtree(dir_to_remove)
            print(f"Removed: {dir_to_remove}")

if __name__ == "__main__":
    main()

