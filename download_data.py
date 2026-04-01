import os
import subprocess
import sys

DATASET = "imakash3011/online-shoppers-purchasing-intention-dataset"

def download_with_kaggle_cli():
    print("Trying Kaggle CLI...")
    
    kaggle_exe = os.path.join(sys.prefix, "Scripts", "kaggle.exe")

    if not os.path.exists(kaggle_exe):
        print("Kaggle CLI not found.")
        return False

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

        os.makedirs("data", exist_ok=True)

        subprocess.run([
            kaggle_exe,
            "datasets",
            "download",
            "-d", DATASET,
            "-p", "data",
            "--unzip"
        ], check=True)

        print("✅ Downloaded via Kaggle CLI")
        return True

    except Exception as e:
        print(f"Kaggle CLI failed: {e}")
        return False


def download_with_kagglehub():
    print("Trying kagglehub...")

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub"], check=True)

        import kagglehub
        import shutil

        path = kagglehub.dataset_download(DATASET)

        os.makedirs("data", exist_ok=True)

        for file in os.listdir(path):
            shutil.copy(os.path.join(path, file), "data")

        print("✅ Downloaded via kagglehub")
        return True

    except Exception as e:
        print(f"kagglehub failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting dataset download...\n")

    if not download_with_kaggle_cli():
        print("\nFalling back to kagglehub...\n")
        if not download_with_kagglehub():
            print("\n❌ Failed to download dataset. Please check setup.")
        else:
            print("\n🎉 Success with kagglehub!")
    else:
        print("\n🎉 Success with Kaggle CLI!")