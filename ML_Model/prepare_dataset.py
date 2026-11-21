
from pathlib import Path
import urllib.request
import zipfile
import shutil
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "esc50.zip"
SOURCE_DIR = BASE_DIR / "esc50_source"
OUTPUT_DIR = BASE_DIR / "safety_audio_dataset"

# Official ESC-50 dataset
URL = (
    "https://github.com/karolpiczak/"
    "ESC-50/archive/refs/heads/master.zip"
)

# Map original dataset categories to our project classes
CLASS_MAP = {
    "background": [
        "rain",
        "wind",
        "clock_tick",
        "crackling_fire",
    ],
    "alarm": [
        "clock_alarm",
        "siren",
    ],
    "impact": [
        "clapping",
        "door_knock",
        "glass_breaking",
    ],
}


def main():
    # 1. Download dataset
    if not ZIP_PATH.exists():
        print("Downloading ESC-50 dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Download complete.")

    # 2. Extract dataset
    if not SOURCE_DIR.exists():
        print("Extracting dataset...")
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(SOURCE_DIR)

    dataset_root = SOURCE_DIR / "ESC-50-master"
    audio_dir = dataset_root / "audio"
    metadata_file = dataset_root / "meta" / "esc50.csv"

    if not metadata_file.exists():
        raise FileNotFoundError(
            "ESC-50 metadata was not found after extraction."
        )

    metadata = pd.read_csv(metadata_file)

    # 3. Create output folders
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}

    for label, categories in CLASS_MAP.items():
        label_dir = OUTPUT_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)

        selected = metadata[
            metadata["category"].isin(categories)
        ]

        copied = 0

        for _, row in selected.iterrows():
            source = audio_dir / row["filename"]
            destination = label_dir / row["filename"]

            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)
                copied += 1

        counts[label] = len(list(label_dir.glob("*.wav")))

    # 4. Report results
    print("\nDataset preparation finished.")
    print(f"Dataset folder: {OUTPUT_DIR}")

    for label, count in counts.items():
        print(f"{label}: {count} audio files")

    if any(count == 0 for count in counts.values()):
        raise RuntimeError(
            "At least one class is empty. Check the dataset."
        )

    print("\nNext: inspect the labels and prepare a "
          "leakage-free training/test split.")


if __name__ == "__main__":
    main()
