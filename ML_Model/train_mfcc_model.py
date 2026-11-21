
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------- PATHS ----------
PROJECT_DIR = Path(__file__).resolve().parent
ESC50_DIR = PROJECT_DIR / "esc50_source" / "ESC-50-master"
AUDIO_DIR = ESC50_DIR / "audio"
METADATA_FILE = ESC50_DIR / "meta" / "esc50.csv"
OUTPUT_DIR = PROJECT_DIR / "model_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- AUDIO SETTINGS ----------
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 40

CATEGORY_MAP = {
    "clock_alarm": "alarm",
    "siren": "alarm",
    "rain": "background",
    "wind": "background",
    "clock_tick": "background",
    "crackling_fire": "background",
    "clapping": "impact",
    "door_wood_knock": "impact",
    "glass_breaking": "impact",
}

CLASS_NAMES = ["alarm", "background", "impact"]


def extract_mfcc_features(window):
    """Extract the same MFCC features used in evaluation."""
    mfcc = librosa.feature.mfcc(
        y=window.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])


def main():
    if not METADATA_FILE.exists() or not AUDIO_DIR.exists():
        raise FileNotFoundError(
            "ESC-50 files not found. Check ESC50_DIR in this script."
        )

    metadata = pd.read_csv(METADATA_FILE)

    metadata = metadata[
        metadata["category"].isin(CATEGORY_MAP.keys())
    ].copy()

    features = []
    labels = []
    files_used = set()

    print("Extracting MFCC features from ESC-50...")

    for _, row in metadata.iterrows():
        audio_path = AUDIO_DIR / row["filename"]

        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path.name}")
            continue

        audio, sr = sf.read(audio_path, dtype="float32")

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
            )

        label = CATEGORY_MAP[row["category"]]

        # ESC-50 recordings are normally 5 seconds long.
        # Keep complete one-second windows only.
        number_of_windows = len(audio) // WINDOW_SIZE

        for i in range(number_of_windows):
            start = i * WINDOW_SIZE
            window = audio[start:start + WINDOW_SIZE]

            features.append(extract_mfcc_features(window))
            labels.append(label)

        files_used.add(row["filename"])

    if not features:
        raise RuntimeError("No audio features were extracted.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    print(f"Original recordings used: {len(files_used)}")
    print(f"Total one-second windows: {len(X)}")
    print(f"Feature count: {X.shape[1]}")
    print("\nClass counts:")
    print(pd.Series(y).value_counts())

    # Final model: train on all selected recordings after
    # completing the model-selection evaluation.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

    print("\nTraining final MFCC model...")
    model.fit(X, y)

    model_path = OUTPUT_DIR / "mfcc_sound_classifier.joblib"

    bundle = {
        "model": model,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_size": WINDOW_SIZE,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "class_names": CLASS_NAMES,
        "category_map": CATEGORY_MAP,
        "training_dataset": "ESC-50 selected categories",
        "feature_type": "MFCC mean and standard deviation",
    }

    joblib.dump(bundle, model_path)

    print(f"\nModel saved successfully:\n{model_path}")
    print("\nNext: test this model on recordings not used for training.")


if __name__ == "__main__":
    main()
