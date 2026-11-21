
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "esc50_source" / "ESC-50-master"
METADATA = DATASET / "meta" / "esc50.csv"
AUDIO_DIR = DATASET / "audio"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
N_FFT = 512
HOP = 256

CATEGORY_MAP = {
    "clock_alarm": "alarm",
    "siren": "alarm",
    "clapping": "impact",
    "door_knock": "impact",
    "glass_breaking": "impact",
    "rain": "background",
    "wind": "background",
    "clock_tick": "background",
    "crackling_fire": "background",
}

CLASSES = ["alarm", "background", "impact"]


def extract_features(audio):
    """Match the project's one-second FFT feature approach."""
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

    window = audio[:SAMPLE_RATE]

    # Same log mean-magnitude FFT feature family as the original model.
    features = []

    for start in range(0, len(window) - N_FFT + 1, HOP):
        frame = window[start:start + N_FFT]
        spectrum = np.abs(np.fft.rfft(frame))
        features.append(spectrum)

    mean_spectrum = np.mean(features, axis=0)
    return np.log1p(mean_spectrum)


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // g, sr // g
        ).astype(np.float32)

    return audio


def main():
    if not METADATA.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA}")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio folder not found: {AUDIO_DIR}")

    metadata = pd.read_csv(METADATA)
    metadata["label"] = metadata["category"].map(CATEGORY_MAP)

    data = metadata[metadata["label"].notna()].copy()

    train_rows = data[data["fold"] != 5]
    test_rows = data[data["fold"] == 5]

    X_train, y_train = [], []
    X_test, y_test = [], []
    skipped = []

    for _, row in pd.concat(
        [train_rows, test_rows]
    ).iterrows():
        path = AUDIO_DIR / row["filename"]

        if not path.exists():
            skipped.append(row["filename"])
            continue

        audio = load_audio(path)
        label = row["label"]

        # One independent feature vector per second.
        for start in range(0, len(audio) - SAMPLE_RATE + 1, SAMPLE_RATE):
            segment = audio[start:start + SAMPLE_RATE]
            feat = extract_features(segment)

            if row["fold"] == 5:
                X_test.append(feat)
                y_test.append(label)
            else:
                X_train.append(feat)
                y_train.append(label)

    if not X_train or not X_test:
        raise RuntimeError(
            "No training or test samples loaded. Check dataset paths."
        )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced")
    )

    model.fit(np.asarray(X_train), np.asarray(y_train))
    predictions = model.predict(np.asarray(X_test))

    print("\n--- UNSEEN FOLD EVALUATION ---")
    print("Training recordings: folds 1–4")
    print("Unseen test recordings: fold 5")
    print(f"Training windows: {len(y_train)}")
    print(f"Test windows: {len(y_test)}")
    print(f"Skipped missing files: {len(skipped)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(CLASSES)
    print(confusion_matrix(
        y_test, predictions, labels=CLASSES
    ))


if __name__ == "__main__":
    main()
