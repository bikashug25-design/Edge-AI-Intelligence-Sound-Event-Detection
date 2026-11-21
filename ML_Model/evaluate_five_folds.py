
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf

from scipy.signal import resample_poly
from scipy.fft import rfft

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
META = DATASET / "meta" / "esc50.csv"
AUDIO_DIR = DATASET / "audio"

SR = 16000
N_SAMPLES = 16000
N_FFT = 512
HOP = 256

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

LABELS = ["alarm", "background", "impact"]


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != SR:
        g = gcd(sr, SR)
        audio = resample_poly(
            audio, SR // g, sr // g
        ).astype(np.float32)

    return audio


def extract_features(segment):
    frames = []

    for pos in range(0, N_SAMPLES - N_FFT + 1, HOP):
        frame = segment[pos:pos + N_FFT]
        spectrum = np.abs(
            rfft(frame * np.hanning(N_FFT))
        )
        frames.append(spectrum)

    mean_spectrum = np.mean(frames, axis=0)
    return np.log1p(mean_spectrum)


def main():
    metadata = pd.read_csv(META)
    metadata["label"] = metadata["category"].map(
        CATEGORY_MAP
    )
    metadata = metadata[
        metadata["label"].notna()
    ].copy()

    X, y, folds = [], [], []

    for _, row in metadata.iterrows():
        path = AUDIO_DIR / row["filename"]

        if not path.exists():
            raise FileNotFoundError(path)

        audio = load_audio(path)

        # Each recording stays entirely within its
        # original predefined ESC-50 fold.
        for start in range(
            0, len(audio) - N_SAMPLES + 1, N_SAMPLES
        ):
            segment = audio[start:start + N_SAMPLES]

            X.append(extract_features(segment))
            y.append(row["label"])
            folds.append(int(row["fold"]))

    X = np.asarray(X)
    y = np.asarray(y)
    folds = np.asarray(folds)

    all_true = []
    all_pred = []
    fold_scores = []

    for fold in range(1, 6):
        train_mask = folds != fold
        test_mask = folds == fold

        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            ),
        )

        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        truth = y[test_mask]

        score = accuracy_score(truth, pred)
        fold_scores.append(score)

        all_true.extend(truth)
        all_pred.extend(pred)

        print(
            f"Fold {fold} accuracy: {score:.4f} "
            f"(test windows: {len(truth)})"
        )

    print("\n--- FIVE-FOLD OUT-OF-FOLD RESULTS ---")
    print(f"Mean fold accuracy: {np.mean(fold_scores):.4f}")
    print(f"Fold accuracy std:  {np.std(fold_scores):.4f}")
    print(
        "Combined accuracy: "
        f"{accuracy_score(all_true, all_pred):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        all_true,
        all_pred,
        labels=LABELS,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(LABELS)
    print(confusion_matrix(
        all_true, all_pred, labels=LABELS
    ))


if __name__ == "__main__":
    main()
