
from pathlib import Path
import numpy as np
import soundfile as sf
import joblib

from scipy.signal import resample_poly
from scipy.fft import rfft
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "safety_audio_dataset"
MODEL_DIR = BASE_DIR / "model_output"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
N_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
N_FFT = 512
HOP = 256


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // g, sr // g
        )

    return audio


def extract_features(audio):
    # Split audio into one-second windows
    features = []

    for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES):
        segment = audio[start:start + N_SAMPLES]

        # Short-time spectral energy features
        frames = []
        for pos in range(0, N_SAMPLES - N_FFT + 1, HOP):
            frame = segment[pos:pos + N_FFT]
            spectrum = np.abs(rfft(frame * np.hanning(N_FFT)))
            frames.append(spectrum)

        spectrum_mean = np.mean(frames, axis=0)
        features.append(np.log1p(spectrum_mean))

    return features


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "Run prepare_dataset.py first."
        )

    X, y, groups = [], [], []

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for wav_path in sorted(class_dir.glob("*.wav")):
            audio = load_audio(wav_path)
            feats = extract_features(audio)

            for feature in feats:
                X.append(feature)
                y.append(label)
                groups.append(wav_path.name)

    if not X:
        raise RuntimeError(
            "No audio windows found in the dataset."
        )

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    # Split by original recording, not by window
    unique_files = np.unique(groups)
    train_files, test_files = train_test_split(
        unique_files,
        test_size=0.25,
        random_state=42,
    )

    train_mask = np.isin(groups, train_files)
    test_mask = np.isin(groups, test_files)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    )

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTest accuracy:",
          round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion matrix (labels shown below):")
    labels = sorted(np.unique(y))
    print(labels)
    print(confusion_matrix(
        y_test, predictions, labels=labels
    ))

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "labels": labels,
        },
        MODEL_DIR / "sound_classifier.joblib",
    )

    print("\nSaved model to:",
          MODEL_DIR / "sound_classifier.joblib")
    print("This model runs on your PC; it is not yet "
          "an ESP32 firmware model.")


if __name__ == "__main__":
    main()
