
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "esc50_source"
    / "ESC-50-master"
)

AUDIO_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "meta" / "esc50.csv"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

# --------------------------------------------------
# 2. Map ESC-50 categories to project classes
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Extract MFCC features from a 1-second window
# --------------------------------------------------

def extract_mfcc(window):
    """
    Return a fixed-size feature vector.
    Input: mono audio at 16 kHz, exactly 1 second.
    """

    window = np.asarray(window, dtype=np.float32)

    if len(window) != WINDOW_SIZE:
        raise ValueError(
            f"Expected {WINDOW_SIZE} samples, got {len(window)}"
        )

    mfcc = librosa.feature.mfcc(
        y=window,
        sr=SAMPLE_RATE,
        n_mfcc=20,
        n_fft=512,
        hop_length=256,
        n_mels=40,
    )

    # Summarize each MFCC over time.
    # 20 means + 20 standard deviations = 40 features.
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)

    return np.concatenate([means, stds])


# --------------------------------------------------
# 4. Load metadata and prepare windows
# --------------------------------------------------

def load_dataset():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    metadata = pd.read_csv(METADATA_FILE)

    # Verify that the expected category names exist.
    available_categories = set(metadata["category"].unique())
    missing_categories = (
        set(CATEGORY_MAP.keys()) - available_categories
    )

    if missing_categories:
        raise ValueError(
            "These mapped categories are missing from ESC-50: "
            f"{sorted(missing_categories)}\n"
            "Check the category names in esc50.csv."
        )

    selected = metadata[
        metadata["category"].isin(CATEGORY_MAP.keys())
    ].copy()

    features = []
    labels = []
    folds = []
    source_files = []

    for row in selected.itertuples(index=False):
        audio_path = AUDIO_DIR / row.filename

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio, original_sr = sf.read(audio_path)

        # Convert stereo audio to mono.
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Resample only if needed.
        if original_sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=original_sr,
                target_sr=SAMPLE_RATE,
            )

        audio = np.asarray(audio, dtype=np.float32)

        label = CATEGORY_MAP[row.category]

        # Use non-overlapping 1-second windows.
        number_of_windows = len(audio) // WINDOW_SIZE

        for index in range(number_of_windows):
            start = index * WINDOW_SIZE
            end = start + WINDOW_SIZE
            window = audio[start:end]

            features.append(extract_mfcc(window))
            labels.append(label)
            folds.append(int(row.fold))
            source_files.append(row.filename)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    fold_ids = np.asarray(folds)
    file_ids = np.asarray(source_files)

    if len(X) == 0:
        raise ValueError("No audio windows were loaded.")

    return X, y, fold_ids, file_ids


# --------------------------------------------------
# 5. Five-fold out-of-fold evaluation
# --------------------------------------------------

def main():
    print("Loading ESC-50 and extracting MFCC features...")
    X, y, fold_ids, file_ids = load_dataset()

    print(f"Total windows: {len(X)}")
    print(f"Feature count: {X.shape[1]}")
    print("Class counts:")
    print(pd.Series(y).value_counts().reindex(CLASS_NAMES))

    # Each original recording belongs to one ESC-50 fold.
    # All its windows remain in that same fold.
    all_true = []
    all_predicted = []
    fold_accuracies = []

    for held_out_fold in range(1, 6):
        train_mask = fold_ids != held_out_fold
        test_mask = fold_ids == held_out_fold

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            ),
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(accuracy)

        all_true.extend(y_test)
        all_predicted.extend(predictions)

        print(
            f"Fold {held_out_fold} accuracy: "
            f"{accuracy:.4f} "
            f"(test windows: {len(y_test)})"
        )

    all_true = np.asarray(all_true)
    all_predicted = np.asarray(all_predicted)

    print("\n--- FIVE-FOLD MFCC RESULTS ---")
    print(
        f"Mean fold accuracy: "
        f"{np.mean(fold_accuracies):.4f}"
    )
    print(
        f"Fold accuracy std:  "
        f"{np.std(fold_accuracies):.4f}"
    )
    print(
        f"Combined accuracy: "
        f"{accuracy_score(all_true, all_predicted):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            all_true,
            all_predicted,
            labels=CLASS_NAMES,
            zero_division=0,
            digits=3,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(CLASS_NAMES)
    print(
        confusion_matrix(
            all_true,
            all_predicted,
            labels=CLASS_NAMES,
        )
    )


if __name__ == "__main__":
    main()
