
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import joblib


# ---------- PATHS ----------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model_output" / "mfcc_sound_classifier.joblib"
TEST_AUDIO = PROJECT_DIR / "test_audio" / "test_sound.wav"


def extract_features(window, config):
    mfcc = librosa.feature.mfcc(
        y=window.astype(np.float32),
        sr=config["sample_rate"],
        n_mfcc=config["n_mfcc"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ])


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    if not TEST_AUDIO.exists():
        print(f"Test audio not found: {TEST_AUDIO}")
        print("Place a WAV file at test_audio/test_sound.wav")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    audio, sr = sf.read(TEST_AUDIO, dtype="float32")

    # Convert stereo to mono.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample to the training sample rate.
    if sr != bundle["sample_rate"]:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=bundle["sample_rate"],
        )

    window_size = bundle["window_size"]
    total_windows = len(audio) // window_size

    if total_windows == 0:
        print("Audio must contain at least one complete second.")
        return

    print(f"\nAudio file: {TEST_AUDIO.name}")
    print(f"Duration: {len(audio) / bundle['sample_rate']:.2f} seconds")
    print(f"Complete windows: {total_windows}\n")

    for i in range(total_windows):
        start = i * window_size
        window = audio[start:start + window_size]

        features = extract_features(window, bundle).reshape(1, -1)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        classes = model.named_steps["classifier"].classes_

        confidence = probabilities[
            np.where(classes == prediction)[0][0]
        ] * 100

        print(
            f"{i + 1:02d}. "
            f"{i:.1f}-{i + 1:.1f}s | "
            f"Prediction: {prediction.upper():10s} | "
            f"Model confidence: {confidence:.1f}%"
        )

    print("\nTesting complete.")
    print("Note: model confidence is not a guarantee of correctness.")


if __name__ == "__main__":
    main()
