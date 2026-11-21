
from pathlib import Path
from math import gcd

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import rfft

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_output" / "sound_classifier.joblib"
AUDIO_PATH = BASE_DIR / "test_audio" / "test_sound.wav"

SAMPLE_RATE = 16000
N_SAMPLES = SAMPLE_RATE
N_FFT = 512
HOP = 256


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // g, sr // g
        )

    return audio


def extract_features(segment):
    frames = []

    for pos in range(0, N_SAMPLES - N_FFT + 1, HOP):
        frame = segment[pos:pos + N_FFT]
        spectrum = np.abs(
            rfft(frame * np.hanning(N_FFT))
        )
        frames.append(spectrum)

    return np.log1p(np.mean(frames, axis=0))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio not found: {AUDIO_PATH}"
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    audio = load_audio(AUDIO_PATH)

    if len(audio) < N_SAMPLES:
        raise ValueError("Audio must be at least 1 second long.")

    print("\n--- SOUND CLASSIFICATION RESULTS ---")
    print(f"Audio file: {AUDIO_PATH.name}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.2f} seconds")

    window_number = 0

    for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES):
        segment = audio[start:start + N_SAMPLES]
        features = extract_features(segment).reshape(1, -1)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100

        window_number += 1
        print(f"\nWindow {window_number}:")
        print(f"Predicted class: {prediction}")
        print(f"Model confidence: {confidence:.1f}%")

        for label, probability in zip(model.classes_, probabilities):
            print(f"  {label}: {probability * 100:.1f}%")

    print("\nTest finished.")
    print("Confidence is not a guarantee of correctness.")


if __name__ == "__main__":
    main()
