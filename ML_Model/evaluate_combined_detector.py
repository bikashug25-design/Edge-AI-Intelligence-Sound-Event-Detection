import os
import numpy as np
import librosa
import joblib
from collections import Counter

# ---------------- CONFIGURATION ----------------

BASE = "real_audio"

FILES = [
    "single_clap.wav",
    "clapping.wav",
    "ac_noise.wav",
    "fan_noise.wav",
    "room_ambience_noise.wav",
    "dragon-studio-glass-breaking-504033.wav",
    "freeeverythingxx-object-fall-soft-275694.wav",
    "alarm.wav",
    "salamisound-8990324-classic-alarm-clock-goes-off.wav",
]

SR = 16000

SILENCE_THRESHOLD = 0.0001

FRAME_SIZE = 800
HOP_SIZE = 400

MIN_RMS = 0.025
MIN_SCORE = 2.5
MAX_GAP_FRAMES = 8

# ---------------- LOAD MODEL ----------------

data = joblib.load(
    "model_output/mfcc_sound_classifier.joblib"
)

model = data["model"]


# ---------------- MFCC-SVM ----------------

def classify_windows(audio):

    predictions = []

    for start in range(0, len(audio) - SR + 1, SR):

        segment = audio[start:start + SR]

        rms = np.sqrt(np.mean(segment ** 2))

        if rms < SILENCE_THRESHOLD:
            prediction = "background"

        else:
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=SR,
                n_mfcc=20,
                n_fft=512,
                hop_length=256,
                n_mels=40
            )

            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1)
            ]).reshape(1, -1)

            prediction = model.predict(features)[0]

        predictions.append(prediction)

    return predictions


# ---------------- TRANSIENT DETECTOR ----------------

def detect_transients(audio):

    rms_values = []
    peaks = []

    for start in range(
        0,
        len(audio) - FRAME_SIZE + 1,
        HOP_SIZE
    ):

        frame = audio[start:start + FRAME_SIZE]

        rms_values.append(
            np.sqrt(np.mean(frame ** 2))
        )

        peaks.append(
            np.max(np.abs(frame))
        )

    rms_values = np.array(rms_values)
    peaks = np.array(peaks)

    candidates = []

    for i in range(2, len(rms_values) - 2):

        neighbours = np.concatenate([
            rms_values[i-2:i],
            rms_values[i+1:i+3]
        ])

        local_noise = np.median(neighbours)

        score = rms_values[i] / (local_noise + 1e-8)

        if rms_values[i] >= MIN_RMS and score >= MIN_SCORE:
            candidates.append(i)

    groups = []

    for index in candidates:

        if (
            not groups
            or index - groups[-1][-1] > MAX_GAP_FRAMES
        ):
            groups.append([index])

        else:
            groups[-1].append(index)

    events = []

    for group in groups:

        events.append({
            "start": group[0] * HOP_SIZE / SR,
            "end": (
                group[-1] * HOP_SIZE + FRAME_SIZE
            ) / SR,
            "peak": float(np.max(peaks[group])),
            "rms": float(np.max(rms_values[group]))
        })

    return events


# ---------------- EVALUATION ----------------

print("\nCOMBINED SOUND DETECTOR EVALUATION")
print("=" * 65)

for filename in FILES:

    path = os.path.join(BASE, filename)

    if not os.path.exists(path):
        print(f"\nMISSING FILE: {filename}")
        continue

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    svm_predictions = classify_windows(audio)

    transient_events = detect_transients(audio)

    counts = Counter(svm_predictions)

    print(f"\nFILE: {filename}")
    print("-" * 65)

    print("SVM predictions:", dict(counts))

    print(
        "Transient events:",
        len(transient_events)
    )

    print("SVM window timeline:")

    for i, prediction in enumerate(svm_predictions):
        print(f"  Second {i}: {prediction}")

    print("Transient event timeline:")

    for i, event in enumerate(transient_events, 1):

        print(
            f"  Event {i}: "
            f"{event['start']:.3f}s - "
            f"{event['end']:.3f}s | "
            f"Peak={event['peak']:.4f} | "
            f"RMS={event['rms']:.4f}"
        )

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")