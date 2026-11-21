
import os
import joblib
import librosa
import numpy as np
from collections import deque, Counter

# ---------------- CONFIGURATION ----------------

SR = 16000
SILENCE_THRESHOLD = 0.0001

FRAME_SIZE = 800       # 50 ms
HOP_SIZE = 400         # 25 ms

MIN_RMS = 0.025
MIN_SCORE = 2.5
MAX_GAP_FRAMES = 8

ALARM_WINDOW = 5
ALARM_REQUIRED = 3

AUDIO_FILES = [
    "single_clap.wav",
    "clapping.wav",
    "ac_noise.wav",
    "fan_noise.wav",
    "room_ambience_noise.wav",
    "dragon-studio-glass-breaking-504033.wav",
    "freeeverythingxx-object-fall-soft-275694.wav",
    "alarm.wav",
    "salamisound-8990324-classic-alarm-clock-goes-off.wav"
]

# ---------------- LOAD MODEL ----------------

data = joblib.load("model_output/mfcc_sound_classifier.joblib")
model = data["model"]


def classify_windows(audio):
    predictions = []

    for start in range(0, len(audio) - SR + 1, SR):
        segment = audio[start:start + SR]

        rms = np.sqrt(np.mean(segment ** 2))

        if rms < SILENCE_THRESHOLD:
            predictions.append("background")
            continue

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

        predictions.append(str(model.predict(features)[0]))

    return predictions


def detect_transients(audio):
    frame_rms = []
    frame_peak = []

    for start in range(0, len(audio) - FRAME_SIZE + 1, HOP_SIZE):
        frame = audio[start:start + FRAME_SIZE]

        frame_rms.append(np.sqrt(np.mean(frame ** 2)))
        frame_peak.append(np.max(np.abs(frame)))

    frame_rms = np.array(frame_rms)
    frame_peak = np.array(frame_peak)

    candidates = []

    for i in range(2, len(frame_rms) - 2):
        neighbours = np.concatenate([
            frame_rms[i-2:i],
            frame_rms[i+1:i+3]
        ])

        local_noise = np.median(neighbours)
        score = frame_rms[i] / (local_noise + 1e-8)

        if frame_rms[i] >= MIN_RMS and score >= MIN_SCORE:
            candidates.append(i)

    # Group nearby candidate frames
    groups = []

    for index in candidates:
        if not groups or index - groups[-1][-1] > MAX_GAP_FRAMES:
            groups.append([index])
        else:
            groups[-1].append(index)

    events = []

    for group in groups:
        first = group[0]
        last = group[-1]

        events.append({
            "start": first * HOP_SIZE / SR,
            "end": (last * HOP_SIZE + FRAME_SIZE) / SR,
            "peak": float(np.max(frame_peak[group])),
            "rms": float(np.max(frame_rms[group]))
        })

    return events


# ---------------- COMBINED DECISION ----------------

for filename in AUDIO_FILES:

    path = os.path.join("real_audio", filename)

    if not os.path.exists(path):
        print(f"\nSKIPPED: {filename} (file not found)")
        continue

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    audio, _ = librosa.load(path, sr=SR, mono=True)

    predictions = classify_windows(audio)
    events = detect_transients(audio)

    # Alarm: 3 alarm predictions in the latest 5 windows
    alarm_history = deque(maxlen=ALARM_WINDOW)
    alarm_triggered = False

    for second, prediction in enumerate(predictions):
        alarm_history.append(prediction == "alarm")

        if sum(alarm_history) >= ALARM_REQUIRED:
            if not alarm_triggered:
                print(
                    f"ALARM ALERT first triggered "
                    f"around second {second + 1}"
                )
                alarm_triggered = True

    if not alarm_triggered:
        print("ALARM ALERT: Not triggered")

    # Impact: transient confirmation only
    if events:
        print(f"IMPACT ALERT: {len(events)} transient event(s)")

        for i, event in enumerate(events, 1):
            print(
                f"  Event {i}: {event['start']:.3f}s - "
                f"{event['end']:.3f}s"
            )
    else:
        print("IMPACT ALERT: Not triggered")

    print("SVM predictions:", dict(Counter(predictions)))

print("\n" + "=" * 60)
print("COMBINED DECISION TEST COMPLETE")
