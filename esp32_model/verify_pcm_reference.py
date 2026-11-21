
import joblib
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "model_output" / "mfcc_sound_classifier.joblib"
AUDIO_PATH = PROJECT_DIR / "esp32_model" / "reference_audio.wav"
FEATURE_PATH = PROJECT_DIR / "esp32_model" / "reference_features.csv"
PREDICTION_PATH = PROJECT_DIR / "esp32_model" / "reference_expected_prediction.txt"

# Load the trained model
bundle = joblib.load(MODEL_PATH)

SR = bundle["SR"]
N_MFCC = bundle["N_MFCC"]
N_FFT = bundle["N_FFT"]
HOP_LENGTH = bundle["HOP_LENGTH"]
N_MELS = bundle["N_MELS"]
model = bundle["model"]

# Read the exact PCM16 WAV used by C++
audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")

print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)

# Calculate MFCC using the exact saved WAV samples
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=SR,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

features = np.concatenate([
    np.mean(mfcc, axis=1),
    np.std(mfcc, axis=1)
]).astype(np.float64)

prediction = model.predict(features.reshape(1, -1))[0]

# Replace the old reference features
np.savetxt(FEATURE_PATH, features.reshape(1, -1), delimiter=",", fmt="%.17g")

PREDICTION_PATH.write_text(str(prediction), encoding="utf-8")

print("\nPCM reference regenerated successfully!")
print("MFCC shape:", mfcc.shape)
print("Feature count:", len(features))
print("New Python prediction:", prediction)
print("Saved:", FEATURE_PATH)
