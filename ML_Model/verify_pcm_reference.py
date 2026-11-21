
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

# Load trained model
bundle = joblib.load(MODEL_PATH)

SR = bundle["sample_rate"]
N_MFCC = bundle["n_mfcc"]
N_FFT = bundle["n_fft"]
HOP_LENGTH = bundle["hop_length"]
N_MELS = bundle["n_mels"]
model = bundle["model"]

# Read exact PCM16 WAV used by C++
audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")

print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Expected sample rate:", SR)

if sample_rate != SR:
    raise ValueError("Sample rate mismatch!")

if audio.ndim != 1 or len(audio) != bundle["window_size"]:
    raise ValueError("Audio must be a mono window of the expected length!")

# Calculate Python reference MFCC
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

# Save reference features and MFCC matrix
np.savetxt(
    FEATURE_PATH,
    features.reshape(1, -1),
    delimiter=",",
    fmt="%.17g"
)

np.savetxt(
    PROJECT_DIR / "esp32_model" / "reference_mfcc.csv",
    mfcc,
    delimiter=",",
    fmt="%.17g"
)

PREDICTION_PATH.write_text(str(prediction), encoding="utf-8")

print("\nPCM reference regenerated successfully!")
print("MFCC matrix shape:", mfcc.shape)
print("Feature count:", len(features))
print("New Python prediction:", prediction)
print("Reference files updated successfully.")
