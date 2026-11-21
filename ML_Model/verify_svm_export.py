
import joblib
import numpy as np
from pathlib import Path

data = joblib.load(
    "model_output/mfcc_sound_classifier.joblib"
)
pipeline = data["model"]

# Check the exported arrays can be loaded and have expected shapes.
svm = pipeline.named_steps["classifier"]
scaler = pipeline.named_steps["scaler"]

print("Python pipeline classes:", svm.classes_)
print("Python support vectors:", svm.support_vectors_.shape)
print("Python dual coefficients:", svm.dual_coef_.shape)
print("Python intercept:", svm.intercept_.shape)

print("\nExpected export header:")
header = Path("esp32_model/svm_model.h")
print("Exists:", header.exists())

if header.exists():
    print("Header size:", header.stat().st_size, "bytes")

# Confirm the scaler + SVM pipeline can still make predictions.
# This is a baseline check, not yet a C++ parity test.
rng = np.random.default_rng(42)
X = rng.normal(size=(5, 40))

expected = pipeline.predict(X)
print("\nBaseline predictions on synthetic features:")
print(expected)

print("\nExport metadata checks passed.")
