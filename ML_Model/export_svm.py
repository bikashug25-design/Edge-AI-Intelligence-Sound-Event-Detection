
import joblib
import numpy as np
from pathlib import Path

data = joblib.load(
    "model_output/mfcc_sound_classifier.joblib"
)

pipeline = data["model"]
scaler = pipeline.named_steps["scaler"]
svm = pipeline.named_steps["classifier"]

out = Path("esp32_model")
out.mkdir(exist_ok=True)

def write_array(f, ctype, name, array):
    array = np.asarray(array)
    flat = array.ravel()
    f.write(f"const {ctype} {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        values = ", ".join(
            f"{float(x):.9g}f" for x in flat[i:i+6]
        )
        f.write("  " + values + ",\n")
    f.write("};\n\n")

with open(out / "svm_model.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Exported from the trained scikit-learn pipeline\n")
    f.write("constexpr int N_FEATURES = 40;\n")
    f.write(f"constexpr int N_SUPPORT = {svm.support_vectors_.shape[0]};\n")
    f.write(f"constexpr int N_CLASSES = {len(svm.classes_)};\n")
    f.write(f"constexpr float SVM_GAMMA = {svm._gamma:.9g}f;\n\n")

    write_array(f, "float", "scaler_mean", scaler.mean_)
    write_array(f, "float", "scaler_scale", scaler.scale_)
    write_array(f, "float", "support_vectors", svm.support_vectors_)
    write_array(f, "float", "dual_coef", svm.dual_coef_)
    write_array(f, "float", "intercept", svm.intercept_)

    f.write("const char* class_names[N_CLASSES] = ")
    f.write('{"alarm", "background", "impact"};\n')

print("Export complete.")
print("Header:", out / "svm_model.h")
print("Support vector shape:", svm.support_vectors_.shape)
print("Dual coefficient shape:", svm.dual_coef_.shape)
print("Intercept shape:", svm.intercept_.shape)
print("Kernel:", svm.kernel)
print("Decision function shape:", svm.decision_function_shape)
