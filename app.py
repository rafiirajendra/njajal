from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os

# Pakai TFLite runtime yang ringan
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)
CORS(app)

# Path model TFLite (pastikan file .tflite ada di sini)
MODEL_PATH = os.path.join("saved_models", "ktp_fraud_cnn_tampering_v1.tflite")

# --- Load TFLite model ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Asumsi ukuran input model 224 x 224 x 3
IMG_HEIGHT = 224
IMG_WIDTH = 224


def preprocess_image(image_bytes):
    """
    Ubah file image mentah jadi array siap masuk model.
    PENTING: Sesuaikan dengan preprocessing saat training.
    Di sini kita TIDAK membagi 255 karena diasumsikan
    model sudah punya layer Rescaling(1./255).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Pastikan tipe data cocok dengan yang diminta model
    input_dtype = input_details[0]["dtype"]
    img_array = img_array.astype(input_dtype)

    return img_array


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediksi KTP fraud detection.
    
    Asumsi model:
      - Output sigmoid = P(VALID)
      - P(FRAUD) = 1 - P(VALID)
    """
    # Terima file dengan key apapun (ambil file pertama yang dikirim)
    if not request.files:
        return jsonify({"error": "No file uploaded."}), 400

    # Ambil file pertama yang ada (key bebas)
    file = next(iter(request.files.values()))
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        input_tensor = preprocess_image(img_bytes)

        # Set input ke interpreter
        interpreter.set_tensor(input_details[0]["index"], input_tensor)

        # Jalankan inferensi
        interpreter.invoke()

        # Ambil output
        output_data = interpreter.get_tensor(output_details[0]["index"])
        # Asumsi output shape: (1, 1) sigmoid
        p_valid = float(output_data[0][0])
        p_fraud = 1.0 - p_valid

        thresh_valid = 0.8
        label = "VALID" if p_valid >= thresh_valid else "FRAUD"

        result = {
            "label": label,
            "p_valid": p_valid,
            "p_fraud": p_fraud,
            "threshold": thresh_valid
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)