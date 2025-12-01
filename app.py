from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    "..", "Fraud_Detectio", "saved_models", "ktp_fraud_cnn_tampering_v1.tflite"
)
model = tf.keras.models.load_model(MODEL_PATH)

IMG_HEIGHT = 224
IMG_WIDTH = 224 


def preprocess_image(image_bytes):
    """
    Ubah file image mentah jadi tensor siap masuk model.
    
    PENTING: TIDAK dibagi 255 karena model sudah punya layer Rescaling(1./255)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
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
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        input_tensor = preprocess_image(img_bytes)

        # Prediksi: output sigmoid -> P(VALID)
        preds = model.predict(input_tensor)
        
        # Model output adalah P(VALID)
        p_valid = float(preds[0][0])
        p_fraud = 1.0 - p_valid
        
        # Threshold untuk keputusan
        thresh_valid = 0.5
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
