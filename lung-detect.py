from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = "lung_cancer_v2_fixed.keras"
model = keras.models.load_model(MODEL_PATH)

class_labels = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

CONFIDENCE_THRESHOLD = 0.75


def preprocess_image(file_storage):
    """Match EXACT training preprocessing."""
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = preprocess_input(img_array)   # <-- THE IMPORTANT FIX
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_batch = preprocess_image(file)
        preds = model.predict(img_batch)[0]

        pred_idx = np.argmax(preds)
        pred_prob = float(preds[pred_idx])
        pred_class = class_labels[pred_idx]

        # Build probability dict
        all_probs = {
            class_labels[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(class_labels))
        }

        # Threshold logic
        if pred_prob < CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": True,
                "is_uncertain": True,
                "predicted_class": "uncertain",
                "confidence": round(pred_prob * 100, 2),
                "all_probabilities": all_probs
            })

        return jsonify({
            "success": True,
            "is_uncertain": False,
            "predicted_class": pred_class,
            "confidence": round(pred_prob * 100, 2),
            "all_probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
