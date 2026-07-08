import os
import io
import json
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "cat_dog_model_best.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "class_names.json")
IMG_SIZE = (64, 64)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

model = None
class_names = ["cats", "dogs"]

def load_model():
    global model, class_names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/", methods=["GET"])
def index():
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded yet. Please try again shortly."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:
        file_bytes = file.read()
        arr = preprocess_image(file_bytes)
        pred = model.predict(arr, verbose=0)[0][0]

        # class_names is alphabetical from image_dataset_from_directory: ['cats', 'dogs']
        dog_prob = float(pred)
        cat_prob = 1.0 - dog_prob

        label = "Dog" if dog_prob >= 0.5 else "Cat"
        confidence = dog_prob if dog_prob >= 0.5 else cat_prob

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2),
            "dog_probability": round(dog_prob * 100, 2),
            "cat_probability": round(cat_prob * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"model_ready": model is not None})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
