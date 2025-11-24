from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from load import predict_disease_with_prob

app = Flask(__name__)
CORS(app)

def preprocess_image(img):
    from tensorflow.keras.preprocessing.image import img_to_array
    img = img.resize((128,128))
    img_arr = img_to_array(img) / 255.0
    return np.expand_dims(img_arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img = Image.open(request.files["file"].stream).convert("RGB")
    processed = preprocess_image(img)

    results, top_class, top_prob, infection_level, suggestion, advice, healing_score = (
        predict_disease_with_prob(processed)
    )

    return jsonify({
        "top_class": top_class,
        "probability": f"{top_prob}%",
        "infection_level": infection_level,
        "suggestion": suggestion,
        "advice": advice,
        "healing_score": healing_score,
        "all_results": results
    })

if __name__ == "__main__":
    app.run(debug=True)
