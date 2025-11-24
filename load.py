import os
import json
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = 'leaf_disease_model.h5'
CLASS_INDEX_PATH = 'class_indices.json'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not os.path.exists(CLASS_INDEX_PATH):
    raise FileNotFoundError(f"❌ Class index not found: {CLASS_INDEX_PATH}")

model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

class_indices = {v: k for k, v in class_indices.items()}

# DISEASE WISE SUGGESTIONS
disease_suggestions = {
    "Apple___Black_rot": "Use copper fungicide and prune infected black spots.",
    "Apple___healthy": "Leaf is healthy. Maintain normal watering and sunlight.",
    "Tomato___Leaf_Mold": "Increase ventilation. Apply Neem oil or copper spray.",
    "Potato___Late_blight": "Remove infected leaves. Apply Metalaxyl fungicide.",
}

def predict_disease_with_prob(img_array):
    preds = model.predict(img_array)[0]

    results = [(class_indices[i], round(float(preds[i]) * 100, 2))
               for i in range(len(preds))]
    results.sort(key=lambda x: x[1], reverse=True)

    top_class, top_prob = results[0]

    # INFECTION PERCENT LOGIC
    if top_prob < 30:
        infection_level = "🟢 Very Low / Healthy"
        infection_advice = "No infection detected. Keep regular care."
        healing_score = 90
    elif 30 <= top_prob < 60:
        infection_level = "🟡 Early Stage"
        infection_advice = "Use mild organic spray like Neem oil."
        healing_score = 70
    elif 60 <= top_prob < 80:
        infection_level = "🟠 Moderate Infection"
        infection_advice = "Apply fungicide and improve ventilation."
        healing_score = 40
    else:
        infection_level = "🔴 Severe Infection"
        infection_advice = "Prune infected parts and apply copper fungicide."
        healing_score = 20

    disease_specific = disease_suggestions.get(
        top_class, "No disease-specific suggestion available."
    )

    combined_advice = f"{disease_specific} {infection_advice}"

    return results, top_class, top_prob, infection_level, disease_specific, combined_advice, healing_score
