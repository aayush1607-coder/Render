from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import pickle
from preprocess import preprocess_image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model/you_are_just_a_chill_ml_model.h5"
model = load_model(MODEL_PATH)

# Load the LabelEncoder
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
with open(LABEL_ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Route for the homepage. Handles image upload and prediction.
    """
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Preprocess the image and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        os.remove(file_path)
        
        # Get the class index and decode it using LabelEncoder
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = np.max(prediction)
        
        result = {
            "class": predicted_class_label,
            "confidence": float(confidence),
        }
        return jsonify(result)
    
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for making predictions.
    """
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # Preprocess the image and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        
        # Get the class index and decode it using LabelEncoder
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = np.max(prediction)
        
        result = {
            "class": predicted_class_label,
            "confidence": float(confidence),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Remove the file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)
