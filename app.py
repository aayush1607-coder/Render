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
MODEL_PATH = "C:\Users\aayus\OneDrive\Desktop\sih lmaoo\model\juice_pila_do (1).h5"
model = load_model(MODEL_PATH)

# Load the LabelEncoder
LABEL_ENCODER_PATH = "C:\Users\aayus\OneDrive\Desktop\sih lmaoo\model\label_encoder.pkl"
with open(LABEL_ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

# Route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
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

if __name__ == "__main__":
    app.run(debug=True)