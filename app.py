from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
from preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Constants
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "model/juice_pila_do.h5"
model = load_model(MODEL_PATH)

# Load the LabelEncoder
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
with open(LABEL_ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            if "image" not in request.files:
                return "No file uploaded!", 400
            
            file = request.files["image"]
            if file.filename == "":
                return "No selected file", 400
            
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Preprocess the image
            try:
                image = preprocess_image(file_path)
            except Exception as e:
                app.logger.error(f"Error in preprocessing: {e}")
                return "Error in preprocessing the image!", 500
            
            # Predict using the model
            try:
                prediction = model.predict(image)
            except Exception as e:
                app.logger.error(f"Error in prediction: {e}")
                return "Error in making a prediction!", 500
            
            # Decode the prediction
            try:
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
                confidence = np.max(prediction)
            except Exception as e:
                app.logger.error(f"Error in decoding prediction: {e}")
                return "Error in decoding the prediction!", 500
            
            # Delete the file after processing
            os.remove(file_path)
            
            # Return the result
            result = {
                "class": predicted_class_label,
                "confidence": float(confidence),
            }
            return jsonify(result)
        
        return render_template("index.html")
    
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred!", 500

if __name__ == "__main__":
    app.run(debug=True)
