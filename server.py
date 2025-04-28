from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model and scaler
try:
    rf_model = joblib.load('artifacts/rf_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# Function to predict GPA
def predict_gpa(user_input):
    try:
        # Create DataFrame from user input
        user_df = pd.DataFrame([user_input])
        
        # Print the input data for debugging
        print(f"Input data: {user_df}")
        
        # Scale the input data using the same scaler
        user_scaled = scaler.transform(user_df)
        
        # Predict GPA
        predicted_gpa = rf_model.predict(user_scaled)
        
        return predicted_gpa[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print the raw request data for debugging
        print(f"Request data: {request.data}")
        
        data = request.json
        print(f"Parsed JSON data: {data}")
        
        # Ensure all required fields are present
        required_fields = ['G1', 'G2', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'health']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        predicted_gpa = predict_gpa(data)
        return jsonify({'predicted_gpa': float(predicted_gpa)})
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)