import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your pre-trained Random Forest model and scaler
rf_model = joblib.load('artifacts/rf_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

# Function to predict GPA
def predict_gpa(user_input):
    # Assuming the model and scaler have been loaded
    
    # Create DataFrame from user input
    user_df = pd.DataFrame([user_input])

    # Scale the input data using the same scaler
    user_scaled = scaler.transform(user_df)

    # Predict GPA
    predicted_gpa = rf_model.predict(user_scaled)
    
    return predicted_gpa[0]

# Streamlit UI
def app():
    st.title("GPA Prediction App")
    
    st.subheader("Enter your details:")

    # Define the options for study time as a dictionary
    study_time_dict = {
        1: '<2h',
        2: '2-5h',
        3: '5-10h',
        4: '>10h'
    }

    # User input fields
    G1 = st.slider("Enter first semester GPA (0.0-5.0)", 0.0, 5.0, 3.75)
    G2 = st.slider("Enter second semester GPA (0.0-5.0)", 0.0, 5.0, 3.75)
    studytime = st.selectbox("Enter weekly study time", [1, 2, 3, 4], format_func=lambda x: study_time_dict[x])
    failures = st.slider("Enter number of past failures (0-4)", 0, 4, 0)
    absences = st.slider("Enter number of absences", 0, 100, 5)
    freetime = st.slider("How much free time do you have after school?", 1, 5, 3)
    goout = st.slider("How often do you go out with friends?", 1, 5, 1)
    health = st.slider("Enter current health status (1-5)", 1, 5, 5)

    # Prepare input data
    user_input = {
        'G1': G1,
        'G2': G2,
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'freetime': freetime,
        'goout': goout,
        'health': health
    }

    if st.button("Predict GPA"):
        # Make the prediction
        predicted_gpa = predict_gpa(user_input)
        
        # Display the result
        st.subheader(f"Predicted Final GPA: {predicted_gpa:.2f}")
        if predicted_gpa >= 3.5:
            st.success("Great! You have a high GPA.")
        elif predicted_gpa >= 2.5:
            st.warning("You're doing okay, but you can improve!")
        else:
            st.error("It looks like you need to focus on improving your GPA.")

# Run the Streamlit app
if __name__ == "__main__":
    app()
