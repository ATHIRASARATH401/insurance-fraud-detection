import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('fraud_detection_model (7).pkl')

# Set up the Streamlit app
st.title("Vehicle Insurance Claim Fraud Detection")

# Create input fields for the features
age = st.number_input("Age", min_value=18, max_value=100)
week_of_month = st.number_input("Week Of Month", min_value=1, max_value=5)
week_of_month_claimed = st.number_input("Week Of Month Claimed", min_value=1, max_value=5)
deductible = st.number_input("Deductible", min_value=0)
driver_rating = st.number_input("Driver Rating", min_value=1, max_value=5)
year = st.number_input("Year", min_value=1994, max_value=2024)

# Create a DataFrame to match model input
input_data = {
    'Age': 30,
    'WeekOfMonth': 4,
    'WeekOfMonthClaimed': 3,
    'Deductible': 500,
    'DriverRating': 4,
    'Year': 1996,
    # Add more input fields based on your model's training data
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure that the input_df has the same columns as the training data
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)  # Fill missing columns with 0

# Prediction button
if st.button("Predict Fraud"):
    # Make the prediction
    prediction = model.predict(input_df)

    # Display the result
    if prediction[0] == 1:
        st.error("Fraud Detected!")  # Use st.error for emphasis
    else:
        st.success("No Fraud Detected!")  # Use st.success for positive message


