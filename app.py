import streamlit as st
import numpy as np
import pickle  # Assuming you have a trained ML model saved as a pickle file
import pymysql
import os

# Load your trained model
# Replace 'model.pkl' with the path to your model file
model = pickle.load(open("prediction_app_files/xgboost_best.pkl", 'rb'))

# Display the image
st.image("prediction_app_files/personal loan.jpg", use_container_width=True)

# Streamlit App
st.title("Loan Approval Prediction")

# Input Features
st.header("Input Applicant Details")

# Dropdowns for Categorical Features
gender = st.selectbox("Gender", ("Male", "Female"))
married = st.selectbox("Married", ("Yes", "No"))
dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
education = st.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.selectbox("Self-Employed", ("Yes", "No"))
property_area = st.selectbox("Property Area", ("Rural", "Urban", "Semiurban"))

# Input Boxes for Numerical Features
applicant_income = st.number_input("Applicant Income", min_value=0, step=1000, value=5000, max_value = 50000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=1000.0, value=0.0, max_value = 50000.0)
loan_amount = st.number_input("Loan Amount (1 = 1 thousand)", min_value=0.0, step=1.0, value=100.0, max_value = 500.0)
loan_amount_term = st.number_input("Loan Amount Term (1 = 1 month)", min_value=0, step=1, value=360)
credit_history = st.selectbox("Credit History (1: Yes, 0: No)", (1, 0))

# Encode Categorical Features
gender_encoded = 1 if gender == "Male" else 0
married_encoded = 1 if married == "Yes" else 0
dependents_encoded = 3 if dependents == "3+" else int(dependents)
education_encoded = 0 if education == "Graduate" else 1
self_employed_encoded = 1 if self_employed == "Yes" else 0
property_area_encoded = {"Rural": 0, "Urban": 1, "Semiurban": 2}[property_area]

# Button to Predict
if st.button("Predict Loan Status"):
    # Prepare the input data for the model
    input_data = np.array([
        gender_encoded, married_encoded, dependents_encoded, education_encoded, self_employed_encoded,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area_encoded
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.success("The loan is likely to be approved!")
    else:
        st.error("The loan is likely to be rejected.")


# To run this app, save it as `app.py` and use the command:
# streamlit run app.py

# To run this app, save it as `app.py` and use the command:
# streamlit run app.py
