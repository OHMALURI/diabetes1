import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# --- Load Model ---
model = load_model("ridge_diabetes")   # your saved model name

st.title("Diabetes Prediction App")

st.write("Enter the patient details below to check diabetes risk.")

# --- INPUT FORM ---
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)

    submit_btn = st.form_submit_button("Predict")

# --- PREDICTION ---
if submit_btn:
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    result = predict_model(model, data=input_df)
    
    pred = result["prediction_label"][0]
    score = result["prediction_score"][0]

    if pred == 1:
        st.error(f"High Risk of Diabetes (Confidence: {score:.2f})")
    else:
        st.success(f"Low Risk of Diabetes (Confidence: {score:.2f})")
