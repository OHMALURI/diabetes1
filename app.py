import streamlint as st
import pandas as pd
from PIL import Image
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# ---------------------------------------------------
# Load PyCaret Model
# ---------------------------------------------------
def load_model_wrapper():
    load_error = None
    try:
        model = load_model("ridge_diabetes")   # loads ridge_diabetes.pkl
    except Exception as e:
        model = None
        load_error = e
        st.sidebar.error("Model failed to load. Ensure 'ridge_diabetes.pkl' exists in this folder.")
        st.error(f"Model load error: {e}")
    return model

# ---------------------------------------------------
# Predict with PyCaret
# ---------------------------------------------------
def predict(model, input_df):
    result = predict_model(model, data=input_df)
    prediction = int(result["prediction_label"][0])
    score = float(result["prediction_score"][0])
    return prediction, score

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
def run():
    st.title("Diabetes Prediction App (PyCaret Version)")
    st.sidebar.header("Input Features")

    # Logo (optional)
    try:
        image = Image.open('logo.png')
        st.image(image)
    except:
        pass

    st.sidebar.info("This app predicts diabetes using a Ridge Classifier ML model trained with PyCaret.")

    # Sidebar image (optional)
    try:
        image_hospital = Image.open('hospital.jpeg')
        st.sidebar.image(image_hospital)
    except:
        pass

    # Mode selection
    mode = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

    # ---------------------------------------------------
    # 🟦 ONLINE PREDICTION
    # ---------------------------------------------------
    if mode == "Online":
        st.sidebar.subheader("Patient Data")

        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

        # Prepare input
        input_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        input_df = pd.DataFrame([input_dict])

        model = load_model_wrapper()

        if st.button("Predict"):
            if model is None:
                st.error("Model not loaded. Check sidebar.")
            else:
                prediction, score = predict(model, input_df)

                if prediction == 1:
                    st.error(f"⚠️ Likely DIABETIC — Probability: {round(score*100,2)}%")
                else:
                    st.success(f"✅ Not likely diabetic — Probability: {round(score*100,2)}%")

    # ---------------------------------------------------
    # 🟩 BATCH PREDICTION
    # ---------------------------------------------------
    else:
        st.subheader("Batch Prediction")
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])

        if file_upload is not None:
            input_df = pd.read_csv(file_upload)
            model = load_model_wrapper()

            if model is None:
                st.error("Model not loaded.")
            else:
                results = predict_model(model, data=input_df)
                st.success("Batch Predictions:")
                st.write(results)

# Run app
if __name__ == '__main__':
    run()
