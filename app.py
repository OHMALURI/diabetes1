import streamlit as st
import pickle
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

def load_model_wrapper():
    load_error = None
    try:
        with open('ridge_diabetes.pkl', 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        model = None
        load_error = e
        st.sidebar.error("Model failed to load. Please ensure 'ridge_diabetes.pkl' is in the same directory.")
        st.error(f"Model load error: {e}")
    return model

def predict(model, input_df):
    prediction = model.predict(input_df)
    return prediction[0]

def run():
    st.title("Diabetes Prediction App")
    st.sidebar.header("Input Features")
    
    try:
        image = Image.open('logo.png')
        st.image(image)
    except:
        pass
    
    st.sidebar.info("This app is created to predict diabetes based on patient data")
    
    try:
        image_hospital = Image.open('hospital.jpeg')
        st.sidebar.image(image_hospital)
    except:
        pass
    
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    
    if add_selectbox == "Online":
        st.sidebar.subheader("Patient Data")
        
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        output = ""
        
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
                st.error("Model is not loaded. See the sidebar for details.")
            else:
                output = predict(model=model, input_df=input_df)
                if output >= 0.5:
                    st.success(f'The patient is likely to have diabetes. Prediction score: {output:.4f}')
                else:
                    st.success(f'The patient is not likely to have diabetes. Prediction score: {output:.4f}')
    
    else:
        st.subheader("Batch Prediction")
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        
        if file_upload is not None:
            input_df = pd.read_csv(file_upload)
            model = load_model_wrapper()
            
            if model is None:
                st.error("Model is not loaded. See the sidebar for details.")
            else:
                predictions = model.predict(input_df)
                input_df['Prediction'] = predictions
                st.success("Predictions:")
                st.write(input_df)

if __name__ == '__main__':
    run()