import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'ridge_diabetes' # The name of the model file without the .pkl extension

# Set page config for aesthetics
st.set_page_config(
    page_title="Diabetes Risk Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BETTER UI (Streamlit doesn't support Tailwind directly, so we use st.markdown) ---
st.markdown("""
<style>
    /* Gradient Background for the main page */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
    }
    
    /* Header Styling */
    h1 {
        color: #059669; /* Green-700 */
        text-align: center;
        margin-top: -15px;
        margin-bottom: 20px;
        font-weight: 800;
    }
    
    /* Main Card/Container Styling */
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid #d1d5db; /* Gray-300 */
    }

    /* Sidebar Styling */
    .css-1lcbmhc { /* Targeting the sidebar container */
        background-color: #f0fdf4; /* Lightest green background */
        border-right: 2px solid #a7f3d0; /* Emerald-200 border */
    }
    
    /* Input Labels */
    .stNumberInput label, .stSelectbox label {
        font-weight: 600;
        color: #10b981; /* Emerald-500 */
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 10px;
    }

    /* Predict Button Styling */
    div.stButton > button {
        background-color: #059669; /* Green-700 */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #047857; /* Darker Green */
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
    }
    
    /* Success/Prediction Output */
    .stSuccess {
        background-color: #ecfdf5; /* Lightest success background */
        color: #065f46; /* Dark success text */
        border-radius: 10px;
        padding: 15px;
        font-size: 1.1em;
        font-weight: 700;
        border: 2px solid #34d399; /* Success border */
    }
    
    /* Custom Indicator Badge */
    .indicator-badge {
        background-color: #d1fae5; /* Light green */
        color: #059669;
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 0.9em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
        border: 1px solid #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_diabetes_model(model_name):
    """Loads the PyCaret model, cached for efficiency."""
    try:
        model = load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_diabetes_model(MODEL_PATH)

# --- PREDICTION FUNCTION ---
def predict_risk(model, input_df):
    """Uses the loaded model to make a prediction."""
    # PyCaret's predict_model is designed to handle the data preparation internally
    predictions_df = predict_model(estimator=model, data=input_df)
    
    # Classification models usually return 'prediction_label' (0 or 1) and 'prediction_score'
    prediction_label = predictions_df.iloc[0]['prediction_label']
    
    # We use prediction_score for confidence, which is typically the probability of the predicted class
    confidence = predictions_df.iloc[0]['prediction_score'] 
    
    return prediction_label, confidence

# --- MAIN APPLICATION LOGIC ---
def run():
    
    # --- Sidebar for Context ---
    st.sidebar.markdown('<div class="indicator-badge">PyCaret Classification App</div>', unsafe_allow_html=True)
    st.sidebar.header("About This App")
    st.sidebar.info("This application uses a pre-trained Ridge Classifier model from PyCaret to assess a patient's risk of having diabetes based on key clinical metrics.")
    st.sidebar.header("Model Features")
    st.sidebar.markdown("""
    * **Pregnancies:** Number of times pregnant.
    * **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
    * **BloodPressure:** Diastolic blood pressure (mm Hg).
    * **SkinThickness:** Triceps skin fold thickness (mm).
    * **Insulin:** 2-Hour serum insulin $(\mu\text{U}/\text{ml})$.
    * **BMI:** Body mass index (weight in $\text{kg} / (\text{height in m})^2$).
    * **DiabetesPedigreeFunction:** A function that scores the likelihood of diabetes based on family history.
    * **Age:** Age in years.
    """)
    
    # --- Main Page Title ---
    st.title("Pima Indian Diabetes Risk Assessment")
    st.markdown("### Enter the patient's data below for risk prediction.")

    # --- Online Prediction Mode (Simplified for this app, no Batch option) ---
    st.markdown('<div class="indicator-badge">Online Prediction Mode</div>', unsafe_allow_html=True)
    
    # Use columns to lay out inputs neatly
    col1, col2, col3 = st.columns(3)
    
    # Input 1: Pregnancies (Integer)
    with col1:
        pregnancies = st.number_input(
            'Number of Pregnancies', 
            min_value=0, 
            max_value=17, 
            value=1, 
            step=1
        )
        
    # Input 2: Glucose (Integer)
    with col2:
        glucose = st.number_input(
            'Glucose (mg/dL)', 
            min_value=0, 
            max_value=200, 
            value=120,
            step=5
        )

    # Input 3: Blood Pressure (Integer)
    with col3:
        blood_pressure = st.number_input(
            'Blood Pressure (mm Hg)', 
            min_value=0, 
            max_value=122, 
            value=70, 
            step=2
        )

    # New row of inputs
    col4, col5, col6 = st.columns(3)

    # Input 4: Skin Thickness (Integer)
    with col4:
        skin_thickness = st.number_input(
            'Skin Thickness (mm)', 
            min_value=0, 
            max_value=99, 
            value=30, 
            step=1
        )

    # Input 5: Insulin (Integer)
    with col5:
        insulin = st.number_input(
            'Insulin ($\mu\text{U}/\text{ml}$)', 
            min_value=0, 
            max_value=846, 
            value=79, 
            step=10
        )
    
    # Input 6: BMI (Float)
    with col6:
        bmi = st.number_input(
            'BMI', 
            min_value=10.0, 
            max_value=67.1, 
            value=32.0, 
            step=0.1
        )

    # Final row of inputs
    col7, col8 = st.columns([1, 2])
    
    # Input 7: Diabetes Pedigree Function (Float)
    with col7:
        dpf = st.number_input(
            'Diabetes Pedigree Function', 
            min_value=0.078, 
            max_value=2.42, 
            value=0.5, 
            step=0.01
        )

    # Input 8: Age (Integer)
    with col8:
        age = st.number_input(
            'Age (Years)', 
            min_value=21, 
            max_value=81, 
            value=30, 
            step=1
        )
    
    # Spacer
    st.markdown("---")

    # --- Prediction Button and Logic ---
    if st.button('Assess Risk', key='predict_btn'):
        
        # 1. Prepare Data for Model
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        input_df = pd.DataFrame([input_data])
        
        # 2. Get Prediction
        try:
            prediction, confidence = predict_risk(model, input_df)
        except Exception as e:
            st.error(f"Prediction failed. Please check the inputs. Error: {e}")
            return # Exit function on error

        # 3. Format Output
        
        # Format confidence as a percentage (e.g., 0.85 -> 85.00%)
        confidence_percent = f"{confidence * 100:.2f}%" 

        if prediction == 1:
            risk_text = "High Risk (Predicted Positive)"
            color = "#ef4444" # Red-500
            st.markdown(f"""
                <div style="background-color: #fee2e2; border-left: 5px solid {color}; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4 style="color: {color}; margin: 0 0 5px 0;">🔴 {risk_text}</h4>
                    <p style="margin: 0;">The model predicts a **positive** result for diabetes with **{confidence_percent}** confidence.</p>
                </div>
            """, unsafe_allow_html=True)
            
        else: # prediction == 0
            risk_text = "Low Risk (Predicted Negative)"
            color = "#10b981" # Emerald-500
            st.markdown(f"""
                <div style="background-color: #ecfdf5; border-left: 5px solid {color}; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4 style="color: {color}; margin: 0 0 5px 0;">🟢 {risk_text}</h4>
                    <p style="margin: 0;">The model predicts a **negative** result for diabetes with **{confidence_percent}** confidence.</p>
                </div>
            """, unsafe_allow_html=True)

# Run the application
if __name__ == '__main__':
    run()