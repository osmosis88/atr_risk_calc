import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Function to load models
def load_model(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data["model"]

# Load all three models
gbm_sys_major = load_model('saved_gbm_sys_major_compl.pkl')
gbm_dvt = load_model('saved_gbm_dvt_compl.pkl')
gbm_ssi = load_model('saved_gbm_ssi_compl.pkl')

def get_user_input():
    """Collects user inputs and returns a processed NumPy array"""
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🧑 Age", 0, 90, 18)
        anesthesia = st.selectbox("💉 Type of Anesthesia", ["MAC", "Regional", "Spinal", "General"])
        smoking_status = st.selectbox("🚬 Smoker?", ["No", "Yes"])
        hypertension_status = st.selectbox("❤️ Hypertensive?", ["No", "Yes"])

    with col2:
        dialysis = st.selectbox("🩸 On Dialysis?", ["No", "Yes"])
        chronic_steroid = st.selectbox("💊 On Chronic Steroids?", ["No", "Yes"])
        asa = st.selectbox("🏥 ASA Level", ["ASA I", "ASA II", "ASA III", "ASA IV"])
        diabetes = st.selectbox("🩺 Diabetic?", ["No", "Yes"])

    # Prepare input array
    
    X = np.array([[age, anesthesia, smoking_status, hypertension_status, dialysis, chronic_steroid, asa, diabetes]], dtype=object)

    # Define encoding
    enc = OrdinalEncoder(categories=[
        ["Regional", "General", "Spinal", "MAC"],  # Anesthesia
        ["No", "Yes"],  # Smoking
        ["No", "Yes"],  # Hypertension
        ["No", "Yes"],  # Dialysis
        ["No", "Yes"],  # Chronic Steroid
        ["ASA I", "ASA II", "ASA III", "ASA IV"],  # ASA category
        ["No", "Yes"]  # Diabetes
    ])

    # Encode categorical variables
    X[:, 1:] = enc.fit_transform(X[:, 1:])
    return X.astype(float)

def predict_risk(X):
    """Returns risk probabilities for each complication"""
    return {
        "sys_major": gbm_sys_major.predict_proba(X)[:, -1],  # Last column
        "dvt": gbm_dvt.predict_proba(X)[:, -1],  # Last column
        "ssi": gbm_ssi.predict_proba(X)[:, -1]  # Last column
    }

def show_results(probabilities):
    """Displays prediction results in the UI"""
    st.markdown("---")
    st.write("### 🏥 Risk Assessment Results")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric(label="⚠️ Systemic Major Complication", value=f"{probabilities['sys_major'][0]:.2%}")
    with col4:
        st.metric(label="🩸 DVT Risk", value=f"{probabilities['dvt'][0]:.2%}")
    with col5:
        st.metric(label="🦠 Surgical Site Infection", value=f"{probabilities['ssi'][0]:.2%}")

    st.markdown("---")
    st.success("✅ Risk assessment completed. Consult a healthcare provider for further evaluation.")

def run_app():
    """Runs the Streamlit app"""
    st.title("🔍 Achilles Tendon Repair - Complication Risk Calculator")
    st.write("### Enter patient details to estimate the risk of complications:")

    # Get user input
    X = get_user_input()

    # Calculate Risk
    if st.button("🧮 Calculate Risk"):
        probabilities = predict_risk(X)
        show_results(probabilities)

# Ensure script runs only when executed directly
if __name__ == "__main__":
    run_app()
