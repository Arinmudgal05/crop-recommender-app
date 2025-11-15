# app.py - Step1: UI skeleton with placeholder prediction
import streamlit as st

st.set_page_config(page_title="Crop Recommender (Demo)", layout="centered")

st.title("🌾 Crop Recommendation — Demo")
st.write("Enter field parameters and click **Predict**. (This is a UI test; model integration comes next.)")

with st.form(key='input_form'):
    N = st.number_input("Enter Nitrogen (N)", min_value=0, max_value=1000, value=90)
    P = st.number_input("Enter Phosphorus (P)", min_value=0, max_value=1000, value=42)
    K = st.number_input("Enter Potassium (K)", min_value=0, max_value=1000, value=43)
    temp = st.number_input("Enter Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, format="%.1f")
    hum = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
    ph = st.number_input("Enter Soil pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
    rain = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=5000.0, value=200.0, format="%.1f")
    submit = st.form_submit_button("Predict")

def placeholder_predict(N,P,K,temp,hum,ph,rain):
    if N>80 and P>30:
        return "Rice", ["Urea", "DAP", "MOP", "Zinc Sulphate"]
    if ph < 6:
        return "Maize", ["DAP", "MOP"]
    return "Wheat", ["NPK", "DAP"]

if submit:
    crop, ferts = placeholder_predict(N,P,K,temp,hum,ph,rain)
    st.success(f"🌿 Predicted Crop: **{crop.upper()}**")
    st.info("Recommended Fertilizer: " + ", ".join(ferts))
    st.caption("This is a demo placeholder. Real model will be hooked in next step.")
