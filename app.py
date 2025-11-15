# app.py - Animated, attractive demo UI for Crop Recommendation
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import time
import random
import math

# ---------- Configuration ----------
st.set_page_config(page_title="Crop Recommender — Demo", layout="wide", initial_sidebar_state="collapsed")

# ---------- Helper functions ----------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

def placeholder_predict(N,P,K,temp,hum,ph,rain):
    # quick deterministic-ish placeholder
    score = (N*0.6 + P*0.2 + K*0.2) + (rain/10) - abs(6.5-ph)*20
    if score > 200:
        return "Rice", ["Urea", "DAP", "MOP", "Zinc Sulphate"], 0.92
    if ph < 6:
        return "Maize", ["DAP", "MOP"], 0.78
    if hum > 85 and temp > 25:
        return "Papaya", ["NPK", "Potash"], 0.81
    return "Wheat", ["NPK", "DAP"], 0.65

def render_fertilizer_pills(ferts):
    pills_html = ""
    for f in ferts:
        pills_html += f"<span class='pill'>{f}</span> "
    return pills_html

# ---------- Load animation ----------
# Lottie animation URLs (free public)
anim_searching = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json")  # searching
anim_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")  # success

# ---------- Page styling ----------
st.markdown(
    """
    <style>
      /* background card */
      .main-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:18px; box-shadow: 0 6px 30px rgba(2,6,23,0.5); }
      h1 { font-family: 'Segoe UI', Roboto, sans-serif; }
      .result-title { font-size:28px; font-weight:700; color:#065f46; margin-bottom:6px; }
      .result-card { background: linear-gradient(90deg,#064e3b,#065f46); color: white; border-radius:12px; padding:18px; }
      .sub-card { background: linear-gradient(90deg,#0f172a,#0b1220); color: #cbd5e1; border-radius:10px; padding:12px; }
      .pill { display:inline-block; background: rgba(255,255,255,0.06); color: #e6f4ea; padding:7px 12px; border-radius:999px; margin-right:8px; font-weight:600; box-shadow: 0 2px 8px rgba(2,6,23,0.5); }
      .confidence { font-size:18px; font-weight:700; color:#fff; opacity:0.95; }
      .centered { display:flex; align-items:center; justify-content:center; }
      .big-crop { font-size:40px; font-weight:800; letter-spacing:1px; color: #eafff0; text-shadow: 0 6px 18px rgba(0,0,0,0.6); }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- App Layout ----------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
left, right = st.columns([1,1])

with left:
    st.markdown("<h1>🌾 Crop Recommendation — Demo</h1>", unsafe_allow_html=True)
    st.markdown("Enter field parameters and press **Predict**. Enjoy the animation ✨", unsafe_allow_html=True)

    with st.form(key='input_form'):
        N = st.number_input("Enter Nitrogen (N)", min_value=0, max_value=1000, value=90, step=1)
        P = st.number_input("Enter Phosphorus (P)", min_value=0, max_value=1000, value=42, step=1)
        K = st.number_input("Enter Potassium (K)", min_value=0, max_value=1000, value=43, step=1)
        temp = st.number_input("Enter Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, format="%.1f")
        hum = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
        ph = st.number_input("Enter Soil pH", min_value=0.0, max_value=14.0, value=6.50, format="%.2f")
        rain = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=10000.0, value=200.0, format="%.1f")
        submit = st.form_submit_button("Predict", help="Click to predict and see animation")

with right:
    # show a friendly illustration / instructions area
    st.markdown("<div class='sub-card'><b>Tips:</b><br>- Use realistic nutrient values.<br>- pH around 6-7 is good for many crops.<br>- This demo shows animated UI; model integration follows.</div>", unsafe_allow_html=True)
    if anim_searching:
        st_lottie(anim_searching, height=240, key="hero_anim")

st.markdown("</div>", unsafe_allow_html=True)



# ---------- Prediction flow with animation (fixed final) ----------
if submit:
    # placeholders for status/progress
    status_text = st.empty()
    prog = st.empty()

    # show searching animation (call st_lottie directly)
    if anim_searching:
        st_lottie(anim_searching, height=200, key="anim_search")

    status_text.markdown("<b>Analysing soil & weather data...</b>", unsafe_allow_html=True)

    # progress bar animation
    p_bar = prog.progress(0)
    for i in range(0, 101, 7):
        p_bar.progress(min(i, 100))
        time.sleep(0.06 + random.random() * 0.03)
    p_bar.progress(100)
    time.sleep(0.15)

    # compute prediction (placeholder function for now)
    crop, ferts, conf = placeholder_predict(N, P, K, temp, hum, ph, rain)

    # show success animation (below searching animation)
    if anim_success:
        st_lottie(anim_success, height=180, key="anim_success")

    # show result with a nice large card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='centered'><div class='big-crop'>{crop.upper()}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>Confidence: {int(conf*100)}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # fertilizer pills (styled)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-card'><b>Recommended Fertilizers:</b><br>" + render_fertilizer_pills(ferts) + "</div>", unsafe_allow_html=True)

    # celebration
    st.balloons()
    st.snow()

    # small shareable summary box (copyable)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.code(f"Predicted Crop: {crop.upper()}  |  Confidence: {int(conf*100)}%  |  Fertilizers: {', '.join(ferts)}", language='text')



    # 2) Compute prediction (placeholder function for now)
    crop, ferts, conf = placeholder_predict(N,P,K,temp,hum,ph,rain)

    # 3) Replace animation with success animation and show big result card
    anim_placeholder.empty()
    if anim_success:
        st_lottie(anim_success, height=180, key="succ")

    # show result with a nice large card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='centered'><div class='big-crop'>{crop.upper()}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>Confidence: {int(conf*100)}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # fertilizer pills (styled)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-card'><b>Recommended Fertilizers:</b><br>" + render_fertilizer_pills(ferts) + "</div>", unsafe_allow_html=True)

    # celebration
    st.balloons()
    st.snow()

    # small shareable summary box (copyable)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.code(f"Predicted Crop: {crop.upper()}  |  Confidence: {int(conf*100)}%  |  Fertilizers: {', '.join(ferts)}", language='text')

    # note: here we will later replace placeholder_predict with model.predict(...)




