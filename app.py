# app.py — Clean, corrected animated demo (no anim_placeholder usage)
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import time
import random

# ---------- Config ----------
st.set_page_config(page_title="Crop Recommender — Demo", layout="wide")

# ---------- Helpers ----------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

def placeholder_predict(N,P,K,temp,hum,ph,rain):
    score = (N*0.6 + P*0.2 + K*0.2) + (rain/10) - abs(6.5-ph)*20
    if score > 200:
        return "Rice", ["Urea", "DAP", "MOP", "Zinc Sulphate"], 0.92
    if ph < 6:
        return "Maize", ["DAP", "MOP"], 0.78
    if hum > 85 and temp > 25:
        return "Papaya", ["NPK", "Potash"], 0.81
    return "Wheat", ["NPK", "DAP"], 0.65

def render_fertilizer_pills(ferts):
    html = ""
    for f in ferts:
        html += f"<span class='pill'>{f}</span> "
    return html

# ---------- Load animations ----------
anim_searching = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json")
anim_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ---------- CSS ----------
st.markdown(
    """
    <style>
      .main-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:18px; box-shadow: 0 6px 30px rgba(2,6,23,0.5); }
      .result-card { background: linear-gradient(90deg,#064e3b,#065f46); color: white; border-radius:12px; padding:18px; margin-top:12px; }
      .sub-card { background: linear-gradient(90deg,#0f172a,#0b1220); color: #cbd5e1; border-radius:10px; padding:12px; }
      .pill { display:inline-block; background: rgba(255,255,255,0.06); color: #e6f4ea; padding:7px 12px; border-radius:999px; margin-right:8px; font-weight:600; }
      .big-crop { font-size:40px; font-weight:800; color: #eafff0; }
      .confidence { font-size:18px; font-weight:700; color:#fff; opacity:0.95; margin-top:8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Layout ----------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
col1, col2 = st.columns([1,1])

with col1:
    st.header("🌾 Crop Recommendation — Demo")
    st.write("Enter soil & weather parameters and click **Predict**. (Model will be hooked later.)")

    with st.form("input_form"):
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=1000, value=90, step=1)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=1000, value=42, step=1)
        K = st.number_input("Potassium (K)", min_value=0, max_value=1000, value=43, step=1)
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, format="%.1f")
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=200.0, format="%.1f")
        submit = st.form_submit_button("Predict")

with col2:
    st.markdown("<div class='sub-card'><b>Tips</b><br>- Use realistic nutrient values.<br>- pH around 6–7 suits many crops.<br>- Demo shows animation only for UX.</div>", unsafe_allow_html=True)
    if anim_searching:
        st_lottie(anim_searching, height=180, key="init_anim")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prediction block (clean) ----------
if submit:
    # show searching animation (direct st_lottie call)
    if anim_searching:
        st_lottie(anim_searching, height=220, key="search_anim")

    status = st.empty()
    status.markdown("<b>Analysing soil & weather data...</b>", unsafe_allow_html=True)

    prog = st.progress(0)
    for pct in range(0, 101, 8):
        prog.progress(min(pct, 100))
        time.sleep(0.06 + random.random() * 0.03)
    prog.progress(100)

    # compute prediction (placeholder)
    crop, ferts, conf = placeholder_predict(N, P, K, temp, hum, ph, rain)

    # success animation
    if anim_success:
        st_lottie(anim_success, height=160, key="success_anim")

    # display result card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-crop'>{crop.upper()}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>Confidence: {int(conf*100)}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # fertilizer pills
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-card'><b>Recommended Fertilizers:</b><br>" + render_fertilizer_pills(ferts) + "</div>", unsafe_allow_html=True)

    # celebration
    st.balloons()
    st.snow()

    # copyable summary
    st.code(f"Predicted Crop: {crop.upper()}  |  Confidence: {int(conf*100)}%  |  Fertilizers: {', '.join(ferts)}", language='text')
