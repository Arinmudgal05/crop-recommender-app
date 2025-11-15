# ----------------- app.py (complete, final fix for feature alignment) -----------------
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import time
import random
import joblib
import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

st.set_page_config(page_title="Crop Recommender — Demo (Real Model)", layout="wide")

# ---------------- helpers ----------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

def render_fertilizer_pills(ferts):
    pills_html = ""
    for f in ferts:
        pills_html += f"<span class='pill'>{f}</span> "
    return pills_html

# ---------------- load animations ----------------
anim_searching = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json")
anim_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ---------------- try to load model artifacts ----------------
MODEL_PATH = "model_model.joblib"
SCALER_PATH = "scaler_scaler.joblib"
LABELENC_PATH = "le_label_encoder.joblib"

model = None
scaler = None
label_encoder = None
load_errors = []

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        load_errors.append(f"Missing model file: {MODEL_PATH}")
except Exception as e:
    load_errors.append(f"Error loading model: {e}")

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        load_errors.append(f"Missing scaler file: {SCALER_PATH}")
except Exception as e:
    load_errors.append(f"Error loading scaler: {e}")

try:
    if os.path.exists(LABELENC_PATH):
        label_encoder = joblib.load(LABELENC_PATH)
    else:
        load_errors.append(f"Missing label encoder file: {LABELENC_PATH}")
except Exception as e:
    load_errors.append(f"Error loading label encoder: {e}")

# ---------------- styling ----------------
st.markdown(
    """
    <style>
      .main-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:18px; box-shadow: 0 6px 30px rgba(2,6,23,0.5); }
      .result-card { background: linear-gradient(90deg,#064e3b,#065f46); color: white; border-radius:12px; padding:18px; margin-top:12px; }
      .sub-card { background: linear-gradient(90deg,#0f172a,#0b1220); color: #cbd5e1; border-radius:10px; padding:12px; }
      .pill { display:inline-block; background: rgba(255,255,255,0.06); color: #e6f4ea; padding:7px 12px; border-radius:999px; margin-right:8px; font-weight:600; }
      .big-crop { font-size:40px; font-weight:800; color: #eafff0; }
      .confidence { font-size:18px; font-weight:700; color:#fff; opacity:0.95; margin-top:8px; }
      .meta { color:#a8b3c7; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- feature builder for single row (fixed: drop pH_bucket string) ----------------
def build_row_features(N,P,K,temperature,humidity,ph,rainfall) -> pd.DataFrame:
    row = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Derived numeric features (include compat names)
    row['N_P'] = row['N'] / (row['P'] + 1e-6)
    row['N_K'] = row['N'] / (row['K'] + 1e-6)
    row['P_K'] = row['P'] / (row['K'] + 1e-6)
    row['N_P_ratio'] = row['N_P']
    row['N_K_ratio'] = row['N_K']
    row['P_K_ratio'] = row['P_K']
    row['NPK_sum'] = row[['N','P','K']].sum(axis=1)
    row['NPK_mean'] = row[['N','P','K']].mean(axis=1)

    # pH bucket -> one-hot expected by model
    row['pH_bucket'] = pd.cut(row['ph'], bins=[-999,5.5,6.5,7.5,999],
                              labels=['acidic','slightly_acidic','neutral','alkaline'])
    pH_dummies = pd.get_dummies(row['pH_bucket'])
    for col in ['acidic','slightly_acidic','neutral','alkaline']:
        cname = f"pH_{col}"
        row[cname] = int(col in pH_dummies.columns and pH_dummies.get(col).iloc[0] == 1)

    # REMOVE raw string column
    if "pH_bucket" in row.columns:
        row = row.drop(columns=["pH_bucket"])

    # season/month placeholders
    row['month'] = pd.NA
    row['is_kharif'] = 0
    row['is_rabi'] = 0
    row['is_zaid'] = 0

    # rainfall numeric / normalized region placeholder
    row['rainfall_numeric'] = row['rainfall']
    row['rain_z_region'] = 0.0

    # temp/hum numeric proxies
    row['temp_numeric'] = row['temperature']
    row['hum_numeric'] = row['humidity']

    # nutrient_rain_proxy
    row['nutrient_rain_proxy'] = row['NPK_mean'] * row['rain_z_region'].fillna(0)

    # Ensure no inf and fill numeric NA with 0
    row.replace([np.inf, -np.inf], 0, inplace=True)
    for c in row.columns:
        if row[c].dtype.kind in 'biufc':
            row[c] = pd.to_numeric(row[c], errors='coerce').fillna(0)
    return row

# ---------------- robust align + predict helper ----------------
def align_and_predict(row: pd.DataFrame) -> Tuple[str, Optional[float], dict]:
    """
    Build an input DataFrame whose columns are exactly model.feature_names_in_ (if available),
    fill each required column by finding the best matching value from 'row',
    then apply scaler (if present) using scaler.feature_names_in_ and overwrite those columns,
    and finally call model.predict / predict_proba.
    """
    if model is None:
        raise RuntimeError("Model not loaded. Check model_model.joblib in repo root.")

    # get required feature names in correct order
    if hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
    else:
        required = list(row.columns)

    # helper to normalize strings for matching
    def norm(s: str) -> str:
        return str(s).replace('_','').replace('-','').replace(' ','').lower()

    # helper: try various matching strategies to find value for required column
    def find_value(req_col: str, src: pd.DataFrame):
        # exact
        if req_col in src.columns:
            return src.at[0, req_col]
        # case-insensitive
        lower_map = {c.lower(): c for c in src.columns}
        if req_col.lower() in lower_map:
            return src.at[0, lower_map[req_col.lower()]]
        # normalized exact
        norm_map = {norm(c): c for c in src.columns}
        if norm(req_col) in norm_map:
            return src.at[0, norm_map[norm(req_col)]]
        # substring heuristic
        for c in src.columns:
            if norm(req_col) in norm(c) or norm(c) in norm(req_col):
                return src.at[0, c]
        # some explicit alias mapping
        alias_map = {
            'npk_sum': ['npk_sum','npksum','NPK_sum','NPKsum','npk'],
            'n_p_ratio': ['n_p_ratio','n_p','N_P_ratio','N_P'],
            'n_k_ratio': ['n_k_ratio','n_k','N_K_ratio','N_K'],
            'p_k_ratio': ['p_k_ratio','p_k','P_K_ratio','P_K']
        }
        nr = norm(req_col)
        if nr in alias_map:
            for a in alias_map[nr]:
                if a in src.columns:
                    return src.at[0, a]
        # fallback 0
        return 0

    # Create input_row with exact required columns (initialized to zeros)
    input_row = pd.DataFrame([ {c: 0 for c in required} ])

    # First pass: fill every required column by finding best match from row
    for col in required:
        val = find_value(col, row)
        # coerce numbers where possible
        try:
            input_row.at[0, col] = float(val)
        except Exception:
            input_row.at[0, col] = val

    # Second pass: if scaler present, apply scaling to scaler.feature_names_in_ and overwrite those columns
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        scaler_names = list(scaler.feature_names_in_)
        # ensure source row has scaler_names (fill zeros if missing)
        for s in scaler_names:
            if s not in row.columns:
                row[s] = 0
        try:
            scaled = scaler.transform(row[scaler_names])
            for i, s in enumerate(scaler_names):
                if s in input_row.columns:
                    input_row.at[0, s] = float(scaled[0, i])
        except Exception:
            # if transform fails, leave unscaled values (best-effort)
            pass

    # Ensure correct column order (model expects this order)
    input_row = input_row[required]

    # Final safety: convert numeric-like columns to numeric dtype
    for c in input_row.columns:
        try:
            input_row[c] = pd.to_numeric(input_row[c], errors='ignore')
        except Exception:
            pass

    # Predict
    preds = model.predict(input_row)
    pred_idx = preds[0]
    pred_label = pred_idx
    try:
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
    except Exception:
        pred_label = pred_idx

    # probabilities
    prob = None
    prob_dict = {}
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_row)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                decoded = []
                for c in classes:
                    try:
                        if label_encoder is not None:
                            decoded.append(label_encoder.inverse_transform([int(c)])[0])
                        else:
                            decoded.append(c)
                    except Exception:
                        decoded.append(c)
                for cname, p in zip(decoded, probs):
                    prob_dict[cname] = float(p)
            else:
                for i, p in enumerate(probs):
                    prob_dict[str(i)] = float(p)
            if prob_dict:
                prob = max(prob_dict.values())
    except Exception:
        prob = None

    return pred_label, prob, prob_dict

# ---------------- App layout & logic ----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
left, right = st.columns([1,1])

with left:
    st.markdown("<h1>🌾 Crop Recommendation — Demo (Real Model)</h1>", unsafe_allow_html=True)
    st.markdown("<div class='meta'>Enter soil parameters and press Predict. Model artifacts are loaded from repo root.</div>", unsafe_allow_html=True)

    with st.form(key='input_form'):
        N = st.number_input("Enter Nitrogen (N)", min_value=0, max_value=1000, value=90, step=1)
        P = st.number_input("Enter Phosphorus (P)", min_value=0, max_value=1000, value=42, step=1)
        K = st.number_input("Enter Potassium (K)", min_value=0, max_value=1000, value=43, step=1)
        temp = st.number_input("Enter Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, format="%.1f")
        hum = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
        ph = st.number_input("Enter Soil pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
        rain = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=10000.0, value=200.0, format="%.1f")
        submit = st.form_submit_button("Predict")

with right:
    st.markdown("<div class='sub-card'><b>Tips</b><br>- Use realistic nutrient values.<br>- pH ~6–7 suits many crops.</div>", unsafe_allow_html=True)
    if anim_searching:
        st_lottie(anim_searching, height=200, key="ui_anim")

st.markdown("</div>", unsafe_allow_html=True)

# show any load errors
if load_errors:
    st.warning("Some model artifacts could not be loaded. The app will run in placeholder mode.\n\n" + "\n".join(load_errors))

# Prediction flow
if submit:
    # animation and progress
    if anim_searching:
        st_lottie(anim_searching, height=200, key=f"search_{random.random()}")
    st.markdown("<b>Analysing data...</b>", unsafe_allow_html=True)
    prog = st.progress(0)
    for i in range(0, 101, 8):
        prog.progress(min(i,100))
        time.sleep(0.04 + random.random()*0.03)
    prog.progress(100)
    time.sleep(0.12)

    # Build row & predict (real if model loaded, else fallback to placeholder)
    try:
        row = build_row_features(N,P,K,temp,hum,ph,rain)
        if model is not None:
            pred_label, prob, prob_dict = align_and_predict(row)
            fertilizer_map = {
                'rice': ['Urea','DAP','MOP','Zinc Sulphate'],
                'maize': ['Urea','SSP','Potash','Zinc Sulphate'],
                'papaya': ['NPK','Potash'],
                'wheat': ['Urea','DAP','MOP']
            }
            ferts = fertilizer_map.get(str(pred_label).lower(), ["General NPK"])
        else:
            pred_label, ferts = ("RICE", ["Urea","DAP","MOP","Zinc Sulphate"])
            prob = None
            prob_dict = {}
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        pred_label = "ERROR"
        ferts = []
        prob = None
        prob_dict = {}

    # success animation
    if anim_success:
        st_lottie(anim_success, height=160, key=f"succ_{random.random()}")

    # result card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-crop'>{str(pred_label).upper()}</div>", unsafe_allow_html=True)
    if prob is not None:
        st.markdown(f"<div class='confidence'>Confidence: {prob*100:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # fertilizer pills
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-card'><b>Recommended Fertilizers:</b><br>" + render_fertilizer_pills(ferts) + "</div>", unsafe_allow_html=True)

    # show top-3 probabilities if present
    if prob_dict:
        top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Top predictions:**")
        for name, p in top3:
            st.write(f"- {name} — {p*100:.2f}%")

    st.balloons()
    st.snow()

    st.code(f"Predicted Crop: {str(pred_label).upper()}  |  Fertilizers: {', '.join(ferts)}", language='text')

# ---------------- end app.py -----------------
