# ----------------- app.py (final robust version) -----------------
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

st.set_page_config(page_title="Crop Recommender — Final", layout="wide")

# ---------- Helpers ----------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

def render_fertilizer_pills(ferts):
    return " ".join([f"<span class='pill'>{f}</span>" for f in ferts])

# ---------- Animations ----------
anim_searching = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json")
anim_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ---------- Artifacts (update names here if different) ----------
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

# ---------- Styling ----------
st.markdown("""
<style>
  .main-card { padding:18px; border-radius:14px; box-shadow: 0 6px 30px rgba(2,6,23,0.15); background:linear-gradient(180deg,#ffffff, #f6fbff); }
  .sub-card { padding:12px; border-radius:10px; background:linear-gradient(90deg,#0f172a,#0b1220); color:#cbd5e1; }
  .result-card { padding:18px; border-radius:12px; background:linear-gradient(90deg,#064e3b,#065f46); color:#fff; }
  .pill { display:inline-block; background: rgba(255,255,255,0.06); color:#e6f4ea; padding:7px 12px; border-radius:999px; margin-right:8px; font-weight:600; }
  .big-crop { font-size:38px; font-weight:800; color:#eafff0; }
  .confidence { font-size:16px; font-weight:700; color:#fff; margin-top:8px; }
  .meta { color:#6b7280; font-size:13px; }
  .debug { font-family: monospace; font-size:12px; color:#8b0000; }
</style>
""", unsafe_allow_html=True)

# ---------- Feature builder ----------
def build_row_features(N, P, K, temperature, humidity, ph, rainfall) -> pd.DataFrame:
    """Build single-row DataFrame with derived features used during training."""
    row = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall
    }])
    # Derived ratios / aggregates
    row['N_P'] = row['N'] / (row['P'] + 1e-6)
    row['N_K'] = row['N'] / (row['K'] + 1e-6)
    row['P_K'] = row['P'] / (row['K'] + 1e-6)
    row['N_P_ratio'] = row['N_P']; row['N_K_ratio'] = row['N_K']; row['P_K_ratio'] = row['P_K']
    row['NPK_sum'] = row[['N','P','K']].sum(axis=1)
    row['NPK_mean'] = row[['N','P','K']].mean(axis=1)

    # pH bucket one-hot (expected by training)
    row['pH_bucket'] = pd.cut(row['ph'], bins=[-999,5.5,6.5,7.5,999],
                              labels=['acidic','slightly_acidic','neutral','alkaline'])
    pH_dummies = pd.get_dummies(row['pH_bucket'])
    for col in ['acidic','slightly_acidic','neutral','alkaline']:
        cname = f"pH_{col}"
        row[cname] = int(col in pH_dummies.columns and pH_dummies.get(col).iloc[0] == 1)

    # Remove the string column (CatBoost / strict models require numeric-only)
    if 'pH_bucket' in row.columns:
        row = row.drop(columns=['pH_bucket'])

    # placeholders used in training
    row['month'] = pd.NA
    row['is_kharif'] = 0; row['is_rabi'] = 0; row['is_zaid'] = 0
    row['rainfall_numeric'] = row['rainfall']
    row['rain_z_region'] = 0.0
    row['temp_numeric'] = row['temperature']
    row['hum_numeric'] = row['humidity']
    row['nutrient_rain_proxy'] = row['NPK_mean'] * row['rain_z_region'].fillna(0)

    # ensure numeric columns are numeric and fillna
    row.replace([np.inf, -np.inf], 0, inplace=True)
    for c in row.columns:
        if row[c].dtype.kind in 'biufc':
            row[c] = pd.to_numeric(row[c], errors='coerce').fillna(0)
    return row

# ---------- Utility: detect model's authoritative feature list ----------
def detect_model_feature_names():
    """Return the feature-name list the model expects (ordered), or None."""
    if model is None:
        return None
    # Try several common attributes / methods in order of preference
    attrs_try = [
        lambda m: m.get_feature_names() if hasattr(m, "get_feature_names") and callable(getattr(m, "get_feature_names")) else None,
        lambda m: getattr(m, "feature_names_", None),
        lambda m: getattr(m, "feature_names", None),
        lambda m: getattr(m, "feature_names_in_", None),
        lambda m: getattr(m, "get_feature_names_out", None)() if hasattr(m, "get_feature_names_out") and callable(getattr(m, "get_feature_names_out")) else None
    ]
    for f in attrs_try:
        try:
            v = f(model)
            if v is not None:
                return list(v)
        except Exception:
            continue
    # fallback
    return None

# ---------- Robust align & predict ----------
def align_and_predict(row: pd.DataFrame) -> Tuple[str, Optional[float], dict]:
    """
    Aligns 'row' to the model's expected feature names exactly and predicts.
    Returns (pred_label, best_prob, prob_dict). If an error occurs, raises with helpful info.
    """
    if model is None:
        raise RuntimeError("Model not loaded (model variable is None).")

    # get authoritative feature names
    required = detect_model_feature_names()
    if required is None:
        # fallback to using row columns (best-effort)
        required = list(row.columns)

    # normalizer for fuzzy match
    def norm(s: str) -> str:
        return str(s).replace("_", "").replace("-", "").replace(" ", "").lower()

    # mapping heuristics for common aliases
    alias_map = {
        'npksum': ['NPK_sum','NPKsum','npk_sum','npksum','npk'],
        'npkmean': ['NPK_mean','NPKmean','npk_mean','npkmean'],
        'npratio': ['N_P_ratio','N_P','n_p_ratio','npratio'],
        'nkratio': ['N_K_ratio','N_K','n_k_ratio','nkratio'],
        'pkratio': ['P_K_ratio','P_K','p_k_ratio','pkratio']
    }

    # helper to find best source column value for required column
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
        # alias mapping
        nr = norm(req_col)
        if nr in alias_map:
            for alias in alias_map[nr]:
                if alias in src.columns:
                    return src.at[0, alias]
                # case-insensitive alias
                for sc in src.columns:
                    if alias.lower() == sc.lower():
                        return src.at[0, sc]
        # fallback to zero
        return 0

    # create input_row with exact required columns
    input_row = pd.DataFrame([{c: 0 for c in required}])

    # fill input_row values using find_value
    for col in required:
        val = find_value(col, row)
        try:
            input_row.at[0, col] = float(val)
        except Exception:
            input_row.at[0, col] = val

    # apply scaler if scaler exposes feature_names_in_ (safe)
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        try:
            scaler_names = list(scaler.feature_names_in_)
            # ensure src row has scaler_names (fill zeros if missing)
            for s in scaler_names:
                if s not in row.columns:
                    row[s] = 0
            scaled = scaler.transform(row[scaler_names])
            for i, s in enumerate(scaler_names):
                if s in input_row.columns:
                    input_row.at[0, s] = float(scaled[0, i])
        except Exception:
            # if scaling fails, we proceed with unscaled values (best-effort)
            pass

    # reorder to required
    input_row = input_row[required]

    # final dtype coercion where reasonable
    for c in input_row.columns:
        try:
            input_row[c] = pd.to_numeric(input_row[c], errors='ignore')
        except Exception:
            pass

    # Predict
    try:
        preds = model.predict(input_row)
    except Exception as e:
        # Provide helpful debug info
        err_msg = (
            f"Model prediction failed: {e}\n\n"
            f"Model expected feature names (first 30): {required[:30]}\n\n"
            f"Constructed input columns (first 40): {list(input_row.columns)[:40]}\n\n"
            f"Sample of input row values:\n{input_row.iloc[0,:50].to_dict()}"
        )
        raise RuntimeError(err_msg)

    pred_idx = preds[0]
    # decode label if possible
    try:
        pred_label = label_encoder.inverse_transform([pred_idx])[0] if label_encoder is not None else pred_idx
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

# ---------- App UI ----------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
left, right = st.columns([1,1])

with left:
    st.markdown("<h1>🌾 Crop Recommendation — Final</h1>", unsafe_allow_html=True)
    st.markdown("<div class='meta'>Enter soil & weather parameters and click Predict. Model artifacts load from repo root.</div>", unsafe_allow_html=True)
    with st.form("input_form"):
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=1000, value=90)
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=1000, value=42)
        K = st.number_input("Potassium (K)", min_value=0, max_value=1000, value=43)
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=20.0, format="%.1f")
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, format="%.1f")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=200.0, format="%.1f")
        submit = st.form_submit_button("Predict")

with right:
    st.markdown("<div class='sub-card'><b>Tips</b><br>- Use realistic nutrient values.<br>- pH around 6–7 suits many crops.</div>", unsafe_allow_html=True)
    if anim_searching:
        st_lottie(anim_searching, height=200, key="ui_anim")

st.markdown("</div>", unsafe_allow_html=True)

# show load warnings
if load_errors:
    st.warning("Model artifacts load warnings:\n" + "\n".join(load_errors))

# Prediction flow
if submit:
    if anim_searching:
        st_lottie(anim_searching, height=200, key=f"search_{random.random()}")
    st.markdown("<b>Analyzing data...</b>", unsafe_allow_html=True)
    prog = st.progress(0)
    for i in range(0, 101, 10):
        prog.progress(min(i,100)); time.sleep(0.03 + random.random()*0.02)
    prog.progress(100)

    try:
        row = build_row_features(N, P, K, temp, hum, ph, rain)
        if model is not None:
            pred_label, prob, prob_dict = align_and_predict(row)
            fert_map = {'rice':['Urea','DAP','MOP','Zinc Sulphate'],'maize':['Urea','SSP','Potash'],
                        'papaya':['NPK','Potash'],'wheat':['Urea','DAP','MOP']}
            ferts = fert_map.get(str(pred_label).lower(), ["General NPK"])
        else:
            pred_label = "RICE"; prob = None; prob_dict = {}; ferts = ["Urea","DAP","MOP"]
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.error(str(e))
        pred_label = "ERROR"; prob = None; prob_dict = {}; ferts = []

    if anim_success: st_lottie(anim_success, height=160, key=f"succ_{random.random()}")

    # show result
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-crop'>{str(pred_label).upper()}</div>", unsafe_allow_html=True)
    if prob is not None:
        st.markdown(f"<div class='confidence'>Confidence: {prob*100:.1f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-card'><b>Recommended Fertilizers:</b><br>" + render_fertilizer_pills(ferts) + "</div>", unsafe_allow_html=True)

    if prob_dict:
        top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Top predictions:**")
        for name, p in top3:
            st.write(f"- {name} — {p*100:.2f}%")

    st.balloons(); st.snow()
    st.code(f"Predicted Crop: {str(pred_label).upper()}  |  Fertilizers: {', '.join(ferts)}", language='text')
# ----------------- end app.py -----------------
