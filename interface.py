import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

#  Page Configuration 
st.set_page_config(
    page_title="Plateau Crop Advisor",
    page_icon="🌱",
    layout="wide"
)

#  Theme 
BG        = "#f7f4ee"        # warm off-white parchment
PANEL     = "#ffffff"
BORDER    = "#ddd8cc"
SOIL      = "#5c4a2a"        # rich dark soil brown — primary accent
SOIL_LITE = "#8a6f48"
LEAF      = "#3d6b3f"        # deep leaf green — secondary accent
LEAF_LITE = "#6aab6d"
MUTED     = "#8a8278"
INK       = "#2a2318"
GOLD      = "#c8923a"        # harvest gold for ranking highlights

#  Google Fonts + Global CSS ─
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
    color: {INK};
}}

.stApp {{
    background-color: {BG};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {SOIL};
    border-right: none;
}}
[data-testid="stSidebar"] * {{
    color: #f0ebe0 !important;
}}
[data-testid="stSidebar"] label {{
    font-size: 0.82em !important;
    letter-spacing: 0.04em !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    color: #c8b99a !important;
}}
[data-testid="stNumberInput"] input {{
    background-color: rgba(255,255,255,0.12) !important;
    color: #f0ebe0 !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 6px !important;
}}

/* Primary Button */
.stButton > button {{
    background-color: {LEAF} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9em !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.65em 1.2em !important;
    width: 100% !important;
    transition: background 0.2s !important;
}}
.stButton > button:hover {{
    background-color: {LEAF_LITE} !important;
}}

hr {{
    border-color: {BORDER} !important;
}}

h1, h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: {INK} !important;
}}

.stSpinner > div {{
    color: {LEAF} !important;
}}

/* Remove streamlit default top padding */
.block-container {{
    padding-top: 2rem !important;
}}
</style>
""", unsafe_allow_html=True)

#  Load Artifacts ─
MODELS_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "label_map.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
features   = meta["features"]
decode_map = meta["decode"]

# Initialize Session State
if "prediction" not in st.session_state:
    st.session_state.prediction = None

#  Constants 
CROP_EMOJI = {
    'Irish-potato': '🥔', 'Tomato': '🍅', 'Groundnut': '🥜',
    'Pepper': '🌶️',  'Cabbage': '🥬', 'Maize': '🌽',
    'Wheat': '🌾',   'Pineapple': '🍍', 'Onion': '🧅',
}

CROP_CONFIG = {
    'Irish-potato': {'pH': {'mean': 4.68, 'std': 0.23, 'min': 4.0,  'max': 5.36}, 'N': {'mean': 100,  'std': 10,   'min': 80,  'max': 120}, 'P': {'mean': 50,   'std': 5,    'min': 40,  'max': 60},  'K': {'mean': 150,  'std': 15,   'min': 120, 'max': 180}, 'Temp': {'mean': 15,   'std': 1,    'min': 10,  'max': 20},  'Rainfall': {'mean': 500,  'std': 50,   'min': 400, 'max': 600},  'Salinity': {'mean': 0.8, 'std': 0.15, 'min': 0.3, 'max': 1.5}},

    'Tomato':  {'pH': {'mean': 6.0,  'std': 0.33, 'min': 5.0,  'max': 7.0},  'N': {'mean': 125,  'std': 8.33, 'min': 100, 'max': 150}, 'P': {'mean': 87.5, 'std': 7.5,  'min': 65,  'max': 110}, 'K': {'mean': 200,  'std': 13.3, 'min': 160, 'max': 240}, 'Temp': {'mean': 21.5, 'std': 1.17, 'min': 18,  'max': 25},  'Rainfall': {'mean': 500,  'std': 33.3, 'min': 400, 'max': 600},  'Salinity': {'mean': 1.0, 'std': 0.2,  'min': 0.5, 'max': 2.0}},

    'Groundnut': {'pH': {'mean': 6.5,  'std': 0.17, 'min': 6.0,  'max': 7.0},  'N': {'mean': 15,   'std': 1.67, 'min': 10,  'max': 20},  'P': {'mean': 27.5, 'std': 4.17, 'min': 15,  'max': 40},  'K': {'mean': 32.5, 'std': 2.5,  'min': 25,  'max': 40},  'Temp': {'mean': 25,   'std': 1,    'min': 22,  'max': 28},  'Rainfall': {'mean': 600,  'std': 33.3, 'min': 500, 'max': 700},  'Salinity': {'mean': 0.6, 'std': 0.1,  'min': 0.2, 'max': 1.2}},

    'Pepper':  {'pH': {'mean': 6.25, 'std': 0.25, 'min': 5.5,  'max': 7.0},  'N': {'mean': 135,  'std': 11.7, 'min': 100, 'max': 170}, 'P': {'mean': 37.5, 'std': 4.17, 'min': 25,  'max': 50},  'K': {'mean': 77.5, 'std': 7.5,  'min': 55,  'max': 100}, 'Temp': {'mean': 22.5, 'std': 1.5,  'min': 18,  'max': 27},  'Rainfall': {'mean': 925,  'std': 108.3,'min': 600, 'max': 1250}, 'Salinity': {'mean': 1.2, 'std': 0.2,  'min': 0.6, 'max': 2.0}},

    'Cabbage':  {'pH': {'mean': 6.5,  'std': 0.17, 'min': 6.0,  'max': 7.0},  'N': {'mean': 125,  'std': 8.33, 'min': 100, 'max': 150}, 'P': {'mean': 57.5, 'std': 2.5,  'min': 50,  'max': 65},  'K': {'mean': 115,  'std': 5,    'min': 100, 'max': 130}, 'Temp': {'mean': 17,   'std': 2.33, 'min': 10,  'max': 24},  'Rainfall': {'mean': 440,  'std': 20,   'min': 380, 'max': 500},  'Salinity': {'mean': 0.7, 'std': 0.12, 'min': 0.2, 'max': 1.3}},

    'Maize':   {'pH': {'mean': 6.5,  'std': 0.33, 'min': 5.5,  'max': 7.5},  'N': {'mean': 175,  'std': 8.33, 'min': 150, 'max': 200}, 'P': {'mean': 65,   'std': 5,    'min': 50,  'max': 80},  'K': {'mean': 80,   'std': 6.67, 'min': 60,  'max': 100}, 'Temp': {'mean': 24,   'std': 1.33, 'min': 20,  'max': 28},  'Rainfall': {'mean': 650,  'std': 50,   'min': 500, 'max': 800},  'Salinity': {'mean': 1.0, 'std': 0.2,  'min': 0.4, 'max': 2.0}},

    'Wheat':   {'pH': {'mean': 7.0,  'std': 0.33, 'min': 6.0,  'max': 8.0},  'N': {'mean': 125,  'std': 8.33, 'min': 100, 'max': 150}, 'P': {'mean': 40,   'std': 1.67, 'min': 35,  'max': 45},  'K': {'mean': 37.5, 'std': 4.17, 'min': 25,  'max': 50},  'Temp': {'mean': 17.5, 'std': 0.83, 'min': 15,  'max': 20},  'Rainfall': {'mean': 550,  'std': 33.3, 'min': 450, 'max': 650},  'Salinity': {'mean': 1.5, 'std': 0.25, 'min': 0.8, 'max': 2.5}},

    'Pineapple':    {'pH': {'mean': 5.5,  'std': 0.33, 'min': 4.5,  'max': 6.5},  'N': {'mean': 265,  'std': 11.7, 'min': 230, 'max': 300}, 'P': {'mean': 55,   'std': 3.33, 'min': 45,  'max': 65},  'K': {'mean': 165,  'std': 18.3, 'min': 110, 'max': 220}, 'Temp': {'mean': 24,   'std': 0.67, 'min': 22,  'max': 26},  'Rainfall': {'mean': 850,  'std': 50,   'min': 700, 'max': 1000}, 'Salinity': {'mean': 0.5, 'std': 0.1,  'min': 0.1, 'max': 1.0}},
    
    'Onion':   {'pH': {'mean': 6.5,  'std': 0.17, 'min': 6.0,  'max': 7.0},  'N': {'mean': 80,   'std': 6.67, 'min': 60,  'max': 100}, 'P': {'mean': 35,   'std': 3.33, 'min': 25,  'max': 45},  'K': {'mean': 62.5, 'std': 5.83, 'min': 45,  'max': 80},  'Temp': {'mean': 17.5, 'std': 0.83, 'min': 15,  'max': 20},  'Rainfall': {'mean': 450,  'std': 33.3, 'min': 350, 'max': 550},  'Salinity': {'mean': 0.9, 'std': 0.15, 'min': 0.4, 'max': 1.8}},
}


UNITS = {
    'pH': 'pH', 'N': 'kg/ha', 'P': 'kg/ha',
    'K': 'kg/ha', 'Temp': '°C', 'Rainfall': 'mm', 'Salinity': 'dS/m'
}

features = ["N", "P", "K", "pH", "Temp", "Rainfall", "Salinity"]

CORRELATION_MATRIX = np.array([
    [ 1.0,  0.4,  0.4,  0.1,  0.0, -0.2,  0.1],  # N
    [ 0.4,  1.0,  0.3,  0.1,  0.0, -0.2,  0.0],  # P
    [ 0.4,  0.3,  1.0,  0.1,  0.0, -0.2,  0.1],  # K
    [ 0.1,  0.1,  0.1,  1.0,  0.0, -0.5, -0.3],  # pH
    [ 0.0,  0.0,  0.0,  0.0,  1.0, -0.3,  0.4],  # Temp
    [-0.2, -0.2, -0.2, -0.5, -0.3,  1.0, -0.4],  # Rainfall
    [ 0.1,  0.0,  0.1, -0.3,  0.4, -0.4,  1.0]   # Salinity
])

STATE_BASELINE = {
    'pH': 6.2,
    'N': 120.0,
    'P': 50.0,
    'K': 110.0,
    'Temp': 20.0,
    'Rainfall': 600.0,
    'Salinity': 1.0
}

LGAs = {
    'Barkin Ladi', 'Bassa', 'Bokkos', 'Jos East', 'Jos North', 'Jos South',
    'Kanam', 'Kanke', 'Langtang North', 'Langtang South', 'Mangu', 'Mikang',
    'Pankshin', "Qua'an-Pan", 'Riyom', 'Shendam', 'Wase'
}
HIGHLAND_LGAS = {'Bassa', 'Jos North', 'Jos South', 'Jos East',
                 'Bokkos', 'Pankshin', 'Mangu', 'Riyom'}
LGA_CLIMATE = {lga: 'Highland' if lga in HIGHLAND_LGAS else 'Lowland' for lga in LGAs}

CLIMATE_ADJUSTMENTS = {
    'Highland': {'Temp': -3.0, 'Rainfall': +100.0, 'pH': -0.3, 'N': -10.0, 'P': -5.0, 'K': -8.0, 'Salinity': -0.2},
    'Lowland':  {'Temp': +3.0, 'Rainfall': -100.0, 'pH': +0.3, 'N': +10.0, 'P': +5.0, 'K': +8.0, 'Salinity': +0.3}
}

np.random.seed(42)
LGA_SOIL_BIAS = {
    lga: {
        'pH':np.random.uniform(-0.25, 0.25),
        'N': np.random.uniform(-12, 12),
        'P': np.random.uniform(-6, 6),
        'K': np.random.uniform(-10, 10),
        'Salinity':np.random.uniform(-0.1, 0.2),
    }
    for lga in LGAs
}

LGA_DESCRIPTION = {
    lga: (
        "A highland LGA with cool temperatures and slightly acidic soils — "
        "ideal for temperature-sensitive crops that need well-drained upland conditions."
    ) if lga in HIGHLAND_LGAS else (
        "A lowland LGA with warmer temperatures and higher salinity soils — "
        "well-suited for heat-tolerant crops that thrive with good rainfall coverage."
    )
    for lga in LGAs
}

#  Sidebar 
st.sidebar.markdown(f"""
    <div style="padding: 8px 0 24px 0;">
        <p style="color:#c8b99a; font-size:0.68em; letter-spacing:0.18em;
                  text-transform:uppercase; margin:0;">
            Plateau State, Nigeria
        </p>
        <h2 style="color:#f0ebe0; margin:6px 0 0 0; font-size:1.35em;
                   font-family:'Playfair Display', serif; letter-spacing:0.02em;">
            Soil & Climate Inputs
        </h2>
        <hr style="border-color:rgba(255,255,255,0.15); margin-top:14px;">
    </div>
""", unsafe_allow_html=True)

N        = st.sidebar.number_input("Nitrogen (N) — kg/ha",    min_value=0.0, max_value=300.0, value=115.0, step=1.0)
P        = st.sidebar.number_input("Phosphorus (P) — kg/ha",  min_value=0.0, max_value=300.0, value=69.0,  step=1.0)
K        = st.sidebar.number_input("Potassium (K) — kg/ha",   min_value=0.0, max_value=300.0, value=225.0, step=1.0)
pH       = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0,  value=5.3,   step=0.1)
Temp     = st.sidebar.number_input("Temperature — °C", min_value=0.0, max_value=50.0,  value=17.0,  step=0.5)
Rainfall = st.sidebar.number_input("Rainfall — mm",  min_value=0.0, max_value=500.0, value=360.0, step=5.0)
Salinity = st.sidebar.number_input("Salinity — dS/m", min_value=0.0, max_value=10.0,  value=1.8,   step=0.1)

st.sidebar.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("Analyse & Recommend", use_container_width=True)

st.sidebar.markdown(f"""
    <div style="margin-top:28px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.15);">
        <p style="color:#c8b99a; font-size:0.75em; line-height:1.65;">
            This tool uses a trained machine learning model to recommend the most
            suitable crop and matching Plateau State LGAs based on your soil and
            climate measurements.
        </p>
    </div>
""", unsafe_allow_html=True)

#  Hero Header 
st.markdown(f"""
    <div style="padding: 8px 0 28px 0; border-bottom: 2px solid {BORDER}; margin-bottom:32px;">
        <p style="color:{MUTED}; font-size:0.72em; letter-spacing:0.22em;
                  text-transform:uppercase; margin:0 0 6px 0;">
            AI-Powered Agriculture · Plateau State
        </p>
        <h1 style="font-family:'Playfair Display', serif; font-size:2.8em; font-weight:800;
                   color:{INK}; margin:0; line-height:1.1;">
            Crop Advisor
        </h1>
        <p style="color:{MUTED}; font-size:0.97em; margin-top:10px; max-width:560px; line-height:1.6;">
            Enter your soil and climate parameters on the left to receive a data-driven
            crop recommendation and find the best-matched LGAs to cultivate in.
        </p>
    </div>
""", unsafe_allow_html=True)

#  Idle/Prediction State Check ─
if st.session_state.prediction is None and not predict_btn:
    st.markdown(f"""
        <div style="background:{PANEL}; border:1.5px dashed {BORDER};
                    border-radius:12px; padding:48px 40px; text-align:center; margin-top:8px;">
            <div style="font-size:3em; margin-bottom:12px;"></div>
            <p style="color:{MUTED}; font-size:1em; margin:0;">
                ← Enter your soil values in the sidebar, then click
                <strong style="color:{LEAF};">Analyse &amp; Recommend</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Run prediction if button was clicked
    if predict_btn:
        user_input   = pd.DataFrame([[N, P, K, pH, Temp, Rainfall, Salinity]], columns=features)
        input_scaled = scaler.transform(user_input)
        pred_crop    = decode_map[model.predict(input_scaled)[0]]
        crop_emoji   = CROP_EMOJI.get(pred_crop)
        
        st.session_state.prediction = {
            "pred_crop": pred_crop,
            "crop_emoji": crop_emoji,
            "user_input": user_input,
            "input_scaled": input_scaled,
            "N": N, "P": P, "K": K, "pH": pH, "Temp": Temp, "Rainfall": Rainfall, "Salinity": Salinity
        }
    
    # Retrieve active prediction snapshot from session state
    pred = st.session_state.prediction
    pred_crop = pred["pred_crop"]
    crop_emoji = pred["crop_emoji"]
    user_input = pred["user_input"]
    input_scaled = pred["input_scaled"]
    user_N = pred["N"]
    user_P = pred["P"]
    user_K = pred["K"]
    user_pH = pred["pH"]
    user_Temp = pred["Temp"]
    user_Rainfall = pred["Rainfall"]
    user_Salinity = pred["Salinity"]

    #  LGA Scoring 
    def lga_match_score(lga):
        climate = LGA_CLIMATE[lga]
        adj     = CLIMATE_ADJUSTMENTS[climate]
        bias    = LGA_SOIL_BIAS[lga]
        
        # 1. Construct user input vector in the exact order of features
        user_vals = np.array([user_N, user_P, user_K, user_pH, user_Temp, user_Rainfall, user_Salinity])
        
        # 2. Extract means and stds in the exact order of features
        mu = []
        stds = []
        for param in features:
            stats = CROP_CONFIG[pred_crop][param]
            mean = stats["mean"] + adj.get(param, 0.0) + bias.get(param, 0.0)
            std = stats["std"] * 3.5  # standard deviation multiplier from data generator
            mu.append(mean)
            stds.append(std)
            
        mu = np.array(mu)
        stds = np.array(stds)
        
        # 3. Construct covariance matrix Sigma
        stds_diag = np.diag(stds)
        Sigma = stds_diag @ CORRELATION_MATRIX @ stds_diag
        
        # 4. Invert covariance matrix with fallback
        try:
            inv_Sigma = np.linalg.inv(Sigma)
            # 5. Compute squared Mahalanobis distance D_squared
            diff = user_vals - mu
            D_squared = diff @ inv_Sigma @ diff.T
            # 6. Calculate continuous matching percentage
            pct = int(round(np.exp(-0.05 * D_squared) * 100))
        except np.linalg.LinAlgError:
            pct = 1
            
        pct = int(np.clip(pct, 1, 100))
        
        # 7. Check which of the 4 display parameters fall within crop range
        display_params = ["pH", "Salinity", "Temp", "Rainfall"]
        matched_params = []
        checks = {"Temp": user_Temp, "Rainfall": user_Rainfall, "pH": user_pH, "Salinity": user_Salinity}
        for param in display_params:
            if CROP_CONFIG[pred_crop][param]["min"] <= checks[param] <= CROP_CONFIG[pred_crop][param]["max"]:
                matched_params.append(param)
                
        return pct, matched_params

    lga_results  = {lga: lga_match_score(lga) for lga in LGAs}
    sorted_lgas  = sorted(lga_results, key=lambda l: lga_results[l][0], reverse=True)
    top_lgas     = sorted_lgas[:3]

    def build_lga_explanation(lga, pct, matched_params):
        climate   = LGA_CLIMATE[lga]
        adj       = CLIMATE_ADJUSTMENTS[climate]
        temp_dir  = "cooler" if adj["Temp"] < 0 else "warmer"
        rain_dir  = "higher" if adj["Rainfall"] > 0 else "lower"
        match_str = ", ".join(matched_params) if matched_params else "few parameters"
        return (
            f"{lga} is a {climate.lower()} LGA with {temp_dir} temperatures and "
            f"{rain_dir} rainfall than the state average — matching {pct}% of the ideal "
            f"conditions for {pred_crop}. "
            f"Your inputs for {match_str} fall within the recommended range for this area, "
            f"making it a {'strong' if pct >= 60 else 'reasonable'} candidate for cultivation."
        )

    #  Crop Result Banner 
    st.markdown(f"""
        <div style="background:{PANEL}; border-left:5px solid {LEAF};
                    border-radius:10px; padding:28px 32px; margin-bottom:28px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);">
            <p style="color:{MUTED}; font-size:0.7em; letter-spacing:0.2em;
                      text-transform:uppercase; margin:0 0 6px 0;">
                Recommended Crop
            </p>
            <h1 style="font-family:'Playfair Display',serif; color:{LEAF};
                       font-size:2.6em; font-weight:800; margin:0; letter-spacing:0.02em;">
                {crop_emoji}&nbsp; {pred_crop}
            </h1>
            <p style="color:{MUTED}; font-size:0.88em; margin:10px 0 0 0; max-width:580px;">
                Based on your soil and climate inputs, the model determined that
                <strong style="color:{INK};">{pred_crop}</strong> is the most suitable crop
                for your conditions among those evaluated. See below for where in Plateau State
                this crop is best cultivated, and why each factor mattered.
            </p>
        </div>
    """, unsafe_allow_html=True)

    #  LGA Cards 
    st.markdown(f"""
        <p style="color:{MUTED}; font-size:0.7em; letter-spacing:0.18em;
                  text-transform:uppercase; margin:0 0 14px 0;">
            Best Matching Locations — Plateau State LGAs
        </p>
    """, unsafe_allow_html=True)

    rank_colors  = [LEAF, SOIL_LITE, MUTED]
    rank_labels  = ["🥇 Best Match", "🥈 Second Match", "🥉 Third Match"]

    cols = st.columns(3)
    for i, (col, lga) in enumerate(zip(cols, top_lgas)):
        pct, matched = lga_results[lga]
        climate = LGA_CLIMATE[lga]
        explanation = build_lga_explanation(lga, pct, matched)
        fill_pct = pct

        with col:
            st.markdown(f"""
                <div style="background:{PANEL}; border:1.5px solid {BORDER};
                            border-top:4px solid {rank_colors[i]};
                            border-radius:10px; padding:22px 20px 20px 20px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.05); height:100%;">
                    <p style="color:{rank_colors[i]}; font-size:0.68em; font-weight:700;
                              letter-spacing:0.14em; text-transform:uppercase; margin:0 0 4px 0;">
                        {rank_labels[i]}
                    </p>
                    <h3 style="color:{INK}; margin:0 0 4px 0; font-size:1.25em;
                               font-family:'Playfair Display',serif;">
                        {lga}
                    </h3>
                    <p style="color:{MUTED}; margin:0 0 10px 0; font-size:0.8em;
                              font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">
                        {climate} · {pct}% match
                    </p>
                    <!-- Match bar -->
                    <div style="background:{BG}; border-radius:4px; height:6px; margin-bottom:14px;">
                        <div style="background:{rank_colors[i]}; width:{fill_pct}%;
                                    height:6px; border-radius:4px;"></div>
                    </div>
                    <p style="color:#5a5248; font-size:0.82em; line-height:1.6; margin:0;">
                        {explanation}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:36px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)

    #  Charts ─
    st.markdown(f"""
        <p style="color:{MUTED}; font-size:0.7em; letter-spacing:0.18em;
                  text-transform:uppercase; margin:0 0 20px 0;">
            Understanding Your Results
        </p>
    """, unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    #  Chart 1: Parameter Comparison Bars (replaces radar) 
    with chart_col1:
        st.markdown(f"""
            <p style="color:{INK}; font-size:0.92em; font-weight:600;
                      margin:0 0 6px 0;">
                📊 Your Soil vs. Ideal Range for {pred_crop}
            </p>
            <p style="color:{MUTED}; font-size:0.8em; margin:0 0 14px 0; line-height:1.55;">
                Each bar shows where your input sits within the recommended range for {pred_crop}.
                A <strong style="color:{LEAF};">green fill</strong> means you're within range;
                <strong style="color:#c0392b;">red</strong> means outside the ideal.
            </p>
        """, unsafe_allow_html=True)

        params = ["N", "P", "K", "pH", "Temp", "Rainfall", "Salinity"]
        user_vals = [user_N, user_P, user_K, user_pH, user_Temp, user_Rainfall, user_Salinity]

        fig, axes = plt.subplots(len(params), 1, figsize=(5.2, 6.0))
        fig.patch.set_facecolor(PANEL)
        fig.subplots_adjust(hspace=0.55)

        for ax, param, val in zip(axes, params, user_vals):
            lo  = CROP_CONFIG[pred_crop][param]["min"]
            hi  = CROP_CONFIG[pred_crop][param]["max"]
            mid = CROP_CONFIG[pred_crop][param]["mean"]
            unit = UNITS[param]
            in_range = lo <= val <= hi

            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw the range track
            ax.barh(0.5, 1.0, height=0.28, color="#ede9e0", left=0, align="center")

            # Normalize positions
            span = hi - lo if hi != lo else 1
            val_norm = max(0, min(1, (val - lo) / span))

            # Draw the ideal zone (mean ± std)
            std = CROP_CONFIG[pred_crop][param]["std"]
            ideal_lo = max(0, (mid - std - lo) / span)
            ideal_hi = min(1, (mid + std - lo) / span)
            ax.barh(0.5, ideal_hi - ideal_lo, height=0.28,
                    color="#c8e6c9", left=ideal_lo, align="center")

            # Draw user value dot
            bar_color = LEAF if in_range else "#c0392b"
            ax.plot(val_norm, 0.5, "o", color=bar_color, markersize=9, zorder=5)
            ax.plot(val_norm, 0.5, "o", color="white", markersize=4, zorder=6)

            # Range labels
            ax.text(0.0, 0.05, f"{lo}", ha="left",  va="bottom",
                    fontsize=6.5, color=MUTED, transform=ax.transData)
            ax.text(1.0, 0.05, f"{hi}", ha="right", va="bottom",
                    fontsize=6.5, color=MUTED, transform=ax.transData)

            # Param label and user value
            status = "✓" if in_range else "✗"
            ax.text(-0.02, 0.5, f"{param} ({unit})", ha="right", va="center",
                    fontsize=8, color=INK, fontweight="600",
                    transform=ax.transData)
            ax.text(1.02, 0.5, f"{val} {status}", ha="left", va="center",
                    fontsize=8, color=bar_color, fontweight="700",
                    transform=ax.transData)

        legend_elements = [
            mpatches.Patch(color="#c8e6c9", label="Ideal zone (mean ± std)"),
            mpatches.Patch(color=LEAF,      label="Your value — in range"),
            mpatches.Patch(color="#c0392b", label="Your value — out of range"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   fontsize=7.5, ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout(rect=[0.12, 0.04, 0.88, 1.0])
        st.pyplot(fig)
        plt.close()

    #  Chart 2: Simplified Feature Importance (replaces SHAP waterfall) 
    with chart_col2:
        st.markdown(f"""
            <p style="color:{INK}; font-size:0.92em; font-weight:600; margin:0 0 6px 0;">
                 Which Factors Influenced This Recommendation?
            </p>
            <p style="color:{MUTED}; font-size:0.8em; margin:0 0 14px 0; line-height:1.55;">
                The bars show how much each soil or climate factor <em>pushed</em> the model
                toward recommending {pred_crop}. Larger bars = more influence on this result.
            </p>
        """, unsafe_allow_html=True)

        with st.spinner("Calculating feature influence..."):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            pred_class  = model.predict(input_scaled)[0]

            shap_vals   = shap_values[0, :, pred_class]
            feat_names  = features

            # Sort by absolute impact
            order       = np.argsort(np.abs(shap_vals))
            sorted_feats = [feat_names[i] for i in order]
            sorted_vals  = [shap_vals[i]  for i in order]
            bar_colors   = [LEAF if v > 0 else SOIL for v in sorted_vals]

            fig2, ax2 = plt.subplots(figsize=(5.2, 5.0))
            fig2.patch.set_facecolor(PANEL)
            ax2.set_facecolor(PANEL)

            bars = ax2.barh(sorted_feats, [abs(v) for v in sorted_vals],
                            color=bar_colors, height=0.55, edgecolor="none")

            # Value labels on bars
            for bar, val in zip(bars, sorted_vals):
                direction = "helped ↑" if val > 0 else "lowered ↓"
                x_pos = bar.get_width() + max([abs(v) for v in sorted_vals]) * 0.02
                ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                         direction, va="center", ha="left",
                         fontsize=7.5, color=LEAF if val > 0 else SOIL,
                         fontweight="600")

            ax2.set_xlabel("Influence on recommendation", fontsize=8.5, color=MUTED, labelpad=8)
            ax2.tick_params(axis="y", labelsize=9, colors=INK)
            ax2.tick_params(axis="x", labelsize=8,  colors=MUTED)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            ax2.spines["left"].set_color(BORDER)
            ax2.spines["bottom"].set_color(BORDER)
            ax2.set_xlim(0, max([abs(v) for v in sorted_vals]) * 1.35)

            legend_elements2 = [
                mpatches.Patch(color=LEAF, label="Supported this recommendation"),
                mpatches.Patch(color=SOIL, label="Worked against it"),
            ]
            ax2.legend(handles=legend_elements2, fontsize=7.5, frameon=False,
                       loc="lower right", labelcolor=INK)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    #  Bottom note 
    st.markdown(f"""
        <div style="margin-top:28px; background:{PANEL}; border:1px solid {BORDER};
                    border-radius:8px; padding:18px 24px;">
            <p style="color:{MUTED}; font-size:0.82em; line-height:1.65; margin:0;">
                <strong style="color:{INK};">How to read this:</strong>
                The bar chart on the left shows each soil/climate parameter as a track —
                the light green zone is the ideal range, and the dot is your value.
                The chart on the right shows which factors most influenced the model's crop choice,
                and whether they supported or worked against the {pred_crop} recommendation.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:36px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)

    #  Granular LGA Analysis (Suitability Table) ─
    st.markdown(f"""
        <p style="color:{MUTED}; font-size:0.7em; letter-spacing:0.18em;
                  text-transform:uppercase; margin:0 0 14px 0;">
            Granular Analysis — Location Suitability Detail
        </p>
    """, unsafe_allow_html=True)

    # LGA selector dropdown directly in the main panel
    default_lga_index = sorted(list(LGAs)).index(top_lgas[0]) if top_lgas[0] in LGAs else 0
    selected_lga = st.selectbox(
        "Select Location (LGA) to analyze baseline suitability:", 
        sorted(list(LGAs)), 
        index=default_lga_index
    )

    # Calculate typical values for the user's selected LGA using the state baseline
    selected_climate = LGA_CLIMATE[selected_lga]
    selected_adj = CLIMATE_ADJUSTMENTS[selected_climate]
    selected_bias = LGA_SOIL_BIAS[selected_lga]

    typical_vals = {}
    for param in features:
        mean_val = STATE_BASELINE[param]
        mean_val += selected_adj.get(param, 0.0)
        mean_val += selected_bias.get(param, 0.0)
        typical_vals[param] = round(mean_val, 2)

    # Build HTML rows for the suitability table (comparing Typical LGA vs Crop Ideal)
    table_rows = ""
    for param in features:
        typ_val = typical_vals[param]
        lo = CROP_CONFIG[pred_crop][param]["min"]
        hi = CROP_CONFIG[pred_crop][param]["max"]
        unit = UNITS[param]

        if typ_val < lo:
            status_html = f"<span style='color:#c0392b; font-weight:bold;'>Deficient ✗ (Needs +{round(lo - typ_val, 2)} {unit})</span>"
        elif typ_val > hi:
            status_html = f"<span style='color:#d35400; font-weight:bold;'>Excessive ✗ (Over by {round(typ_val - hi, 2)} {unit})</span>"
        else:
            status_html = f"<span style='color:{LEAF}; font-weight:bold;'>Optimal ✓</span>"

        table_rows += f"""<tr style="border-bottom: 1px solid {BORDER};">
<td style="padding: 10px 8px; color:{INK}; font-weight: 500;">{param} ({unit})</td>
<td style="padding: 10px 8px; color:{MUTED};">{typ_val}</td>
<td style="padding: 10px 8px; color:{MUTED};">{lo} - {hi}</td>
<td style="padding: 10px 8px;">{status_html}</td>
</tr>"""

    html_str = f"""
<div style="background:{PANEL}; border:1.5px solid {BORDER};
            border-radius:10px; padding:24px 28px; margin-bottom:28px;
            box-shadow:0 2px 8px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {BORDER}; padding-bottom: 12px; margin-bottom: 16px;">
        <h3 style="margin: 0; color: {INK}; font-family: 'Playfair Display', serif; font-size: 1.4em;">
            📍 Suitability Report for {selected_lga} LGA
        </h3>
        <span style="background: {BG}; color: {SOIL}; font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 4px 10px; border-radius: 20px;">
            {selected_climate} Zone
        </span>
    </div>
    <p style="color:{MUTED}; font-size:0.85em; margin:0 0 16px 0; line-height: 1.55;">
        {LGA_DESCRIPTION[selected_lga]} Here is how the typical soil/climate baseline of <strong>{selected_lga}</strong> compares to the ideal ranges for <strong>{pred_crop}</strong>.
    </p>
    <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 0.88em;">
        <thead>
            <tr style="border-bottom: 2px solid {BORDER}; color: {INK}; font-weight: 600;">
                <th style="padding: 8px;">Parameter</th>
                <th style="padding: 8px;">Typical {selected_lga}</th>
                <th style="padding: 8px;">Ideal {pred_crop} Range</th>
                <th style="padding: 8px;">LGA Status</th>
            </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
    </table>
</div>
"""
    st.markdown(html_str, unsafe_allow_html=True)

    # Build structured text report with ASCII table
    report_text = f"""PLATEAU STATE CROP ADVISOR REPORT
================================================================================
RECOMMENDED CROP: {pred_crop.upper()}
Selected LGA Suitability: {selected_lga} ({selected_climate} Zone)

Description:
{LGA_DESCRIPTION[selected_lga]}

SUITABILITY ANALYSIS FOR {selected_lga} LGA:
+-------------------+-------------------+-------------------+---------------------------+
| Parameter         | Typical LGA Value | Ideal Crop Range  | Suitability Status        |
+-------------------+-------------------+-------------------+---------------------------+
"""
    for param in features:
        typ_val = typical_vals[param]
        lo = CROP_CONFIG[pred_crop][param]["min"]
        hi = CROP_CONFIG[pred_crop][param]["max"]
        unit = UNITS[param]

        param_str = f"{param} ({unit})"
        typ_str = f"{typ_val}"
        range_str = f"{lo} - {hi}"

        status = "Optimal"
        if typ_val < lo:
            status = f"Deficient (Needs +{round(lo - typ_val, 2)})"
        elif typ_val > hi:
            status = f"Excessive (Over by {round(typ_val - hi, 2)})"

        report_text += f"| {param_str:<17} | {typ_str:<17} | {range_str:<17} | {status:<25} |\n"

    report_text += f"""+-------------------+-------------------+-------------------+---------------------------+

TOP 3 GENERAL BEST-MATCHING LGAS FOR {pred_crop.upper()}:
"""
    for rank, lga in enumerate(top_lgas, 1):
        pct, matched = lga_results[lga]
        report_text += f"{rank}. {lga:<18} ({LGA_CLIMATE[lga]} Zone) - {pct}% match (Key matches: {', '.join(matched) if matched else 'none'})\n"

    report_text += "\nReport generated by Plateau Crop Advisor AI. All values are advisory based on machine learning modeling.\n"
    report_text += "================================================================================"

    col_dl, col_space = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="📥 Download Advisor Report",
            data=report_text,
            file_name=f"crop_advisor_report_{pred_crop.lower()}.txt",
            mime="text/plain",
            use_container_width=True
        )
    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)