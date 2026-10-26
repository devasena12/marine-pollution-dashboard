import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# ============================================================
# MARINE POLLUTION INTELLIGENCE DASHBOARD
# Replace your existing app.py with this complete file.
# Keep your CSV/model files in the same folder as app.py.
# ============================================================

st.set_page_config(
    page_title="Marine Pollution Intelligence V2",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dark ocean-blue theme
st.markdown("""
<style>
.stApp {
 background: radial-gradient(ellipse at 10% 0%, rgba(34,211,238,.22), transparent 34%),
 linear-gradient(145deg,#020617 0%,#0b1738 48%,#102c4a 100%);
 color:#eafaff;
}
[data-testid="stHeader"] {background:rgba(4,17,31,.85);}
[data-testid="stSidebar"] {
 background:linear-gradient(180deg,#03101d 0%,#08283b 100%);
 border-right:1px solid rgba(52,211,214,.25);
}
h1,h2,h3,p,label,span {color:#eafaff;}
.eyebrow {color:#50e0dc;font-size:.75rem;font-weight:800;letter-spacing:2px;}
.marine-subtitle {color:#7ddfe4;font-size:1rem;margin-top:-8px;}
.metric-card {
 background:linear-gradient(145deg,rgba(11,52,72,.98),rgba(5,27,43,.98));
 border:1px solid rgba(34,211,238,.60);border-radius:16px;
 padding:18px 19px;min-height:125px;box-shadow:0 8px 26px rgba(0,0,0,.15);
}
.metric-label {color:#a8cbd6;font-size:.79rem;margin-bottom:9px;letter-spacing:.4px;}
.metric-value {color:#f4ffff;font-size:1.7rem;line-height:1.2;font-weight:750;overflow-wrap:anywhere;}
.metric-foot {color:#62d9d4;font-size:.74rem;margin-top:7px;}
.panel {background:rgba(10,25,55,.88);border:1px solid rgba(56,189,248,.42);border-radius:16px;padding:16px 18px;}
.section-kicker {color:#62d9d4;font-size:.73rem;font-weight:800;letter-spacing:1.6px;}
.stButton button,.stDownloadButton button {
 background:linear-gradient(90deg,#22d3ee,#a3e635);color:#04111f;
 border:0;border-radius:9px;font-weight:750;
}
div[data-testid="stDataFrame"] {border:1px solid rgba(89,202,211,.45);border-radius:10px;}
/* Force readable text in Streamlit's small metric/info containers */
div[data-testid="stMetric"] {background:linear-gradient(145deg,#092f45,#061d30)!important;border:1px solid #1b8292!important;border-radius:14px!important;padding:16px!important;min-height:112px!important;}
div[data-testid="stMetricLabel"],div[data-testid="stMetricLabel"] p {color:#c5f3ff!important;font-size:.95rem!important;font-weight:700!important;opacity:1!important;}
div[data-testid="stMetricValue"],div[data-testid="stMetricValue"] div {color:#ffffff!important;font-size:1.65rem!important;font-weight:850!important;opacity:1!important;}
div[data-testid="stMetricDelta"] {color:#5eead4!important;}
div[data-testid="stAlert"] {background:#092d43!important;color:#ffffff!important;border:1px solid #21a6b4!important;border-radius:12px!important;}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {color:#ffffff!important;opacity:1!important;}
.stCaption, [data-testid="stCaptionContainer"] {color:#c2e8f2!important;}
[data-testid="stProgressBar"] div[role="progressbar"] {background:#12384b!important;}
[data-testid="stDataFrame"] * {color:#f4fbff!important;}
</style>
""", unsafe_allow_html=True)

# ---------- File helpers ----------
def find_file(candidates):
    for filename in candidates:
        path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(path):
            return path
    return None

@st.cache_data(show_spinner=False)
def read_csv(path):
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

GIS_PATH = find_file([
    "gis_predictions.csv", "marine_pollution_GIS_results.csv",
    "hotspot_results.csv", "FINAL_MARINE_POLLUTION_RESULTS.csv"
