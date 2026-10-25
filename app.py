
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Marine Pollution AI",
    page_icon="🌊",
    layout="wide"
)

BASE = Path(__file__).parent

st.markdown("""
<style>
.stApp {background:#071B33;color:#EAF8FF}
h1,h2,h3 {color:#39D5BB}
[data-testid="stMetric"] {
    background:#102D4A;
    padding:18px;
    border-radius:12px;
    border:1px solid #39D5BB;
}
</style>
""", unsafe_allow_html=True)

st.title("🌊 AI-Based Marine Pollution Prediction")
st.caption("CatBoost • Explainable AI • GIS Hotspots • Spatial Risk Assessment")

def load_csv(filename):
    path = BASE / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

summary = load_csv("dashboard_summary.csv")
predictions = load_csv("gis_predictions.csv")
shap_data = load_csv("top_10_shap_features.csv")
hotspots = load_csv("hotspot_results.csv")
ranking = load_csv("hotspot_ranking.csv")
comparison = load_csv("final_model_comparison.csv")
risk = load_csv("prediction_risk_results.csv")

if not summary.empty:
    st.subheader("📊 Project Summary")
    st.dataframe(summary, use_container_width=True)

if not predictions.empty:
    st.subheader("🗺️ GIS Pollution Predictions")
    st.dataframe(predictions.head(100), use_container_width=True)
    numeric = predictions.select_dtypes(include="number")
    if {"Latitude", "Longitude"}.issubset(predictions.columns):
        st.map(predictions.rename(columns={
            "Latitude": "latitude", "Longitude": "longitude"
        }).dropna(subset=["Latitude", "Longitude"]))

if not risk.empty:
    st.subheader("🚦 Pollution Risk Assessment")
    st.dataframe(risk.head(100), use_container_width=True)

if not hotspots.empty:
    st.subheader("🔥 Pollution Hotspots")
    st.dataframe(hotspots.head(100), use_container_width=True)

if not ranking.empty:
    st.subheader("📍 Hotspot Ranking")
    st.dataframe(ranking.head(20), use_container_width=True)

if not shap_data.empty:
    st.subheader("🔎 Top SHAP Features")
    st.dataframe(shap_data.head(10), use_container_width=True)
    st.bar_chart(shap_data.set_index(shap_data.columns[0]).select_dtypes("number"))

if not comparison.empty:
    st.subheader("🤖 Model Performance Comparison")
    st.dataframe(comparison, use_container_width=True)

st.subheader("📁 Available Project Files")
for filename in [
    "catboost_model.cbm", "shap_importance.csv",
    "test_data.csv", "actual_values.csv",
    "marine_pollution_risk_map.html"
]:
    st.write(("✅ " if (BASE / filename).exists() else "⚠️ ") + filename)

st.success("Dashboard loaded. Existing saved results are used; no model retraining.")
