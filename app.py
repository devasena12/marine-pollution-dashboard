
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Marine Pollution AI Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

# --------------------------------------------------
# THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071B33, #0B2945);
    color: #EAF8FF;
}

h1, h2, h3 {
    color: #39D5BB !important;
}

p, label {
    color: #EAF8FF;
}

[data-testid="stSidebar"] {
    background: #EAF8FF !important;
    border-right: 3px solid #39D5BB;
}

[data-testid="stSidebar"] > div {
    background: #EAF8FF !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #071B33 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #FFFFFF !important;
    border: 1px solid #B5D9E8 !important;
    border-radius: 9px !important;
    padding: 10px 12px !important;
    margin-bottom: 4px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {
    color: #071B33 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: #39D5BB !important;
    border: 2px solid #087F70 !important;
}

[data-testid="stMetric"] {
    background: #102D4A;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #39D5BB;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #EAF8FF !important;
}

.stButton > button {
    background-color: #39D5BB;
    color: #071B33;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

[data-testid="stDataFrame"] {
    border: 1px solid #234D6B;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌊 AI-Based Marine Pollution Prediction")
st.markdown(
    "### Explainable AI • CatBoost • GIS Hotspots • Spatial Risk Assessment"
)
st.divider()

# --------------------------------------------------
# CSV LOADER
# --------------------------------------------------
def load_csv(filename):
    path = BASE / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Unable to read {filename}: {error}")
        return pd.DataFrame()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
summary = load_csv("dashboard_summary.csv")
predictions = load_csv("gis_predictions.csv")
shap_data = load_csv("top_10_shap_features.csv")
hotspots = load_csv("hotspot_results.csv")
ranking = load_csv("hotspot_ranking.csv")
comparison = load_csv("final_model_comparison.csv")
risk = load_csv("prediction_risk_results.csv")

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🌊 Marine AI")
st.sidebar.markdown("### Dashboard Navigation")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "GIS Pollution Map",
        "Risk Assessment",
        "Pollution Hotspots",
        "SHAP Explainability",
        "Model Comparison",
        "Project Files"
    ],
    key="dashboard_navigation"
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if section == "Overview":
    st.header("📊 Project Overview")

    if not summary.empty:
        st.subheader("Project Summary")
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Project summary data is unavailable.")

    if not predictions.empty:
        st.subheader("Prediction Preview")
        st.dataframe(predictions.head(10), use_container_width=True)

# --------------------------------------------------
# GIS POLLUTION MAP
# --------------------------------------------------
elif section == "GIS Pollution Map":
    st.header("🗺️ GIS-Based Pollution Mapping")

    if predictions.empty:
        st.warning("GIS prediction data is unavailable.")
    else:
        st.dataframe(predictions.head(100), use_container_width=True)

        columns = {
            str(c).strip().lower(): c for c in predictions.columns
        }

        lat_col = next(
            (columns[n] for n in ["latitude", "lat"] if n in columns),
            None
        )
        lon_col = next(
            (columns[n] for n in ["longitude", "lon", "long"] if n in columns),
            None
        )

        if lat_col is not None and lon_col is not None:
            map_data = predictions[[lat_col, lon_col]].copy()
            map_data.columns = ["latitude", "longitude"]

            map_data["latitude"] = pd.to_numeric(
                map_data["latitude"], errors="coerce"
            )
            map_data["longitude"] = pd.to_numeric(
                map_data["longitude"], errors="coerce"
            )

            map_data = map_data.dropna(
                subset=["latitude", "longitude"]
            )

            map_data = map_data[
                map_data["latitude"].between(-90, 90)
                & map_data["longitude"].between(-180, 180)
            ]

            if not map_data.empty:
                st.subheader("Pollution Locations")
                st.map(map_data)
            else:
                st.info("No valid geographic coordinates were found.")
        else:
            st.warning(
                "Latitude and longitude columns are missing from this dataset."
            )

    html_path = BASE / "marine_pollution_risk_map.html"

    if html_path.exists():
        with st.expander("Open Saved Interactive Risk Map"):
            try:
                st.components.v1.html(
                    html_path.read_text(encoding="utf-8"),
                    height=600,
                    scrolling=True
                )
            except Exception as error:
                st.warning(f"Unable to display map: {error}")

# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------
elif section == "Risk Assessment":
    st.header("🚦 Pollution Risk Assessment")

    if not risk.empty:
        st.dataframe(risk.head(200), use_container_width=True)

        columns = {
            str(c).strip().lower(): c for c in risk.columns
        }

        risk_col = next(
            (
                columns[n] for n in
                ["risk_level", "risk_category", "pollution_risk", "risk"]
                if n in columns
            ),
            None
        )

        if risk_col:
            counts = (
                risk[risk_col].fillna("Unknown").astype(str).value_counts()
            )
            st.subheader("Risk Category Distribution")
            st.bar_chart(counts)
    else:
        st.warning("Risk assessment data is unavailable.")

# --------------------------------------------------
# POLLUTION HOTSPOTS
# --------------------------------------------------
elif section == "Pollution Hotspots":
    st.header("🔥 Marine Pollution Hotspots")

    if not hotspots.empty:
        st.subheader("Hotspot Results")
        st.dataframe(hotspots.head(100), use_container_width=True)
    else:
        st.info("Hotspot results are unavailable.")

    if not ranking.empty:
        st.subheader("📍 Hotspot Ranking")
        st.dataframe(ranking.head(20), use_container_width=True)

# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------
elif section == "SHAP Explainability":
    st.header("🔎 SHAP Feature Importance")

    if not shap_data.empty:
        st.dataframe(shap_data.head(10), use_container_width=True)

        feature_col = shap_data.columns[0]
        numeric_cols = shap_data.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols:
            value_col = st.selectbox(
                "Select importance metric", numeric_cols
            )

            chart_data = shap_data[
                [feature_col, value_col]
            ].dropna()

            if not chart_data.empty:
                st.bar_chart(chart_data.set_index(feature_col))
        else:
            st.info("No numeric importance values were found.")
    else:
        st.warning("SHAP feature data is unavailable.")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif section == "Model Comparison":
    st.header("🤖 Model Performance Comparison")

    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)

        numeric_data = comparison.select_dtypes(include="number")
        if not numeric_data.empty:
            st.subheader("Performance Metrics")
            st.bar_chart(numeric_data)
    else:
        st.warning("Model comparison data is unavailable.")

# --------------------------------------------------
# PROJECT FILES
# --------------------------------------------------
elif section == "Project Files":
    st.header("📁 Project Files")

    filenames = [
        "dashboard_summary.csv",
        "gis_predictions.csv",
        "top_10_shap_features.csv",
        "hotspot_results.csv",
        "hotspot_ranking.csv",
        "prediction_risk_results.csv",
        "final_model_comparison.csv",
        "catboost_model.cbm",
        "shap_importance.csv",
        "test_data.csv",
        "actual_values.csv",
        "FINAL_MARINE_POLLUTION_RESULTS.csv",
        "marine_pollution_risk_map.html"
    ]

    status = [
        {
            "File": filename,
            "Status": (
                "✅ Available"
                if (BASE / filename).exists()
                else "⚠️ Not found"
            )
        }
        for filename in filenames
    ]

    st.dataframe(
        pd.DataFrame(status),
        use_container_width=True,
        hide_index=True
    )

st.divider()
st.caption(
    "Marine Pollution AI Dashboard | CatBoost • SHAP • GIS • Risk Assessment"
)
