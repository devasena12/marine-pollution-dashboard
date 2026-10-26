
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Marine Pollution AI Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

# --------------------------------------------------
# THEME: HIGH-CONTRAST SIDEBAR + DEEP OCEAN MAIN PAGE
# --------------------------------------------------
st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background: linear-gradient(135deg, #071B33, #0B2945);
    color: #EAF8FF;
}

/* Main page headings */
h1, h2, h3 {
    color: #39D5BB !important;
}

p, label {
    color: #EAF8FF;
}

/* Sidebar: light background and dark text */
[data-testid="stSidebar"] {
    background: #EAF8FF !important;
    border-right: 3px solid #39D5BB;
}

[data-testid="stSidebar"] > div {
    background: #EAF8FF !important;
}

/* All sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #071B33 !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}

/* Each navigation option */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #FFFFFF !important;
    border: 1px solid #B5D9E8 !important;
    border-radius: 9px !important;
    padding: 10px 12px !important;
    margin-bottom: 4px !important;
    color: #071B33 !important;
}

/* Navigation option text */
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {
    color: #071B33 !important;
    font-weight: 600 !important;
}

/* Selected navigation option */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: #39D5BB !important;
    border: 2px solid #087F70 !important;
}

/* Sidebar information box */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #D5F5EE !important;
    border: 1px solid #39AFA0 !important;
}

/* Metric cards */
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

/* Buttons */
.stButton > button {
    background-color: #39D5BB;
    color: #071B33;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

/* Data tables */
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
st.caption(
    "Marine microplastic pollution prediction using saved machine learning results."
)
st.divider()

# --------------------------------------------------
# SAFE CSV LOADER
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
# LOAD SAVED DATA
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

st.sidebar.markdown("---")
st.sidebar.info(
    "Saved project results are displayed here. "
    "Models are not retrained by this dashboard."
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
        st.info("dashboard_summary.csv is not available.")

    st.subheader("Prediction Preview")

    if not predictions.empty:
        st.dataframe(predictions.head(10), use_container_width=True)
    else:
        st.info("GIS prediction data is not available.")

# --------------------------------------------------
# GIS POLLUTION MAP
# --------------------------------------------------
elif section == "GIS Pollution Map":

    st.header("🗺️ GIS-Based Pollution Mapping")

    if predictions.empty:
        st.warning("gis_predictions.csv is missing or empty.")
    else:
        st.dataframe(predictions.head(100), use_container_width=True)

        column_lookup = {
            str(column).strip().lower(): column
            for column in predictions.columns
        }

        lat_col = next(
            (
                column_lookup[name]
                for name in ["latitude", "lat"]
                if name in column_lookup
            ),
            None
        )

        lon_col = next(
            (
                column_lookup[name]
                for name in ["longitude", "lon", "long"]
                if name in column_lookup
            ),
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
                st.caption(
                    f"Showing {len(map_data)} valid coordinate records."
                )
            else:
                st.info("No valid geographic coordinates were found.")
        else:
            st.warning(
                "Latitude and longitude columns were not found. "
                "Actual coordinates are required to plot locations."
            )

        html_path = BASE / "marine_pollution_risk_map.html"

        if html_path.exists():
            st.subheader("Saved Interactive Risk Map")

            with st.expander("Open saved map"):
                try:
                    st.components.v1.html(
                        html_path.read_text(encoding="utf-8"),
                        height=600,
                        scrolling=True
                    )
                except Exception as error:
                    st.warning(f"Unable to display saved map: {error}")

# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------
elif section == "Risk Assessment":

    st.header("🚦 Pollution Risk Assessment")

    if not risk.empty:
        st.dataframe(risk.head(200), use_container_width=True)

        risk_lookup = {
            str(column).strip().lower(): column
            for column in risk.columns
        }

        risk_col = next(
            (
                risk_lookup[name]
                for name in [
                    "risk_level",
                    "risk_category",
                    "pollution_risk",
                    "risk"
                ]
                if name in risk_lookup
            ),
            None
        )

        if risk_col is not None:
            st.subheader("Risk Category Distribution")
            counts = (
                risk[risk_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )
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
        st.info("No hotspot results file was found.")

    if not ranking.empty:
        st.subheader("📍 Hotspot Ranking")
        st.dataframe(ranking.head(20), use_container_width=True)
    else:
        st.info("No hotspot ranking file was found.")

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
                "Select importance metric",
                numeric_cols
            )

            chart_data = shap_data[
                [feature_col, value_col]
            ].dropna()

            if not chart_data.empty:
                chart_data = chart_data.set_index(feature_col)
                st.subheader("Feature Importance Chart")
                st.bar_chart(chart_data)
        else:
            st.info("No numeric feature importance values were found.")
    else:
        st.warning("top_10_shap_features.csv is missing or empty.")

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
        st.warning("final_model_comparison.csv is missing or empty.")

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

    status = []

    for filename in filenames:
        status.append({
            "File": filename,
            "Status": (
                "✅ Available"
                if (BASE / filename).exists()
                else "⚠️ Not found"
            )
        })

    st.dataframe(
        pd.DataFrame(status),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Marine Pollution AI Dashboard | CatBoost • SHAP • GIS • Risk Assessment"
)
st.success("Dashboard uses saved results. No model retraining is performed.")
