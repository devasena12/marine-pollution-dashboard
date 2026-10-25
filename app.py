
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
# DEEP OCEAN BLUE THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071B33, #0B2945);
    color: #EAF8FF;
}

[data-testid="stHeader"] {
    background: rgba(7, 27, 51, 0.95);
}

h1, h2, h3 {
    color: #39D5BB !important;
}

p, label, span {
    color: #EAF8FF;
}

[data-testid="stMetric"] {
    background: #102D4A;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #39D5BB;
}

[data-testid="stDataFrame"] {
    border: 1px solid #234D6B;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    color: #EAF8FF;
}

.stTabs [aria-selected="true"] {
    color: #39D5BB !important;
}

div.stButton > button {
    background-color: #39D5BB;
    color: #071B33;
    border-radius: 8px;
    border: none;
    font-weight: bold;
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
    file_path = BASE / filename

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return pd.DataFrame()


# --------------------------------------------------
# LOAD SAVED RESULTS
# --------------------------------------------------
summary = load_csv("dashboard_summary.csv")
predictions = load_csv("gis_predictions.csv")
shap_data = load_csv("top_10_shap_features.csv")
hotspots = load_csv("hotspot_results.csv")
ranking = load_csv("hotspot_ranking.csv")
comparison = load_csv("final_model_comparison.csv")
risk = load_csv("prediction_risk_results.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌊 Navigation")

section = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "GIS Pollution Map",
        "Risk Assessment",
        "Pollution Hotspots",
        "SHAP Explainability",
        "Model Comparison",
        "Project Files"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard displays previously saved project results. "
    "It does not retrain the models."
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if section == "Overview":

    st.header("📊 Project Overview")

    if not summary.empty:
        st.subheader("Saved Project Summary")
        st.dataframe(summary, use_container_width=True)

        # Display useful summary metrics if available
        summary_values = {}

        for _, row in summary.iterrows():
            if len(row) >= 2:
                summary_values[str(row.iloc[0]).strip().lower()] = row.iloc[1]

        def find_summary_value(keywords):
            for key, value in summary_values.items():
                if any(word in key for word in keywords):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        continue
            return None

        total_locations = find_summary_value(
            ["location", "sample", "record", "total"]
        )
        average_pollution = find_summary_value(
            ["average", "mean", "avg"]
        )
        maximum_pollution = find_summary_value(
            ["maximum", "max"]
        )
        hotspot_count = find_summary_value(
            ["hotspot"]
        )

        metrics = st.columns(4)

        with metrics[0]:
            if total_locations is not None:
                st.metric("Total Records", f"{total_locations:,.0f}")
            else:
                st.metric("Prediction Records", len(predictions))

        with metrics[1]:
            if average_pollution is not None:
                st.metric("Average Pollution", f"{average_pollution:,.2f}")
            else:
                st.metric("Prediction Data", "Available" if not predictions.empty else "Not found")

        with metrics[2]:
            if maximum_pollution is not None:
                st.metric("Maximum Pollution", f"{maximum_pollution:,.2f}")
            else:
                st.metric("Hotspot Records", len(hotspots))

        with metrics[3]:
            if hotspot_count is not None:
                st.metric("Hotspot Count", f"{hotspot_count:,.0f}")
            else:
                st.metric("Risk Records", len(risk))

    else:
        st.info("dashboard_summary.csv is missing or empty.")

    st.subheader("📌 Prediction Preview")

    if not predictions.empty:
        st.dataframe(predictions.head(10), use_container_width=True)
    else:
        st.warning("gis_predictions.csv is not available.")

# --------------------------------------------------
# GIS POLLUTION MAP
# --------------------------------------------------
elif section == "GIS Pollution Map":

    st.header("🗺️ GIS-Based Pollution Mapping")

    if predictions.empty:
        st.warning("No GIS prediction data was found.")
    else:
        st.subheader("Saved Prediction Records")
        st.dataframe(predictions.head(100), use_container_width=True)

        # Detect common latitude/longitude column names
        column_lookup = {
            str(column).strip().lower(): column
            for column in predictions.columns
        }

        latitude_column = next(
            (
                column_lookup[name]
                for name in ["latitude", "lat"]
                if name in column_lookup
            ),
            None
        )

        longitude_column = next(
            (
                column_lookup[name]
                for name in ["longitude", "lon", "long"]
                if name in column_lookup
            ),
            None
        )

        if latitude_column is not None and longitude_column is not None:

            map_data = predictions[
                [latitude_column, longitude_column]
            ].copy()

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
                st.subheader("Pollution Sampling Locations")
                st.map(map_data)
                st.caption(
                    f"Map displays {len(map_data)} records with valid coordinates."
                )
            else:
                st.info(
                    "Latitude and longitude columns were found, "
                    "but no valid coordinates are available."
                )

        else:
            st.warning(
                "Latitude/Longitude columns were not found in "
                "gis_predictions.csv. The table is shown above, "
                "but a geographic map requires actual coordinate columns."
            )

        # Show a previously saved interactive map if present
        html_map = BASE / "marine_pollution_risk_map.html"

        if html_map.exists():
            st.subheader("Saved Interactive Pollution Risk Map")

            with st.expander("Show saved HTML map"):
                try:
                    html_content = html_map.read_text(
                        encoding="utf-8"
                    )

                    st.components.v1.html(
                        html_content,
                        height=600,
                        scrolling=True
                    )
                except Exception as error:
                    st.warning(
                        f"Could not display saved HTML map: {error}"
                    )

# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------
elif section == "Risk Assessment":

    st.header("🚦 Marine Pollution Risk Assessment")

    if not risk.empty:
        st.dataframe(risk.head(200), use_container_width=True)

        # Display risk category counts when a suitable column exists
        risk_columns = {
            str(column).strip().lower(): column
            for column in risk.columns
        }

        risk_column = next(
            (
                risk_columns[name]
                for name in [
                    "risk_level",
                    "risk_category",
                    "pollution_risk",
                    "risk"
                ]
                if name in risk_columns
            ),
            None
        )

        if risk_column is not None:
            st.subheader("Risk Category Distribution")
            risk_counts = (
                risk[risk_column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )
            st.bar_chart(risk_counts)

    else:
        st.warning("prediction_risk_results.csv is missing or empty.")

# --------------------------------------------------
# POLLUTION HOTSPOTS
# --------------------------------------------------
elif section == "Pollution Hotspots":

    st.header("🔥 Marine Pollution Hotspots")

    if not hotspots.empty:
        st.subheader("Hotspot Results")
        st.dataframe(hotspots.head(100), use_container_width=True)
    else:
        st.info("hotspot_results.csv is not available.")

    if not ranking.empty:
        st.subheader("📍 Hotspot Ranking")
        st.dataframe(ranking.head(20), use_container_width=True)
    else:
        st.info("hotspot_ranking.csv is not available.")

# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------
elif section == "SHAP Explainability":

    st.header("🔎 Explainable AI — SHAP Feature Importance")

    if not shap_data.empty:

        st.subheader("Top 10 Important Features")
        st.dataframe(shap_data.head(10), use_container_width=True)

        # Use the first column as feature labels
        feature_column = shap_data.columns[0]

        numeric_columns = shap_data.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_column = st.selectbox(
                "Select feature importance value",
                numeric_columns
            )

            chart_data = shap_data[
                [feature_column, selected_column]
            ].copy()

            chart_data = chart_data.dropna()

            chart_data = chart_data.set_index(feature_column)

            st.subheader("Feature Importance Chart")
            st.bar_chart(chart_data)

        else:
            st.info(
                "No numeric SHAP importance column was found."
            )

    else:
        st.warning("top_10_shap_features.csv is missing or empty.")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif section == "Model Comparison":

    st.header("🤖 Classical and Hybrid Model Comparison")

    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)

        numeric_comparison = comparison.select_dtypes(
            include="number"
        )

        if not numeric_comparison.empty:
            st.subheader("Numeric Performance Metrics")
            st.bar_chart(numeric_comparison)
    else:
        st.warning("final_model_comparison.csv is missing or empty.")

# --------------------------------------------------
# PROJECT FILES
# --------------------------------------------------
elif section == "Project Files":

    st.header("📁 Project Files")

    files_to_check = [
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

    file_status = []

    for filename in files_to_check:
        file_path = BASE / filename

        file_status.append({
            "File": filename,
            "Status": (
                "Available"
                if file_path.exists()
                else "Not found"
            )
        })

    st.dataframe(
        pd.DataFrame(file_status),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Marine Pollution AI Dashboard | "
    "CatBoost • SHAP • GIS • Hotspot Detection • Risk Assessment"
)

st.success(
    "Dashboard is using saved project results. "
    "Model retraining is not performed on this page."
)
