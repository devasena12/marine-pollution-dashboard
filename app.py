
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Marine Pollution AI Dashboard",
    page_icon="🌊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
.stApp {
    background-color: #061629;
    color: #EAF8FF;
}
[data-testid="stSidebar"] {
    background-color: #DDF4FF;
}
[data-testid="stSidebar"] * {
    color: #071B33 !important;
}
h1, h2, h3 {
    color: #39D5BB !important;
}
[data-testid="stMetric"] {
    background-color: #102B45;
    border: 1px solid #24617A;
    padding: 15px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #B9DDF0 !important;
}
[data-testid="stMetricValue"] {
    color: #39D5BB !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD CSV FILES SAFELY
# --------------------------------------------------
def load_csv(filename):
    path = BASE_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


summary = load_csv("dashboard_summary.csv")
gis = load_csv("gis_predictions.csv")
shap_data = load_csv("top_10_shap_features.csv")
importance = load_csv("shap_importance.csv")
hotspots = load_csv("hotspot_results.csv")
hotspot_ranking = load_csv("hotspot_ranking.csv")
risk_data = load_csv("prediction_risk_results.csv")
comparison = load_csv("final_model_comparison.csv")
final_results = load_csv("FINAL_MARINE_POLLUTION_RESULTS.csv")
test_data = load_csv("test_data.csv")
actual_data = load_csv("actual_values.csv")


def find_col(df, keywords):
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() == str(col).lower():
                return col

    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in str(col).lower():
                return col

    return None


def numeric_col(df, keywords):
    col = find_col(df, keywords)
    if col is None:
        return None, pd.Series(dtype=float)

    values = pd.to_numeric(df[col], errors="coerce").dropna()
    return col, values


def show_table(df):
    if df.empty:
        st.info("No data available. Check the project CSV files.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def download_csv(df, filename):
    if not df.empty:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌊 Marine Pollution AI")
st.sidebar.caption("Prediction • GIS • Risk • Explainability")

section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "GIS Pollution Map",
        "Risk Assessment",
        "Marine Plastic Analysis",
        "Pollution Hotspots",
        "SHAP Explainability",
        "Model Comparison",
        "Project Files"
    ]
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if section == "Overview":
    st.title("🌊 Marine Pollution Prediction Dashboard")

    st.write(
        "An Explainable Hybrid Classical–Quantum Machine Learning "
        "Framework for Marine Microplastic Pollution Prediction, "
        "Hotspot Detection and Spatial Risk Assessment."
    )

    location_count = len(gis) if not gis.empty else 0

    pollution_col, pollution_values = numeric_col(
        gis,
        [
            "predicted_abundance",
            "predicted_pollution",
            "prediction",
            "plastic_abundance_converted"
        ]
    )

    average = (
        pollution_values.mean()
        if not pollution_values.empty else None
    )
    maximum = (
        pollution_values.max()
        if not pollution_values.empty else None
    )

    hotspot_count = (
        len(hotspot_ranking)
        if not hotspot_ranking.empty
        else len(hotspots)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Locations Analysed", f"{location_count:,}")
    c2.metric(
        "Average Pollution",
        f"{average:,.2f}" if average is not None else "N/A"
    )
    c3.metric(
        "Maximum Pollution",
        f"{maximum:,.2f}" if maximum is not None else "N/A"
    )
    c4.metric("Hotspot Records", f"{hotspot_count:,}")

    st.divider()
    st.subheader("Project Workflow")

    st.write(
        "Dataset → Preprocessing → Feature Engineering → "
        "CatBoost → Hybrid Quantum-Classical Model → "
        "Prediction → SHAP → GIS Mapping → Hotspot Detection → "
        "Risk Assessment"
    )

    st.subheader("Main Project Features")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        - Marine microplastic abundance analysis
        - Classical ML prediction
        - Quantum-classical model comparison
        - Model evaluation metrics
        """)

    with right:
        st.markdown("""
        - SHAP explainability
        - GIS location visualisation
        - Pollution hotspot detection
        - Pollution risk classification
        """)


# --------------------------------------------------
# GIS POLLUTION MAP
# --------------------------------------------------
elif section == "GIS Pollution Map":
    st.title("🗺️ GIS-Based Marine Pollution Map")

    if gis.empty:
        st.warning("gis_predictions.csv file is unavailable.")
    else:
        lat_col = find_col(gis, ["latitude", "lat"])
        lon_col = find_col(gis, ["longitude", "lon", "lng"])

        if lat_col and lon_col:
            map_df = gis.copy()

            map_df[lat_col] = pd.to_numeric(
                map_df[lat_col], errors="coerce"
            )
            map_df[lon_col] = pd.to_numeric(
                map_df[lon_col], errors="coerce"
            )

            map_df = map_df.dropna(subset=[lat_col, lon_col])

            map_df = map_df[
                map_df[lat_col].between(-90, 90)
                & map_df[lon_col].between(-180, 180)
            ]

            if not map_df.empty:
                map_df = map_df.rename(
                    columns={
                        lat_col: "latitude",
                        lon_col: "longitude"
                    }
                )

                st.map(
                    map_df[["latitude", "longitude"]],
                    latitude="latitude",
                    longitude="longitude"
                )

                st.caption(
                    "Map displays the available sampling coordinates. "
                    "It does not automatically colour locations by risk."
                )
            else:
                st.warning("No valid geographic coordinates found.")
        else:
            st.warning(
                "Latitude/longitude columns were not found in the GIS file."
            )

        st.subheader("GIS Data")
        show_table(gis.head(100))
        download_csv(gis, "gis_predictions.csv")


# --------------------------------------------------
# RISK ASSESSMENT
# --------------------------------------------------
elif section == "Risk Assessment":
    st.title("⚠️ Marine Pollution Risk Assessment")

    source = risk_data if not risk_data.empty else gis

    if source.empty:
        st.warning("Risk assessment CSV file is unavailable.")
    else:
        risk_col = find_col(
            source,
            ["risk_category", "risk_level", "pollution_risk", "risk"]
        )

        if risk_col:
            counts = source[risk_col].dropna().astype(str).value_counts()

            st.subheader("Risk Category Distribution")
            st.bar_chart(counts)

            lookup = {
                str(k).strip().lower(): int(v)
                for k, v in counts.items()
            }

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Low", lookup.get("low", 0))
            c2.metric("Moderate", lookup.get("moderate", 0))
            c3.metric("High", lookup.get("high", 0))
            c4.metric("Critical", lookup.get("critical", 0))

            show_table(source)
            download_csv(source, "prediction_risk_results.csv")
        else:
            st.warning("Risk category column was not found.")
            show_table(source)


# --------------------------------------------------
# MARINE PLASTIC ANALYSIS
# --------------------------------------------------
elif section == "Marine Plastic Analysis":
    st.title("🧪 Marine Plastic Pollution Analysis")

    st.write(
        "Analyse saved plastic abundance values from the project CSV files."
    )

    candidates = [
        ("GIS Predictions", gis),
        ("Final Project Results", final_results),
        ("Risk Results", risk_data),
        ("Test Data", test_data),
        ("Actual Values", actual_data)
    ]

    source_name = None
    source = pd.DataFrame()
    abundance_col = None

    # Prefer explicitly predicted abundance, then measured abundance.
    for name, df in candidates:
        if df.empty:
            continue

        col = find_col(
            df,
            [
                "predicted_abundance",
                "predicted_pollution",
                "plastic_abundance_converted",
                "plastic_abundance",
                "microplastic_abundance",
                "abundance"
            ]
        )

        if col:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                source_name = name
                source = df.copy()
                abundance_col = col
                break

    if source.empty or abundance_col is None:
        st.warning(
            "No plastic abundance column was found in the available files. "
            "Check the column names in your CSV files."
        )
    else:
        values = pd.to_numeric(
            source[abundance_col], errors="coerce"
        ).dropna()

        is_prediction = (
            "predict" in abundance_col.lower()
        )

        label = (
            "Average Predicted Abundance"
            if is_prediction
            else "Average Plastic Abundance"
        )

        st.caption(f"Data source: {source_name}")
        st.caption(f"Column: {abundance_col}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Records", f"{len(values):,}")
        c2.metric(label, f"{values.mean():,.2f}")
        c3.metric("Maximum Value", f"{values.max():,.2f}")

        st.divider()
        st.subheader("Plastic Abundance Distribution")

        st.line_chart(
            values.reset_index(drop=True).rename("Abundance")
        )

        st.subheader("Descriptive Statistics")

        stats = pd.DataFrame({
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Standard Deviation"
            ],
            "Value": [
                values.count(),
                values.mean(),
                values.median(),
                values.min(),
                values.max(),
                values.std()
            ]
        })

        show_table(stats)

        st.subheader("Top 20 Abundance Records")

        source[abundance_col] = pd.to_numeric(
            source[abundance_col], errors="coerce"
        )

        top_records = source.dropna(
            subset=[abundance_col]
        ).sort_values(
            abundance_col, ascending=False
        ).head(20)

        show_table(top_records)
        download_csv(
            top_records,
            "top_plastic_abundance_records.csv"
        )


# --------------------------------------------------
# POLLUTION HOTSPOTS
# --------------------------------------------------
elif section == "Pollution Hotspots":
    st.title("🔥 Marine Pollution Hotspot Detection")

    source = (
        hotspot_ranking
        if not hotspot_ranking.empty
        else hotspots
    )

    if source.empty:
        st.warning(
            "hotspot_ranking.csv and hotspot_results.csv are unavailable."
        )
    else:
        avg_col = find_col(
            source,
            ["average_pollution", "avg_pollution", "mean_pollution"]
        )
        cluster_col = find_col(
            source,
            ["cluster_id", "cluster", "hotspot_id"]
        )
        max_col = find_col(
            source,
            ["max_pollution", "maximum_pollution"]
        )

        if avg_col:
            source[avg_col] = pd.to_numeric(
                source[avg_col], errors="coerce"
            )

            ranked = source.dropna(
                subset=[avg_col]
            ).sort_values(avg_col, ascending=False)

            if cluster_col:
                st.subheader("Average Pollution by Hotspot")
                st.bar_chart(ranked.set_index(cluster_col)[[avg_col]])

        if max_col:
            source[max_col] = pd.to_numeric(
                source[max_col], errors="coerce"
            )

            if cluster_col:
                st.subheader("Maximum Pollution by Hotspot")
                max_df = source.dropna(
                    subset=[max_col]
                ).set_index(cluster_col)[[max_col]]
                st.bar_chart(max_df)

        show_table(source)
        download_csv(source, "hotspot_ranking.csv")


# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------
elif section == "SHAP Explainability":
    st.title("🔍 SHAP Explainability")

    source = shap_data if not shap_data.empty else importance

    if source.empty:
        st.warning(
            "SHAP results were not found. Check the SHAP CSV files."
        )
    else:
        feature_col = find_col(
            source,
            ["feature", "feature_name", "variable"]
        )
        value_col = find_col(
            source,
            [
                "mean_abs_shap",
                "mean_shap",
                "shap_importance",
                "importance",
                "mean_abs_shap_value"
            ]
        )

        if feature_col and value_col:
            chart = source[[feature_col, value_col]].copy()

            chart[value_col] = pd.to_numeric(
                chart[value_col], errors="coerce"
            )

            chart = chart.dropna(
                subset=[value_col]
            ).sort_values(
                value_col, ascending=False
            ).head(10)

            st.subheader("Top 10 Important Features")
            st.bar_chart(chart.set_index(feature_col)[[value_col]])

            st.caption(
                "This chart shows saved feature-importance values. "
                "Importance does not prove that a feature causes pollution."
            )
            show_table(chart)
        else:
            st.warning(
                "Feature or importance column names were not recognised."
            )
            show_table(source)

        download_csv(source, "shap_explainability_results.csv")


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif section == "Model Comparison":
    st.title("⚖️ Classical vs Quantum-Classical Models")

    if comparison.empty:
        st.warning("final_model_comparison.csv is unavailable.")
    else:
        show_table(comparison)

        model_col = find_col(
            comparison,
            ["model", "model_name", "algorithm", "method"]
        )

        metric_keywords = [
            "mae", "rmse", "r2", "r_squared",
            "accuracy", "precision", "recall", "f1"
        ]

        metrics = []

        for col in comparison.columns:
            if any(
                keyword in str(col).lower()
                for keyword in metric_keywords
            ):
                converted = pd.to_numeric(
                    comparison[col], errors="coerce"
                )

                if converted.notna().any():
                    comparison[col] = converted
                    metrics.append(col)

        if model_col and metrics:
            for metric in metrics:
                st.subheader(f"{metric.upper()} Comparison")

                chart = comparison[
                    [model_col, metric]
                ].dropna().set_index(model_col)

                if not chart.empty:
                    st.bar_chart(chart)

            st.caption(
                "Lower MAE/RMSE generally indicates smaller errors. "
                "Higher R² generally indicates a better fit. "
                "Compare models evaluated on the same test data."
            )
        else:
            st.warning(
                "Could not identify model and metric columns automatically."
            )

        download_csv(comparison, "final_model_comparison.csv")


# --------------------------------------------------
# PROJECT FILES
# --------------------------------------------------
elif section == "Project Files":
    st.title("📁 Marine Pollution Project Files")

    expected_files = [
        "dashboard_summary.csv",
        "gis_predictions.csv",
        "top_10_shap_features.csv",
        "hotspot_results.csv",
        "hotspot_ranking.csv",
        "prediction_risk_results.csv",
        "final_model_comparison.csv",
        "shap_importance.csv",
        "test_data.csv",
        "actual_values.csv",
        "FINAL_MARINE_POLLUTION_RESULTS.csv",
        "catboost_model.cbm",
        "marine_pollution_risk_map.html"
    ]

    status = pd.DataFrame({
        "File Name": expected_files,
        "Status": [
            "Available" if (BASE_DIR / name).exists()
            else "Not found"
            for name in expected_files
        ]
    })

    show_table(status)

    available = sorted(BASE_DIR.glob("*.csv"))

    if available:
        selected = st.selectbox(
            "Select a CSV file to view",
            [p.name for p in available]
        )

        selected_df = load_csv(selected)

        st.write(f"Rows: {len(selected_df):,}")
        st.write(f"Columns: {len(selected_df.columns):,}")

        show_table(selected_df.head(100))
        download_csv(selected_df, selected)
    else:
        st.warning("No CSV files were found in the app folder.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Marine Pollution AI • CatBoost • SHAP • GIS • "
    "Hybrid Quantum-Classical Model Evaluation"
)
