
import streamlit as st
import pandas as pd
from pathlib import Path

# Plotly is optional; dashboard will still start if unavailable.
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    PLOTLY_AVAILABLE = False


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Marine Pollution AI Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

RISK_COLORS = {
    "Low": "#39D98A",
    "Moderate": "#FFD166",
    "High": "#FF9F43",
    "Critical": "#FF5C6C",
    "Unknown": "#94A3B8"
}


# ==================================================
# THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071B33, #0B2945);
    color: #EAF8FF;
}
h1, h2, h3 {
    color: #39D5BB !important;
}
p, label, [data-testid="stMarkdownContainer"] {
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
    font-weight: bold;
}
[data-testid="stDataFrame"] {
    border: 1px solid #234D6B;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================
st.title("🌊 AI-Based Marine Pollution Prediction")
st.markdown(
    "### Explainable AI • CatBoost • GIS Hotspots • Spatial Risk Assessment"
)
st.caption(
    "Marine microplastic pollution prediction using saved machine learning results."
)
st.divider()


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def load_csv(filename):
    """Safely load a CSV from the dashboard folder."""
    path = BASE / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Unable to read {filename}: {error}")
        return pd.DataFrame()


def column_lookup(data):
    """Case-insensitive column-name lookup."""
    return {
        str(column).strip().lower(): column
        for column in data.columns
    }


def find_column(data, candidates):
    """Find the first matching column from a candidate list."""
    lookup = column_lookup(data)

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def normalize_risk(value):
    """Standardize risk category labels."""
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().title()

    # Handle labels such as "Low Risk" or "Critical Risk".
    value = value.replace(" Risk", "").strip()

    if value in RISK_COLORS:
        return value

    return "Unknown"


def get_risk_column(data):
    return find_column(
        data,
        [
            "risk_level",
            "risk_category",
            "pollution_risk",
            "risk",
            "predicted_risk",
            "risk_class"
        ]
    )


def get_numeric_columns(data):
    return data.select_dtypes(include="number").columns.tolist()


# ==================================================
# LOAD SAVED DATA
# ==================================================
summary = load_csv("dashboard_summary.csv")
predictions = load_csv("gis_predictions.csv")
shap_data = load_csv("top_10_shap_features.csv")

if shap_data.empty:
    shap_data = load_csv("shap_importance.csv")

hotspots = load_csv("hotspot_results.csv")
ranking = load_csv("hotspot_ranking.csv")
comparison = load_csv("final_model_comparison.csv")
risk = load_csv("prediction_risk_results.csv")


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
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
    "Dashboard displays saved project results. "
    "Models are not retrained here."
)


# ==================================================
# OVERVIEW
# ==================================================
if section == "Overview":

    st.header("📊 Project Overview")

    if not summary.empty:
        st.subheader("Project Summary")
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("dashboard_summary.csv is not available.")

    st.subheader("Prediction Preview")

    if not predictions.empty:
        st.dataframe(
            predictions.head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("GIS prediction data is not available.")


# ==================================================
# GIS POLLUTION MAP
# ==================================================
elif section == "GIS Pollution Map":

    st.header("🗺️ GIS-Based Pollution Mapping")
    st.caption(
        "Pollution locations are colored according to their risk category."
    )

    if predictions.empty:
        st.warning("gis_predictions.csv is missing or empty.")

    else:
        # Find latitude and longitude columns.
        lat_col = find_column(predictions, ["latitude", "lat"])
        lon_col = find_column(
            predictions,
            ["longitude", "lon", "long", "lng"]
        )

        if lat_col is None or lon_col is None:
            st.warning(
                "Latitude/longitude columns were not found in "
                "gis_predictions.csv. Please check the CSV column names."
            )

        else:
            map_data = predictions.copy()

            map_data["_latitude"] = pd.to_numeric(
                map_data[lat_col], errors="coerce"
            )
            map_data["_longitude"] = pd.to_numeric(
                map_data[lon_col], errors="coerce"
            )

            map_data = map_data.dropna(
                subset=["_latitude", "_longitude"]
            ).copy()

            map_data = map_data[
                map_data["_latitude"].between(-90, 90)
                & map_data["_longitude"].between(-180, 180)
            ].copy()

            # Use risk labels from the prediction file first.
            prediction_risk_col = get_risk_column(map_data)

            if prediction_risk_col is not None:
                map_data["Risk Level"] = map_data[
                    prediction_risk_col
                ].apply(normalize_risk)

            # If the prediction file has no risk column, try the
            # separately saved risk results file.
            elif (
                not risk.empty
                and len(risk) == len(predictions)
                and get_risk_column(risk) is not None
            ):
                original_index = map_data.index
                risk_col = get_risk_column(risk)

                map_data["Risk Level"] = (
                    risk.loc[original_index, risk_col]
                    .apply(normalize_risk)
                    .values
                )

            else:
                map_data["Risk Level"] = "Unknown"

            # Look for an area or location name for hover details.
            area_col = find_column(
                map_data,
                [
                    "area_name",
                    "area",
                    "location_name",
                    "location",
                    "site_name",
                    "site",
                    "region",
                    "station",
                    "country",
                    "city",
                    "sampling_location",
                    "location_id"
                ]
            )

            if not map_data.empty:
                st.subheader("Pollution Locations")

                # Legend
                legend_cols = st.columns(4)

                for index, (label, color) in enumerate(
                    [
                        ("Low", RISK_COLORS["Low"]),
                        ("Moderate", RISK_COLORS["Moderate"]),
                        ("High", RISK_COLORS["High"]),
                        ("Critical", RISK_COLORS["Critical"])
                    ]
                ):
                    with legend_cols[index]:
                        st.markdown(
                            f"""
                            <div style="
                                background:{color};
                                color:#071B33;
                                padding:9px;
                                border-radius:8px;
                                text-align:center;
                                font-weight:bold;">
                                {label}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.write("")

                if PLOTLY_AVAILABLE:
                    hover_fields = ["Risk Level"]

                    if area_col is not None:
                        hover_fields.insert(0, area_col)

                    # Show a pollution abundance value in the hover
                    # panel if a suitable column exists.
                    abundance_col = find_column(
                        map_data,
                        [
                            "predicted_abundance",
                            "predicted_pollution",
                            "predicted_value",
                            "prediction",
                            "plastic_abundance_converted",
                            "plastic_abundance",
                            "abundance",
                            "predicted"
                        ]
                    )

                    if (
                        abundance_col is not None
                        and abundance_col not in hover_fields
                    ):
                        hover_fields.append(abundance_col)

                    try:
                        fig = px.scatter_mapbox(
                            map_data,
                            lat="_latitude",
                            lon="_longitude",
                            color="Risk Level",
                            color_discrete_map=RISK_COLORS,
                            category_orders={
                                "Risk Level": [
                                    "Low",
                                    "Moderate",
                                    "High",
                                    "Critical",
                                    "Unknown"
                                ]
                            },
                            hover_name=(
                                area_col if area_col is not None else None
                            ),
                            hover_data=hover_fields,
                            zoom=1,
                            height=570,
                            title="Marine Pollution Risk Map"
                        )

                        fig.update_traces(
                            marker=dict(size=11, opacity=0.9)
                        )

                        fig.update_layout(
                            mapbox_style="carto-darkmatter",
                            mapbox_center={
                                "lat": float(
                                    map_data["_latitude"].mean()
                                ),
                                "lon": float(
                                    map_data["_longitude"].mean()
                                )
                            },
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#EAF8FF"),
                            margin=dict(l=0, r=0, t=50, b=0),
                            legend_title_text="Risk Category"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    except Exception as error:
                        st.warning(
                            "The colored map could not be rendered. "
                            "Showing the available coordinate map instead."
                        )
                        st.caption(f"Map detail: {error}")

                        st.map(
                            map_data[
                                ["_latitude", "_longitude"]
                            ].rename(
                                columns={
                                    "_latitude": "latitude",
                                    "_longitude": "longitude"
                                }
                            )
                        )

                else:
                    st.info(
                        "Plotly is not installed. Showing a basic map. "
                        "Install plotly in requirements.txt to enable "
                        "risk-colored dots."
                    )

                    st.map(
                        map_data[
                            ["_latitude", "_longitude"]
                        ].rename(
                            columns={
                                "_latitude": "latitude",
                                "_longitude": "longitude"
                            }
                        )
                    )

                st.caption(
                    f"Showing {len(map_data)} valid coordinate records."
                )

                st.subheader("Mapped Location Details")

                detail_cols = []

                if area_col is not None:
                    detail_cols.append(area_col)

                detail_cols.extend(
                    ["_latitude", "_longitude", "Risk Level"]
                )

                st.dataframe(
                    map_data[detail_cols].rename(
                        columns={
                            "_latitude": "Latitude",
                            "_longitude": "Longitude"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("No valid geographic coordinates were found.")


# ==================================================
# RISK ASSESSMENT
# ==================================================
elif section == "Risk Assessment":

    st.header("🚦 Pollution Risk Assessment")

    if not risk.empty:
        st.dataframe(
            risk.head(200),
            use_container_width=True,
            hide_index=True
        )

        risk_col = get_risk_column(risk)

        # Use the GIS prediction file if the risk CSV does not contain
        # a risk category.
        if risk_col is None and not predictions.empty:
            risk_col = get_risk_column(predictions)

            if risk_col is not None:
                risk_values = predictions[risk_col]

            else:
                risk_values = None
        else:
            risk_values = risk[risk_col] if risk_col is not None else None

        if risk_values is not None:
            normalized = risk_values.apply(normalize_risk)
            counts = normalized.value_counts()

            st.subheader("Risk Category Distribution")

            risk_summary = pd.DataFrame({
                "Risk Level": [
                    "Low", "Moderate", "High", "Critical", "Unknown"
                ]
            })

            risk_summary["Locations"] = risk_summary[
                "Risk Level"
            ].map(counts).fillna(0).astype(int)

            total = risk_summary["Locations"].sum()

            if total > 0:
                risk_summary["Percentage"] = (
                    risk_summary["Locations"] / total * 100
                ).round(1)

            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    risk_summary,
                    x="Risk Level",
                    y="Locations",
                    color="Risk Level",
                    color_discrete_map=RISK_COLORS,
                    text="Locations",
                    category_orders={
                        "Risk Level": [
                            "Low", "Moderate", "High",
                            "Critical", "Unknown"
                        ]
                    },
                    title="Pollution Risk Distribution"
                )

                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#EAF8FF"),
                    showlegend=False,
                    yaxis_title="Number of Locations",
                    xaxis_title="Risk Category"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.bar_chart(
                    risk_summary.set_index("Risk Level")["Locations"]
                )

            st.dataframe(
                risk_summary,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Risk category column was not found. "
                "Check the risk CSV column names."
            )

    elif not predictions.empty and get_risk_column(predictions):
        st.info(
            "Risk results file is unavailable. "
            "Using risk categories from GIS predictions."
        )

        pred_risk_col = get_risk_column(predictions)
        counts = (
            predictions[pred_risk_col]
            .apply(normalize_risk)
            .value_counts()
        )

        st.bar_chart(counts)

    else:
        st.warning(
            "Risk assessment data is unavailable. "
            "Check prediction_risk_results.csv."
        )


# ==================================================
# POLLUTION HOTSPOTS
# ==================================================
elif section == "Pollution Hotspots":

    st.header("🔥 Marine Pollution Hotspots")

    if not hotspots.empty:
        st.subheader("Hotspot Results")
        st.dataframe(
            hotspots.head(100),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hotspot results file was found.")

    if not ranking.empty:
        st.subheader("📍 Hotspot Ranking")
        st.dataframe(
            ranking.head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hotspot ranking file was found.")


# ==================================================
# SHAP EXPLAINABILITY
# ==================================================
elif section == "SHAP Explainability":

    st.header("🔎 SHAP Feature Importance")
    st.caption(
        "Feature importance helps explain which features contribute "
        "most to the saved model explanation."
    )

    if shap_data.empty:
        st.warning(
            "SHAP data is unavailable. Check top_10_shap_features.csv "
            "or shap_importance.csv."
        )

    else:
        feature_col = shap_data.columns[0]
        numeric_cols = get_numeric_columns(shap_data)

        if not numeric_cols:
            st.warning(
                "No numeric SHAP importance column was found."
            )
            st.dataframe(
                shap_data,
                use_container_width=True,
                hide_index=True
            )

        else:
            # Find the most likely importance column.
            preferred_names = [
                "mean_abs_shap",
                "shap_importance",
                "mean_shap",
                "importance",
                "shap_value",
                "value"
            ]

            value_col = None

            for preferred in preferred_names:
                for col in numeric_cols:
                    if preferred in str(col).strip().lower():
                        value_col = col
                        break

                if value_col is not None:
                    break

            if value_col is None:
                value_col = numeric_cols[0]

            chart_data = shap_data[
                [feature_col, value_col]
            ].copy()

            chart_data[value_col] = pd.to_numeric(
                chart_data[value_col],
                errors="coerce"
            )

            chart_data = chart_data.dropna(
                subset=[feature_col, value_col]
            )

            if chart_data.empty:
                st.warning(
                    "No valid numeric SHAP importance values are available."
                )

            else:
                # Global feature importance is commonly shown using
                # absolute SHAP values.
                chart_data["_importance"] = (
                    chart_data[value_col].abs()
                )

                chart_data[feature_col] = (
                    chart_data[feature_col].astype(str)
                )

                chart_data = chart_data.sort_values(
                    "_importance",
                    ascending=False
                ).head(15).copy()

                total_importance = chart_data[
                    "_importance"
                ].sum()

                if total_importance > 0:
                    chart_data["_percentage"] = (
                        chart_data["_importance"]
                        / total_importance * 100
                    )
                else:
                    chart_data["_percentage"] = 0.0

                chart_data["_label"] = chart_data.apply(
                    lambda row: (
                        f"{row['_importance']:.3f} "
                        f"({row['_percentage']:.1f}%)"
                    ),
                    axis=1
                )

                st.subheader(
                    "Top SHAP Features — Importance and Percentage"
                )

                if PLOTLY_AVAILABLE:
                    chart_data = chart_data.sort_values(
                        "_importance",
                        ascending=True
                    )

                    color_sequence = [
                        "#39D5BB",
                        "#4CC9F0",
                        "#4895EF",
                        "#7B61FF",
                        "#B388FF",
                        "#F15BB5",
                        "#FF6B6B",
                        "#FF9F43",
                        "#FFD166",
                        "#39D98A",
                        "#00B4D8",
                        "#80ED99",
                        "#C77DFF",
                        "#F28482",
                        "#90BE6D"
                    ]

                    fig = px.bar(
                        chart_data,
                        x="_importance",
                        y=feature_col,
                        orientation="h",
                        color=feature_col,
                        color_discrete_sequence=color_sequence,
                        text="_label",
                        custom_data=["_percentage"],
                        labels={
                            "_importance": "Absolute SHAP Importance",
                            feature_col: "Feature"
                        },
                        title="Feature Importance"
                    )

                    fig.update_traces(
                        textposition="outside",
                        cliponaxis=False,
                        textfont=dict(
                            color="#FFFFFF",
                            size=11
                        ),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Importance: %{x:.4f}<br>"
                            "Share of displayed features: "
                            "%{customdata[0]:.2f}%"
                            "<extra></extra>"
                        )
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#EAF8FF"),
                        xaxis_title="Absolute SHAP Importance",
                        yaxis_title="Feature",
                        showlegend=False,
                        height=max(420, 38 * len(chart_data)),
                        margin=dict(l=10, r=120, t=70, b=30),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="#23415F"
                        ),
                        yaxis=dict(
                            showgrid=False,
                            automargin=True
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "Plotly is not installed. Showing the SHAP "
                        "importance chart without custom bar labels. "
                        "Add plotly to requirements.txt for the "
                        "full colored chart."
                    )

                    st.bar_chart(
                        chart_data.set_index(feature_col)[
                            ["_importance"]
                        ]
                    )

                st.caption(
                    "Percentages are calculated across the displayed "
                    "top features only. They are not pollution percentages "
                    "and do not represent causal effects."
                )

                # Summary cards
                chart_sorted = chart_data.sort_values(
                    "_importance",
                    ascending=False
                )

                top_feature = chart_sorted.iloc[0]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Features Displayed",
                    str(len(chart_data))
                )

                col2.metric(
                    "Top Feature",
                    str(top_feature[feature_col])
                )

                col3.metric(
                    "Top Feature Share",
                    f"{top_feature['_percentage']:.1f}%"
                )

                # Detailed SHAP table
                st.subheader("📊 Feature Importance Details")

                details = chart_sorted[
                    [
                        feature_col,
                        value_col,
                        "_importance",
                        "_percentage"
                    ]
                ].rename(
                    columns={
                        feature_col: "Feature",
                        value_col: "Original SHAP Value",
                        "_importance": "Absolute Importance",
                        "_percentage": "Percentage (%)"
                    }
                )

                st.dataframe(
                    details.round(4),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="⬇️ Download SHAP Details",
                    data=details.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="shap_explainability_details.csv",
                    mime="text/csv"
                )


# ==================================================
# MODEL COMPARISON
# ==================================================
elif section == "Model Comparison":

    st.header("🤖 Model Performance Comparison")

    if not comparison.empty:
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        numeric_data = comparison.select_dtypes(
            include="number"
        )

        if not numeric_data.empty:
            st.subheader("Performance Metrics")

            if PLOTLY_AVAILABLE:
                # Plot numeric metrics without assuming a specific
                # model-name column exists.
                chart_data = comparison.copy()
                model_col = find_column(
                    chart_data,
                    [
                        "model",
                        "model_name",
                        "algorithm",
                        "method"
                    ]
                )

                if model_col is not None:
                    numeric_cols = get_numeric_columns(chart_data)

                    if numeric_cols:
                        melted = chart_data[
                            [model_col] + numeric_cols
                        ].melt(
                            id_vars=[model_col],
                            var_name="Metric",
                            value_name="Value"
                        )

                        fig = px.bar(
                            melted,
                            x="Metric",
                            y="Value",
                            color=model_col,
                            barmode="group",
                            title="Model Metrics Comparison"
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#EAF8FF")
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:
                        st.info("No numeric comparison metrics found.")

                else:
                    st.bar_chart(numeric_data)

            else:
                st.bar_chart(numeric_data)

    else:
        st.warning(
            "final_model_comparison.csv is missing or empty."
        )


# ==================================================
# PROJECT FILES
# ==================================================
elif section == "Project Files":

    st.header("📁 Project Files")
    st.caption(
        "This section checks whether saved project files exist. "
        "It does not delete or modify them."
    )

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
        "FINAL_MARINE_POLLUTION_RESULTS.csv"
    ]

    status = []

    for filename in filenames:
        exists = (BASE / filename).exists()

        status.append({
            "File": filename,
            "Status": (
                "Available" if exists else "Not found"
            )
        })

    st.dataframe(
        pd.DataFrame(status),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "Marine Pollution AI Dashboard | CatBoost • SHAP • GIS • Risk Assessment"
)
st.success(
    "Dashboard uses saved results. No model retraining is performed."
)
