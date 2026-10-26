
import streamlit as st
import pandas as pd
from pathlib import Path

# Plotly is optional.
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

RISK_ORDER = ["Low", "Moderate", "High", "Critical", "Unknown"]


# ==================================================
# THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061426 0%, #0B2945 55%, #103653 100%);
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
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #23617A;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #EAF8FF !important;
}
.stButton > button,
.stDownloadButton > button {
    background-color: #39D5BB;
    color: #071B33;
    border-radius: 8px;
    font-weight: bold;
}
[data-testid="stDataFrame"] {
    border: 1px solid #234D6B;
    border-radius: 10px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def load_csv(filename):
    """Load a CSV safely from the dashboard folder."""
    path = BASE / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def column_lookup(data):
    return {
        str(column).strip().lower(): column
        for column in data.columns
    }


def find_column(data, candidates):
    """Find a column without depending on capitalization."""
    lookup = column_lookup(data)

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def normalize_risk(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().title()
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


def get_abundance_column(data):
    """Find a likely plastic or pollution abundance column."""
    return find_column(
        data,
        [
            "plastic_abundance_converted",
            "predicted_abundance",
            "predicted_pollution",
            "predicted_value",
            "plastic_abundance",
            "microplastic_abundance",
            "microplastic_concentration",
            "abundance",
            "prediction",
            "predicted",
            "target"
        ]
    )


def format_number(value):
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def make_risk_summary(data, risk_col):
    """Create counts and percentages for all risk categories."""
    if data.empty or risk_col is None:
        return pd.DataFrame(
            columns=["Risk Level", "Locations", "Percentage"]
        )

    normalized = data[risk_col].apply(normalize_risk)
    counts = normalized.value_counts()

    result = pd.DataFrame({"Risk Level": RISK_ORDER})
    result["Locations"] = (
        result["Risk Level"].map(counts).fillna(0).astype(int)
    )

    total = result["Locations"].sum()

    if total:
        result["Percentage"] = (
            result["Locations"] / total * 100
        ).round(1)
    else:
        result["Percentage"] = 0.0

    return result


def show_risk_chart(data, title="Risk Category Distribution"):
    if data.empty:
        st.info("Risk distribution data is not available.")
        return

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            data,
            x="Risk Level",
            y="Locations",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk Level": RISK_ORDER},
            text="Percentage",
            title=title,
            custom_data=["Percentage"]
        )

        fig.update_traces(
            texttemplate="%{y} (%{customdata[0]:.1f}%)",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAF8FF"),
            showlegend=False,
            xaxis_title="Risk Category",
            yaxis_title="Number of Locations",
            margin=dict(t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(data.set_index("Risk Level")["Locations"])


def show_dataframe(data, max_rows=200):
    if data.empty:
        st.info("No records are available in this data file.")
        return

    st.dataframe(
        data.head(max_rows),
        use_container_width=True,
        hide_index=True
    )


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

# Alternative datasets used by the Plastic Pollution Analysis page.
plastic_sources = [
    "cleaned_marine_pollution.csv",
    "marine_pollution_GIS_results.csv",
    "preprocessed_marine_pollution.csv",
    "Global data set on micro- and mesoplastic loads in marine sediments.csv",
    "gis_predictions.csv"
]

plastic_data = pd.DataFrame()
plastic_source_name = None

# Prefer a dataset containing an identifiable abundance column.
for filename in plastic_sources:
    candidate = load_csv(filename)

    if not candidate.empty and get_abundance_column(candidate):
        plastic_data = candidate.copy()
        plastic_source_name = filename
        break

# Fallback to GIS predictions if no abundance column was found elsewhere.
if plastic_data.empty and not predictions.empty:
    plastic_data = predictions.copy()
    plastic_source_name = "gis_predictions.csv"


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
        "Plastic Pollution Analysis",
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
    "Explore saved marine pollution predictions, "
    "risk categories and explainability results."
)


# ==================================================
# HEADER
# ==================================================
st.title("🌊 AI-Based Marine Pollution Prediction")
st.markdown(
    "### Explainable AI  •  CatBoost  •  GIS Hotspots  •  Spatial Risk Assessment"
)
st.caption(
    "Explore marine microplastic abundance, pollution hotspots "
    "and spatial risk using the available project results."
)
st.divider()


# ==================================================
# OVERVIEW
# ==================================================
if section == "Overview":

    st.header("🌐 Marine Pollution Command Centre")
    st.write(
        "A quick view of the available pollution predictions, "
        "risk distribution and important project results."
    )

    # Find abundance and risk columns in prediction data.
    abundance_col = get_abundance_column(predictions)
    prediction_risk_col = get_risk_column(predictions)

    # Use saved risk data if prediction data has no risk labels.
    risk_source = predictions
    risk_source_col = prediction_risk_col

    if (
        risk_source_col is None
        and not risk.empty
        and get_risk_column(risk) is not None
    ):
        risk_source = risk
        risk_source_col = get_risk_column(risk)

    valid_abundance = pd.Series(dtype=float)

    if abundance_col is not None and not predictions.empty:
        valid_abundance = pd.to_numeric(
            predictions[abundance_col],
            errors="coerce"
        ).dropna()

    total_locations = (
        len(predictions) if not predictions.empty else 0
    )

    avg_abundance = (
        valid_abundance.mean()
        if not valid_abundance.empty
        else None
    )

    max_abundance = (
        valid_abundance.max()
        if not valid_abundance.empty
        else None
    )

    critical_count = 0

    if risk_source_col is not None and not risk_source.empty:
        critical_count = int(
            (
                risk_source[risk_source_col].apply(normalize_risk)
                == "Critical"
            ).sum()
        )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Prediction Records",
        f"{total_locations:,}"
    )

    metric_cols[1].metric(
        "Average Abundance",
        format_number(avg_abundance)
        if avg_abundance is not None else "N/A"
    )

    metric_cols[2].metric(
        "Maximum Abundance",
        format_number(max_abundance)
        if max_abundance is not None else "N/A"
    )

    metric_cols[3].metric(
        "Critical-Risk Records",
        f"{critical_count:,}"
    )

    st.divider()

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("🚦 Risk Distribution")

        if risk_source_col is not None and not risk_source.empty:
            risk_summary = make_risk_summary(
                risk_source,
                risk_source_col
            )
            show_risk_chart(risk_summary, "Marine Pollution Risk Profile")
            st.dataframe(
                risk_summary,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "Risk categories are not available in the current "
                "prediction or risk-results CSV."
            )

    with right:
        st.subheader("🧪 Plastic Abundance Profile")

        if not valid_abundance.empty:
            stats = pd.DataFrame({
                "Statistic": [
                    "Valid records",
                    "Mean",
                    "Median",
                    "Minimum",
                    "Maximum",
                    "Standard deviation"
                ],
                "Value": [
                    f"{len(valid_abundance):,}",
                    format_number(valid_abundance.mean()),
                    format_number(valid_abundance.median()),
                    format_number(valid_abundance.min()),
                    format_number(valid_abundance.max()),
                    format_number(valid_abundance.std())
                ]
            })

            st.dataframe(
                stats,
                use_container_width=True,
                hide_index=True
            )

            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    pd.DataFrame({"Abundance": valid_abundance}),
                    x="Abundance",
                    nbins=30,
                    title="Abundance Distribution",
                    color_discrete_sequence=["#39D5BB"]
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#EAF8FF"),
                    xaxis_title="Abundance",
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No numeric abundance column was found in the "
                "available GIS prediction file."
            )

    st.divider()
    st.subheader("📍 Recent Prediction Records")

    if not predictions.empty:
        st.dataframe(
            predictions.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction Records",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="marine_pollution_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("gis_predictions.csv is not available.")


# ==================================================
# PLASTIC POLLUTION ANALYSIS
# ==================================================
elif section == "Plastic Pollution Analysis":

    st.header("🧪 Plastic Pollution Analysis")
    st.write(
        "Explore the available plastic or microplastic abundance data, "
        "its statistical distribution and the highest recorded values."
    )

    if plastic_data.empty:
        st.warning(
            "No suitable plastic abundance dataset was found. "
            "Check that the original CSV files are uploaded to the "
            "same folder as app.py."
        )

    else:
        st.caption(f"Data source: {plastic_source_name}")

        abundance_col = get_abundance_column(plastic_data)

        if abundance_col is None:
            st.warning(
                "The dataset is available, but a recognized abundance "
                "column could not be identified."
            )
            show_dataframe(plastic_data)
        else:
            analysis = plastic_data.copy()

            analysis["_abundance_value"] = pd.to_numeric(
                analysis[abundance_col],
                errors="coerce"
            )

            valid = analysis.dropna(
                subset=["_abundance_value"]
            ).copy()

            if valid.empty:
                st.warning(
                    f"No numeric values were found in '{abundance_col}'."
                )
            else:
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Valid Records", f"{len(valid):,}")
                c2.metric(
                    "Average Abundance",
                    format_number(valid["_abundance_value"].mean())
                )
                c3.metric(
                    "Median Abundance",
                    format_number(valid["_abundance_value"].median())
                )
                c4.metric(
                    "Maximum Abundance",
                    format_number(valid["_abundance_value"].max())
                )

                st.caption(
                    f"Abundance values are read from the actual "
                    f"'{abundance_col}' column. Units depend on the source dataset."
                )

                st.divider()

                left, right = st.columns(2)

                with left:
                    st.subheader("📈 Abundance Distribution")

                    if PLOTLY_AVAILABLE:
                        fig = px.histogram(
                            valid,
                            x="_abundance_value",
                            nbins=35,
                            title="Plastic Abundance Frequency",
                            color_discrete_sequence=["#39D5BB"]
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#EAF8FF"),
                            xaxis_title="Abundance",
                            yaxis_title="Number of Records"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )
                    else:
                        st.bar_chart(
                            valid["_abundance_value"].reset_index(
                                drop=True
                            )
                        )

                with right:
                    st.subheader("📊 Summary Statistics")

                    stats = valid["_abundance_value"].describe().rename(
                        "Value"
                    ).reset_index()

                    stats.columns = ["Statistic", "Value"]
                    stats["Value"] = stats["Value"].round(4)

                    st.dataframe(
                        stats,
                        use_container_width=True,
                        hide_index=True
                    )

                st.divider()
                st.subheader("🔎 Highest Abundance Records")

                top_records = valid.nlargest(
                    20,
                    "_abundance_value"
                ).copy()

                area_col = find_column(
                    top_records,
                    [
                        "area_name",
                        "area",
                        "location_name",
                        "location",
                        "site_name",
                        "site",
                        "region",
                        "country",
                        "city",
                        "sampling_location"
                    ]
                )

                display_cols = []

                if area_col is not None:
                    display_cols.append(area_col)

                lat_col = find_column(
                    top_records, ["latitude", "lat"]
                )
                lon_col = find_column(
                    top_records, ["longitude", "lon", "long", "lng"]
                )

                if lat_col is not None:
                    display_cols.append(lat_col)

                if lon_col is not None:
                    display_cols.append(lon_col)

                display_cols.append("_abundance_value")

                top_records = top_records[display_cols].rename(
                    columns={
                        "_abundance_value": "Abundance"
                    }
                )

                st.dataframe(
                    top_records,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Plastic Analysis CSV",
                    data=valid.drop(
                        columns=["_abundance_value"],
                        errors="ignore"
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="plastic_pollution_analysis.csv",
                    mime="text/csv"
                )


# ==================================================
# GIS POLLUTION MAP
# ==================================================
elif section == "GIS Pollution Map":

    st.header("🗺️ GIS-Based Pollution Mapping")
    st.caption(
        "Map markers are colored by the available pollution risk category."
    )

    if predictions.empty:
        st.warning("gis_predictions.csv is missing or empty.")

    else:
        lat_col = find_column(predictions, ["latitude", "lat"])
        lon_col = find_column(
            predictions, ["longitude", "lon", "long", "lng"]
        )

        if lat_col is None or lon_col is None:
            st.warning(
                "Latitude/longitude columns were not found in "
                "gis_predictions.csv."
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

            prediction_risk_col = get_risk_column(map_data)

            if prediction_risk_col is not None:
                map_data["Risk Level"] = map_data[
                    prediction_risk_col
                ].apply(normalize_risk)

            elif (
                not risk.empty
                and len(risk) == len(predictions)
                and get_risk_column(risk) is not None
            ):
                risk_col = get_risk_column(risk)
                map_data["Risk Level"] = (
                    risk.loc[map_data.index, risk_col]
                    .apply(normalize_risk)
                    .values
                )
            else:
                map_data["Risk Level"] = "Unknown"

            area_col = find_column(
                map_data,
                [
                    "area_name", "area", "location_name", "location",
                    "site_name", "site", "region", "station",
                    "country", "city", "sampling_location", "location_id"
                ]
            )

            if map_data.empty:
                st.info("No valid geographic coordinates were found.")

            else:
                st.subheader("Pollution Locations")

                legend_cols = st.columns(5)

                for index, label in enumerate(RISK_ORDER):
                    with legend_cols[index]:
                        st.markdown(
                            f"""
                            <div style="
                                background:{RISK_COLORS[label]};
                                color:#071B33;
                                padding:9px 5px;
                                border-radius:8px;
                                text-align:center;
                                font-weight:bold;">
                                {label}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                if PLOTLY_AVAILABLE:
                    hover_fields = ["Risk Level"]

                    if area_col is not None:
                        hover_fields.insert(0, area_col)

                    abundance_col = get_abundance_column(map_data)

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
                            category_orders={"Risk Level": RISK_ORDER},
                            hover_name=area_col if area_col else None,
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
                                "lat": float(map_data["_latitude"].mean()),
                                "lon": float(map_data["_longitude"].mean())
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

                    except Exception:
                        st.warning(
                            "The interactive map could not be rendered. "
                            "Showing the available coordinate map instead."
                        )
                        st.map(
                            map_data[["_latitude", "_longitude"]].rename(
                                columns={
                                    "_latitude": "latitude",
                                    "_longitude": "longitude"
                                }
                            )
                        )
                else:
                    st.warning(
                        "Plotly is not installed. Install plotly to show "
                        "risk-colored map markers."
                    )
                    st.map(
                        map_data[["_latitude", "_longitude"]].rename(
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


# ==================================================
# RISK ASSESSMENT
# ==================================================
elif section == "Risk Assessment":

    st.header("🚦 Pollution Risk Assessment")

    risk_source = risk
    risk_col = get_risk_column(risk_source)

    if risk_col is None and not predictions.empty:
        risk_source = predictions
        risk_col = get_risk_column(risk_source)

    if risk_col is not None and not risk_source.empty:
        show_dataframe(risk_source, 200)

        risk_summary = make_risk_summary(risk_source, risk_col)

        st.subheader("Risk Category Distribution")
        show_risk_chart(risk_summary)

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Risk Summary",
            data=risk_summary.to_csv(index=False).encode("utf-8"),
            file_name="marine_pollution_risk_summary.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "Risk assessment data is unavailable. "
            "Check prediction_risk_results.csv and gis_predictions.csv."
        )


# ==================================================
# POLLUTION HOTSPOTS
# ==================================================
elif section == "Pollution Hotspots":

    st.header("🔥 Marine Pollution Hotspots")

    if not hotspots.empty:
        st.subheader("Hotspot Results")
        show_dataframe(hotspots, 100)

        st.download_button(
            "⬇️ Download Hotspot Results",
            data=hotspots.to_csv(index=False).encode("utf-8"),
            file_name="hotspot_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No hotspot results file was found.")

    if not ranking.empty:
        st.subheader("📍 Hotspot Ranking")
        show_dataframe(ranking, 20)

        st.download_button(
            "⬇️ Download Hotspot Ranking",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="hotspot_ranking.csv",
            mime="text/csv"
        )
    else:
        st.info("No hotspot ranking file was found.")


# ==================================================
# SHAP EXPLAINABILITY
# ==================================================
elif section == "SHAP Explainability":

    st.header("🔎 SHAP Feature Importance")
    st.caption(
        "Feature importance summarizes the available saved SHAP results. "
        "It does not by itself establish causation."
    )

    if shap_data.empty:
        st.warning(
            "SHAP data is unavailable. Check top_10_shap_features.csv "
            "or shap_importance.csv."
        )

    else:
        feature_col = find_column(
            shap_data,
            ["feature", "feature_name", "variable", "name"]
        )

        if feature_col is None:
            feature_col = shap_data.columns[0]

        numeric_cols = get_numeric_columns(shap_data)

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

        if value_col is None and numeric_cols:
            value_col = numeric_cols[0]

        if value_col is None:
            st.warning("No numeric SHAP importance column was found.")
            show_dataframe(shap_data)
        else:
            chart_data = shap_data[
                [feature_col, value_col]
            ].copy()

            chart_data[value_col] = pd.to_numeric(
                chart_data[value_col], errors="coerce"
            )

            chart_data = chart_data.dropna(
                subset=[feature_col, value_col]
            ).copy()

            chart_data[feature_col] = (
                chart_data[feature_col].astype(str)
            )
            chart_data["_importance"] = chart_data[value_col].abs()

            chart_data = chart_data.sort_values(
                "_importance", ascending=False
            ).head(15).copy()

            total_importance = chart_data["_importance"].sum()

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
                "Top Features — Importance and Percentage"
            )

            color_sequence = [
                "#39D5BB", "#4CC9F0", "#4895EF", "#7B61FF",
                "#B388FF", "#F15BB5", "#FF6B6B", "#FF9F43",
                "#FFD166", "#39D98A", "#00B4D8", "#80ED99",
                "#C77DFF", "#F28482", "#90BE6D"
            ]

            chart_data = chart_data.sort_values(
                "_importance", ascending=True
            )

            if PLOTLY_AVAILABLE:
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
                    textfont=dict(color="#FFFFFF", size=11),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Importance: %{x:.4f}<br>"
                        "Share of displayed features: %{customdata[0]:.2f}%"
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
                    xaxis=dict(showgrid=True, gridcolor="#23415F"),
                    yaxis=dict(showgrid=False, automargin=True)
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Install plotly to display multicolour bars "
                    "with importance labels."
                )
                st.bar_chart(
                    chart_data.set_index(feature_col)[["_importance"]]
                )

            st.caption(
                "Percentages represent each feature's share of the "
                "displayed top features only. They are not pollution "
                "percentages or causal effects."
            )

            chart_sorted = chart_data.sort_values(
                "_importance", ascending=False
            )
            top_feature = chart_sorted.iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Features Displayed", str(len(chart_sorted)))
            c2.metric("Top Feature", str(top_feature[feature_col]))
            c3.metric(
                "Top Feature Share",
                f"{top_feature['_percentage']:.1f}%"
            )

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
                "⬇️ Download SHAP Details",
                data=details.to_csv(index=False).encode("utf-8"),
                file_name="shap_explainability_details.csv",
                mime="text/csv"
            )


# ==================================================
# MODEL COMPARISON
# ==================================================
elif section == "Model Comparison":

    st.header("🤖 Model Performance Comparison")

    if not comparison.empty:
        show_dataframe(comparison, 200)

        numeric_data = comparison.select_dtypes(
            include="number"
        )

        if not numeric_data.empty:
            st.subheader("Performance Metrics")

            if PLOTLY_AVAILABLE:
                model_col = find_column(
                    comparison,
                    ["model", "model_name", "algorithm", "method"]
                )

                if model_col is not None:
                    numeric_cols = get_numeric_columns(comparison)

                    if numeric_cols:
                        melted = comparison[
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
                    st.bar_chart(numeric_data)
            else:
                st.bar_chart(numeric_data)

        st.download_button(
            "⬇️ Download Model Comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="final_model_comparison.csv",
            mime="text/csv"
        )
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
        "This section checks whether project files are available. "
        "Files are not deleted or modified."
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
        "FINAL_MARINE_POLLUTION_RESULTS.csv",
        "cleaned_marine_pollution.csv",
        "marine_pollution_GIS_results.csv",
        "preprocessed_marine_pollution.csv"
    ]

    status = []

    for filename in filenames:
        exists = (BASE / filename).exists()

        status.append({
            "File": filename,
            "Status": "Available" if exists else "Not found"
        })

    files_df = pd.DataFrame(status)

    st.dataframe(
        files_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download File Status Report",
        data=files_df.to_csv(index=False).encode("utf-8"),
        file_name="project_file_status.csv",
        mime="text/csv"
    )


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "Marine Pollution AI Dashboard | CatBoost • SHAP • GIS • Risk Assessment"
)
