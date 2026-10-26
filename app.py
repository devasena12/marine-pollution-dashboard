
import os
import glob
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Marine Pollution Intelligence Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DARK OCEAN BLUE THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061629, #0A2342);
    color: #F1F7FF;
}

[data-testid="stSidebar"] {
    background-color: #0B2545 !important;
    border-right: 1px solid #1B4965;
}

[data-testid="stSidebar"] * {
    color: #F1F7FF !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #123456;
    border: 1px solid #245477;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: #087F8C;
    border-color: #39D5BB;
}

h1, h2, h3 {
    color: #39D5BB !important;
}

p, li, label, .stMarkdown {
    color: #EAF4FF;
}

[data-testid="stMetric"] {
    background-color: #102D4D;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #245477;
}

[data-testid="stMetricLabel"] {
    color: #A9D6E5 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

.stButton button, .stDownloadButton button {
    background-color: #087F8C;
    color: white;
    border: 1px solid #39D5BB;
    border-radius: 8px;
}

.stButton button:hover, .stDownloadButton button:hover {
    background-color: #0B9A9E;
    color: white;
}

[data-baseweb="select"] > div {
    background-color: #123456;
    color: white;
    border-color: #39D5BB;
}

input, textarea {
    background-color: #123456 !important;
    color: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #245477;
    border-radius: 8px;
}

hr {
    border-color: #245477;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT DIRECTORY AND FILE LOADER
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    """Find a project file in the app directory or its subfolders."""
    direct = os.path.join(BASE_DIR, filename)
    if os.path.isfile(direct):
        return direct

    matches = glob.glob(
        os.path.join(BASE_DIR, "**", filename),
        recursive=True
    )
    return matches[0] if matches else None


@st.cache_data(show_spinner=False)
def read_csv_file(path):
    return pd.read_csv(path, low_memory=False)


def load_csv(filename):
    path = find_file(filename)
    if not path:
        return pd.DataFrame()

    try:
        return read_csv_file(path)
    except Exception:
        return pd.DataFrame()


# =========================================================
# LOAD PROJECT DATA
# =========================================================
gis_df = load_csv("gis_predictions.csv")

if gis_df.empty:
    gis_df = load_csv("marine_pollution_GIS_results.csv")

if gis_df.empty:
    gis_df = load_csv("prediction_risk_results.csv")

summary_df = load_csv("dashboard_summary.csv")
shap_df = load_csv("top_10_shap_features.csv")

if shap_df.empty:
    shap_df = load_csv("shap_importance.csv")

hotspot_df = load_csv("hotspot_results.csv")

if hotspot_df.empty:
    hotspot_df = load_csv("hotspot_ranking.csv")

comparison_df = load_csv("final_model_comparison.csv")
test_df = load_csv("test_data.csv")
actual_df = load_csv("actual_values.csv")


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    if df.empty:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for col in df.columns:
        normalized = str(col).strip().lower()
        if any(candidate.lower() in normalized for candidate in candidates):
            return col

    return None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[column], errors="coerce").dropna()


def get_prediction_column(df):
    return find_column(df, [
        "Predicted_Pollution",
        "Predicted_Abundance",
        "Predicted_Value",
        "Prediction",
        "y_pred",
        "prediction",
        "_prediction",
        "Plastic_abundance_converted",
        "Predicted"
    ])


def get_risk(value):
    """Risk thresholds are fallback display thresholds."""
    if pd.isna(value):
        return "Unknown"
    if value < 25:
        return "Low"
    if value < 100:
        return "Moderate"
    if value < 500:
        return "High"
    return "Critical"


RISK_COLORS = {
    "Low": "#39D98A",
    "Moderate": "#FFD166",
    "High": "#FF9F43",
    "Critical": "#FF5C6C",
    "Unknown": "#A9B8C8"
}


def prepare_data(df):
    """Add a display prediction and risk field without overwriting source columns."""
    if df.empty:
        return df.copy()

    result = df.copy()
    pred_col = get_prediction_column(result)

    if pred_col:
        result["_display_prediction"] = pd.to_numeric(
            result[pred_col], errors="coerce"
        )
    else:
        result["_display_prediction"] = np.nan

    risk_col = find_column(result, [
        "Risk_Level", "Pollution_Risk", "Risk_Category", "Risk"
    ])

    if risk_col:
        def normalize_risk(value):
            if pd.isna(value):
                return "Unknown"

            value = str(value).strip().title()

            for level in ["Low", "Moderate", "High", "Critical"]:
                if level.lower() in value.lower():
                    return level

            return "Unknown"

        result["_display_risk"] = result[risk_col].apply(normalize_risk)
    else:
        result["_display_risk"] = result["_display_prediction"].apply(get_risk)

    return result


display_df = prepare_data(gis_df)

lat_col = find_column(display_df, [
    "Latitude", "Lat", "Latitude_DD", "decimal_latitude"
])

lon_col = find_column(display_df, [
    "Longitude", "Lon", "Long", "Longitude_DD", "decimal_longitude"
])

location_col = find_column(display_df, [
    "Location", "Location_Name", "Site_Name", "Area",
    "Region", "Country", "Sampling_Location", "Station"
])

prediction_col = "_display_prediction" if not display_df.empty else None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("# 🌊 MARINE AI")
    st.markdown("### Pollution Intelligence")
    st.caption("CatBoost • Explainable AI • GIS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Ocean GIS Explorer",
            "AI Pollution Predictor",
            "AI Explainability",
            "Hotspot Analysis",
            "CatBoost vs QML",
            "Research Data"
        ],
        index=0
    )

    st.divider()
    st.markdown("**Risk Legend**")
    st.markdown("🟢 Low")
    st.markdown("🟡 Moderate")
    st.markdown("🟠 High")
    st.markdown("🔴 Critical")

    st.divider()
    st.caption("Marine Pollution Research Dashboard")


# =========================================================
# COMMON HEADER
# =========================================================
st.markdown("# 🌊 Marine Pollution Intelligence Dashboard")
st.markdown(
    "AI-powered marine microplastic prediction, explainability, "
    "pollution hotspot detection and spatial risk exploration."
)
st.divider()


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    st.subheader("📊 Project Overview")

    values = numeric_series(display_df, prediction_col)

    if values.empty and not summary_df.empty:
        summary_prediction_col = find_column(summary_df, [
            "Predicted_Pollution", "Prediction", "Plastic_abundance_converted"
        ])
        values = numeric_series(summary_df, summary_prediction_col)

    if not values.empty:
        minimum = values.min()
        maximum = values.max()
        average = values.mean()
        location_count = len(values)
    else:
        minimum = maximum = average = np.nan
        location_count = len(display_df)

    risk_counts = (
        display_df["_display_risk"].value_counts()
        if not display_df.empty and "_display_risk" in display_df.columns
        else pd.Series(dtype=int)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📍 Locations", f"{location_count:,}")

    c2.metric(
        "⬇️ Minimum Pollution",
        f"{minimum:,.2f}" if pd.notna(minimum) else "N/A"
    )

    c3.metric(
        "⬆️ Maximum Pollution",
        f"{maximum:,.2f}" if pd.notna(maximum) else "N/A"
    )

    c4.metric(
        "📈 Average Pollution",
        f"{average:,.2f}" if pd.notna(average) else "N/A"
    )

    st.markdown("### Pollution Risk Distribution")

    risk_order = ["Low", "Moderate", "High", "Critical"]

    total_risk = sum(int(risk_counts.get(r, 0)) for r in risk_order)

    risk_columns = st.columns(4)

    for i, risk in enumerate(risk_order):
        count = int(risk_counts.get(risk, 0))
        percentage = count / total_risk * 100 if total_risk else 0

        with risk_columns[i]:
            st.markdown(
                f"""
                <div style="
                    background:{RISK_COLORS[risk]};
                    padding:16px;
                    border-radius:12px;
                    text-align:center;
                    color:#061629;
                    margin-bottom:10px;">
                    <h3 style="color:#061629 !important;">{risk}</h3>
                    <h2 style="color:#061629 !important;">{percentage:.1f}%</h2>
                    <b>{count} records</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    left, right = st.columns(2)

    with left:
        st.markdown("### Risk Category Chart")

        if total_risk:
            chart_df = pd.DataFrame({
                "Risk Level": risk_order,
                "Locations": [int(risk_counts.get(r, 0)) for r in risk_order]
            })

            fig = px.bar(
                chart_df,
                x="Risk Level",
                y="Locations",
                color="Risk Level",
                color_discrete_map=RISK_COLORS
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk distribution will appear when prediction data is available.")

    with right:
        st.markdown("### Pollution Prediction Distribution")

        if not values.empty:
            chart_data = pd.DataFrame({"Pollution": values})
            fig = px.histogram(
                chart_data,
                x="Pollution",
                nbins=30,
                color_discrete_sequence=["#39D5BB"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Prediction values are not available in the loaded CSV files.")

    st.markdown("### Project Modules")

    st.markdown("""
    - **AI Prediction:** CatBoost-based pollution abundance prediction.
    - **GIS Explorer:** Explore available sampling locations geographically.
    - **Explainable AI:** Review saved SHAP feature importance results.
    - **Hotspot Analysis:** Inspect recorded pollution hotspot data.
    - **Model Comparison:** Compare recorded CatBoost and hybrid QML results.
    """)


# =========================================================
# OCEAN GIS EXPLORER
# =========================================================
elif page == "Ocean GIS Explorer":

    st.subheader("🗺️ Ocean Pollution GIS Explorer")

    if display_df.empty:
        st.warning(
            "GIS data could not be loaded. Please check "
            "gis_predictions.csv or marine_pollution_GIS_results.csv."
        )

    elif lat_col is None or lon_col is None:
        st.warning(
            "Latitude/Longitude columns were not found. "
            "Please check the column names in your GIS CSV file."
        )

    else:
        map_df = display_df.copy()

        map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
        map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")
        map_df = map_df.dropna(subset=[lat_col, lon_col])

        map_df = map_df[
            map_df[lat_col].between(-90, 90)
            & map_df[lon_col].between(-180, 180)
        ]

        if map_df.empty:
            st.warning("No valid geographic coordinates are available.")
        else:
            st.caption(
                "Select a point on the map or choose a record below to inspect "
                "its available location, prediction and risk details."
            )

            map_df["_map_label"] = map_df.apply(
                lambda row: (
                    str(row[location_col])
                    if location_col and pd.notna(row[location_col])
                    else f"Location {row.name}"
                ),
                axis=1
            )

            map_df["_map_color"] = map_df["_display_risk"].map(
                RISK_COLORS
            ).fillna(RISK_COLORS["Unknown"])

            map_df["_map_size"] = (
                map_df["_display_prediction"].fillna(0).abs()
            )
            map_df["_map_size"] = (
                map_df["_map_size"].clip(lower=1).apply(
                    lambda x: min(18, max(7, 7 + np.log1p(x)))
                )
            )

            fig = px.scatter_mapbox(
                map_df,
                lat=lat_col,
                lon=lon_col,
                color="_display_risk",
                size="_map_size",
                hover_name="_map_label",
                hover_data={
                    "_display_prediction": ":.2f",
                    "_display_risk": True,
                    lat_col: ":.4f",
                    lon_col: ":.4f",
                    "_map_size": False,
                    "_map_color": False
                },
                color_discrete_map=RISK_COLORS,
                zoom=1,
                height=550
            )

            fig.update_layout(
                mapbox_style="carto-darkmatter",
                mapbox_center={
                    "lat": float(map_df[lat_col].mean()),
                    "lon": float(map_df[lon_col].mean())
                },
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                legend_title_text="Pollution Risk"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📍 Location Details")

            options = map_df.index.tolist()

            selected_index = st.selectbox(
                "Choose a mapped record",
                options=options,
                format_func=lambda idx: (
                    f"{map_df.loc[idx, '_map_label']} — "
                    f"{map_df.loc[idx, '_display_risk']}"
                )
            )

            selected = map_df.loc[selected_index]

            d1, d2, d3 = st.columns(3)

            d1.metric("Risk Level", str(selected["_display_risk"]))

            pred_value = selected["_display_prediction"]
            d2.metric(
                "Predicted Pollution",
                f"{pred_value:,.2f}" if pd.notna(pred_value) else "N/A"
            )

            d3.metric("Latitude", f"{selected[lat_col]:.4f}")

            st.write("**Location / Area:**", selected["_map_label"])
            st.write("**Longitude:**", f"{selected[lon_col]:.4f}")

            with st.expander("View all available details"):
                st.dataframe(
                    pd.DataFrame([selected.drop(labels=[
                        "_map_color", "_map_size"
                    ], errors="ignore")]),
                    use_container_width=True,
                    hide_index=True
                )

            st.download_button(
                "⬇️ Download GIS Data",
                data=map_df.to_csv(index=False).encode("utf-8"),
                file_name="marine_pollution_gis_data.csv",
                mime="text/csv"
            )


# =========================================================
# AI POLLUTION PREDICTOR
# =========================================================
elif page == "AI Pollution Predictor":

    st.subheader("🤖 AI Pollution Predictor")

    st.write(
        "Enter environmental or geographic feature values to estimate "
        "pollution using the saved CatBoost model."
    )

    model_path = find_file("catboost_model.cbm")

    if not model_path:
        st.warning(
            "catboost_model.cbm was not found. Upload the trained model "
            "file to the project repository to enable model prediction."
        )

    elif test_df.empty:
        st.warning(
            "test_data.csv was not found or is empty. The trained model "
            "needs the same input feature columns used during training."
        )

    else:
        try:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor()
            model.load_model(model_path)

            expected_features = list(model.feature_names_)
            available_features = [
                col for col in expected_features if col in test_df.columns
            ]

            if len(available_features) != len(expected_features):
                st.warning(
                    "Some model feature columns are missing from test_data.csv. "
                    "Prediction may require the original training feature dataset."
                )

            feature_input = {}

            with st.form("prediction_form"):
                input_columns = st.columns(2)

                for i, feature in enumerate(expected_features):
                    with input_columns[i % 2]:
                        series = (
                            test_df[feature]
                            if feature in test_df.columns
                            else pd.Series(dtype=float)
                        )

                        if (
                            not series.empty
                            and (
                                pd.api.types.is_numeric_dtype(series)
                                or pd.to_numeric(series, errors="coerce").notna().mean() > 0.8
                            )
                        ):
                            numeric_values = pd.to_numeric(
                                series, errors="coerce"
                            ).dropna()

                            default_value = (
                                float(numeric_values.median())
                                if not numeric_values.empty else 0.0
                            )

                            feature_input[feature] = st.number_input(
                                feature,
                                value=default_value
                            )
                        else:
                            categories = (
                                series.dropna().astype(str).unique().tolist()
                                if not series.empty else []
                            )

                            if categories:
                                feature_input[feature] = st.selectbox(
                                    feature, categories
                                )
                            else:
                                feature_input[feature] = st.text_input(
                                    feature, value=""
                                )

                submitted = st.form_submit_button("Predict Pollution")

            if submitted:
                input_df = pd.DataFrame(
                    [feature_input],
                    columns=expected_features
                )

                prediction = float(model.predict(input_df)[0])
                risk = get_risk(prediction)

                st.success("Prediction completed.")

                r1, r2 = st.columns(2)

                r1.metric("Predicted Pollution Abundance", f"{prediction:,.3f}")

                r2.markdown(
                    f"""
                    <div style="
                        background:{RISK_COLORS[risk]};
                        padding:18px;
                        border-radius:12px;
                        text-align:center;
                        color:#061629;">
                        <h3 style="color:#061629 !important;">Risk Level</h3>
                        <h2 style="color:#061629 !important;">{risk}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "Risk thresholds are dashboard display thresholds; "
                    "they should be calibrated against the project's research criteria."
                )

                st.download_button(
                    "Download Prediction",
                    data=input_df.assign(
                        Predicted_Pollution=prediction,
                        Risk_Level=risk
                    ).to_csv(index=False).encode("utf-8"),
                    file_name="marine_pollution_prediction.csv",
                    mime="text/csv"
                )

        except ImportError:
            st.error(
                "CatBoost is not installed. Add catboost to requirements.txt "
                "and redeploy the app."
            )
        except Exception as error:
            st.error("The model could not generate a prediction.")
            st.caption(f"Technical detail: {error}")


# =========================================================
# AI EXPLAINABILITY / SHAP
# =========================================================
elif page == "AI Explainability":

    st.subheader("🧠 AI Explainability — SHAP Feature Importance")

    if shap_df.empty:
        st.warning(
            "SHAP importance CSV was not found. Upload top_10_shap_features.csv "
            "or shap_importance.csv to display saved feature importance."
        )

    else:
        feature_col = find_column(shap_df, [
            "Feature", "Feature_Name", "feature_name", "Column"
        ])

        importance_col = find_column(shap_df, [
            "SHAP_Importance", "Mean_SHAP", "Importance",
            "Mean_Abs_SHAP", "SHAP Value", "Value"
        ])

        if feature_col is None or importance_col is None:
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
            st.info(
                "The SHAP file is loaded. Check its column names to enable "
                "the feature-importance chart."
            )
        else:
            chart_df = shap_df.copy()
            chart_df[importance_col] = pd.to_numeric(
                chart_df[importance_col], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=[importance_col])
            chart_df = chart_df.sort_values(
                importance_col, ascending=True
            ).tail(20)

            fig = px.bar(
                chart_df,
                x=importance_col,
                y=feature_col,
                orientation="h",
                color=importance_col,
                color_continuous_scale=["#123456", "#087F8C", "#39D5BB"],
                title="Saved SHAP Feature Importance"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Feature Importance Table")
            st.dataframe(
                chart_df.sort_values(importance_col, ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This chart uses saved importance values from the supplied CSV; "
                "it does not recalculate SHAP values at runtime."
            )

            st.download_button(
                "Download SHAP Importance",
                data=shap_df.to_csv(index=False).encode("utf-8"),
                file_name="shap_feature_importance.csv",
                mime="text/csv"
            )


# =========================================================
# HOTSPOT ANALYSIS
# =========================================================
elif page == "Hotspot Analysis":

    st.subheader("🔥 Marine Pollution Hotspot Analysis")

    if hotspot_df.empty:
        st.warning(
            "Hotspot data was not found. Upload hotspot_results.csv "
            "or hotspot_ranking.csv to display recorded hotspot results."
        )

    else:
        st.markdown("### Hotspot Records")

        st.dataframe(
            hotspot_df,
            use_container_width=True,
            hide_index=True
        )

        hotspot_value_col = find_column(hotspot_df, [
            "Predicted_Pollution", "Pollution_Value", "Abundance",
            "Hotspot_Score", "Prediction", "Score"
        ])

        hotspot_name_col = find_column(hotspot_df, [
            "Location", "Location_Name", "Site_Name", "Area",
            "Region", "Country", "Sampling_Location"
        ])

        if hotspot_value_col:
            chart_df = hotspot_df.copy()
            chart_df[hotspot_value_col] = pd.to_numeric(
                chart_df[hotspot_value_col], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=[hotspot_value_col])

            if not chart_df.empty:
                if hotspot_name_col is None:
                    chart_df["_Location"] = [
                        f"Record {i + 1}" for i in range(len(chart_df))
                    ]
                    hotspot_name_col = "_Location"

                chart_df = chart_df.nlargest(15, hotspot_value_col)

                fig = px.bar(
                    chart_df,
                    x=hotspot_value_col,
                    y=hotspot_name_col,
                    orientation="h",
                    color=hotspot_value_col,
                    color_continuous_scale=["#087F8C", "#FFD166", "#FF5C6C"],
                    title="Highest Recorded Hotspot Values"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Hotspot Data",
            data=hotspot_df.to_csv(index=False).encode("utf-8"),
            file_name="marine_pollution_hotspots.csv",
            mime="text/csv"
        )


# =========================================================
# CATBOOST VS HYBRID QUANTUM-CLASSICAL MODEL
# =========================================================
elif page == "CatBoost vs QML":

    st.subheader("⚛️ Classical AI vs Hybrid Quantum-Classical ML")

    if not comparison_df.empty:
        st.markdown("### Saved Model Comparison Results")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        model_col = find_column(comparison_df, [
            "Model", "Model_Name", "Algorithm"
        ])

        score_col = find_column(comparison_df, [
            "R2", "R²", "R2_Score", "R2 Score"
        ])

        if model_col and score_col:
            plot_df = comparison_df.copy()
            plot_df[score_col] = pd.to_numeric(
                plot_df[score_col], errors="coerce"
            )
            plot_df = plot_df.dropna(subset=[score_col])

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x=model_col,
                    y=score_col,
                    color=model_col,
                    title="Recorded R² Score Comparison"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "Using previously recorded model results because "
            "final_model_comparison.csv was not found."
        )

        fallback_df = pd.DataFrame({
            "Model": [
                "CatBoost",
                "Hybrid Quantum-Classical ML"
            ],
            "MAE": [629.2820, 1157.3159],
            "RMSE": [9351.6746, 11517.7038],
            "R2": [0.3340, -0.0102]
        })

        st.dataframe(
            fallback_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are previously recorded project results, not a fresh "
            "model evaluation performed by this dashboard."
        )

    st.markdown("### Metric Interpretation")

    st.markdown("""
    - **MAE:** Average absolute difference between actual and predicted values.
    - **RMSE:** Prediction error metric that penalizes large errors more heavily.
    - **R²:** Indicates how much variation is explained by the model; values below zero
      indicate performance worse than the mean-prediction baseline on that evaluation set.
    """)


# =========================================================
# RESEARCH DATA
# =========================================================
elif page == "Research Data":

    st.subheader("📁 Research Data & Downloads")

    files_to_show = [
        "gis_predictions.csv",
        "marine_pollution_GIS_results.csv",
        "dashboard_summary.csv",
        "top_10_shap_features.csv",
        "shap_importance.csv",
        "hotspot_results.csv",
        "hotspot_ranking.csv",
        "final_model_comparison.csv",
        "test_data.csv",
        "actual_values.csv",
        "prediction_risk_results.csv"
    ]

    available_count = 0

    for filename in files_to_show:
        path = find_file(filename)

        if path:
            available_count += 1

            with st.expander(f"📄 {filename}"):
                try:
                    df = load_csv(filename)
                    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
                    st.dataframe(
                        df.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        f"Download {filename}",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=filename,
                        mime="text/csv",
                        key=f"download_{filename}"
                    )

                except Exception as error:
                    st.error(f"Could not read this file: {error}")

    st.metric("Available Research CSV Files", available_count)

    if available_count == 0:
        st.warning(
            "No expected CSV files were found in the app folder. "
            "Check the repository file names and upload paths."
        )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#A9D6E5; padding:10px;">
        🌊 Marine Pollution Intelligence Dashboard<br>
        AI Prediction • SHAP Explainability • GIS • Hotspot Analysis
    </div>
    """,
    unsafe_allow_html=True
)
