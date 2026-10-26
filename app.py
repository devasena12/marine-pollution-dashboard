
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Optional libraries
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False

st.set_page_config(
    page_title="Marine Pollution Intelligence",
    page_icon="🌊",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

COLORS = {
    "Low": "#39D98A",
    "Moderate": "#FACC15",
    "High": "#F97316",
    "Critical": "#FF5252"
}

# ---------- Theme ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#06182c,#082b46,#041323);
    color: #eafaff;
}
h1,h2,h3,p,label {
    color: #eafaff !important;
}
[data-testid="stMetric"] {
    background: #10334a;
    border: 1px solid #236079;
    border-radius: 14px;
    padding: 15px;
}
[data-testid="stMetricValue"] {
    color: #38dfca;
}
.stButton>button {
    background: #16b8ae;
    color: #041923;
    border-radius: 10px;
    font-weight: bold;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #236079;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def load_csv(names):
    for name in names:
        path = BASE / name
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not read {name}: {exc}")
    return pd.DataFrame()

def find_col(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for col in df.columns:
        for candidate in candidates:
            if candidate.lower() in str(col).lower():
                return col
    return None

def numeric_series(df, candidates):
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(dtype=float), None
    return pd.to_numeric(df[col], errors="coerce"), col

def get_coordinates(df):
    lat = find_col(df, ["Latitude", "Lat", "latitude"])
    lon = find_col(df, ["Longitude", "Lon", "Long", "longitude"])
    return lat, lon

def get_prediction_col(df):
    return find_col(df, [
        "_prediction", "Predicted_Pollution",
        "Predicted_Abundance", "Prediction",
        "Plastic_abundance_converted",
        "predicted", "prediction"
    ])

def risk_label(value, q1, q2, q3):
    if pd.isna(value):
        return "Unknown"
    if value <= q1:
        return "Low"
    if value <= q2:
        return "Moderate"
    if value <= q3:
        return "High"
    return "Critical"

def add_risk(df, pred_col):
    result = df.copy()
    result["_prediction"] = pd.to_numeric(
        result[pred_col], errors="coerce"
    )
    values = result["_prediction"].dropna()

    if values.empty:
        result["Risk"] = "Unknown"
        return result

    q1, q2, q3 = values.quantile([0.25, 0.50, 0.75])
    result["Risk"] = result["_prediction"].apply(
        lambda x: risk_label(x, q1, q2, q3)
    )
    return result

def card(title, value, color):
    st.markdown(
        f"""
        <div style="
            background:#10334a;
            border:1px solid {color};
            border-radius:14px;
            padding:16px;
            margin-bottom:8px;">
            <div style="font-size:14px;color:#d5edf5">{title}</div>
            <div style="font-size:28px;font-weight:700;color:{color}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Load project files ----------
gis_df = load_csv([
    "gis_predictions.csv",
    "marine_pollution_GIS_results.csv",
    "hotspot_results.csv"
])

summary_df = load_csv(["dashboard_summary.csv"])

shap_df = load_csv([
    "top_10_shap_features.csv",
    "shap_importance.csv"
])

hotspot_df = load_csv([
    "hotspot_results.csv",
    "hotspot_ranking.csv"
])

compare_df = load_csv(["final_model_comparison.csv"])
test_df = load_csv(["test_data.csv"])
actual_df = load_csv(["actual_values.csv"])
risk_df = load_csv(["prediction_risk_results.csv"])

# Find prediction column
pred_col = get_prediction_col(gis_df)
lat_col, lon_col = get_coordinates(gis_df)

if pred_col:
    gis_df = add_risk(gis_df, pred_col)
else:
    gis_df["_prediction"] = np.nan
    gis_df["Risk"] = "Unknown"

# ---------- Sidebar ----------
st.sidebar.title("🌊 Marine Intelligence")
st.sidebar.caption("AI-powered pollution research dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Ocean GIS Explorer",
        "AI Pollution Predictor",
        "AI Explainability",
        "Hotspot Analysis",
        "CatBoost vs QML",
        "Research Data"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · CSV/model based")

# ---------- Header ----------
st.title("🌊 Marine Pollution Intelligence")
st.caption(
    "Microplastic abundance prediction, spatial risk assessment "
    "and explainable machine learning"
)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Project Overview")

    values, _ = numeric_series(
        gis_df,
        ["_prediction", "Predicted_Pollution", "Prediction"]
    )

    if values.empty:
        values, _ = numeric_series(
            summary_df,
            ["Average_Prediction", "avg_predicted_pollution",
             "Mean_Prediction"]
        )

    total = len(gis_df)
    valid = values.dropna()

    if not valid.empty:
        avg = valid.mean()
        minimum = valid.min()
        maximum = valid.max()
    else:
        avg = minimum = maximum = None

    a, b, c, d = st.columns(4)
    a.metric("Data locations", f"{total:,}")

    b.metric(
        "Average pollution",
        f"{avg:,.2f}" if avg is not None else "N/A"
    )
    c.metric(
        "Minimum pollution",
        f"{minimum:,.2f}" if minimum is not None else "N/A"
    )
    d.metric(
        "Maximum pollution",
        f"{maximum:,.2f}" if maximum is not None else "N/A"
    )

    st.markdown("### Pollution risk distribution")

    if not valid.empty:
        risk_counts = gis_df["Risk"].value_counts()
        order = ["Low", "Moderate", "High", "Critical"]

        counts = pd.DataFrame({
            "Risk": order,
            "Locations": [
                int(risk_counts.get(x, 0)) for x in order
            ]
        })
        counts["Percentage"] = (
            counts["Locations"] / max(counts["Locations"].sum(), 1) * 100
        )

        cols = st.columns(4)
        for i, name in enumerate(order):
            pct = counts.loc[
                counts["Risk"] == name, "Percentage"
            ].iloc[0]
            card(name, f"{pct:.1f}%", COLORS[name])
            with cols[i]:
                st.caption(f"{counts['Locations'].iloc[i]:,} locations")

        fig = px.bar(
            counts,
            x="Risk",
            y="Percentage",
            text=counts["Percentage"].map(lambda x: f"{x:.1f}%"),
            color="Risk",
            color_discrete_map=COLORS,
            category_orders={"Risk": order}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eafaff",
            yaxis_title="Locations (%)",
            xaxis_title="Risk category",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "Prediction data is unavailable. Check your GIS CSV file "
            "and confirm that it contains a valid prediction column."
        )

    st.markdown("### Research workflow")
    st.write(
        "Dataset → Preprocessing → Feature Engineering → "
        "CatBoost Prediction → SHAP Explainability → "
        "GIS Mapping → Hotspot Analysis → Risk Assessment"
    )

    if not gis_df.empty:
        with st.expander("Preview GIS dataset"):
            st.dataframe(gis_df.head(100), use_container_width=True)

# ---------- GIS Explorer ----------
elif page == "Ocean GIS Explorer":
    st.subheader("🗺️ Interactive Ocean Pollution Map")

    if gis_df.empty:
        st.warning("GIS data file was not found or contains no rows.")
    elif not lat_col or not lon_col:
        st.error("Latitude/longitude columns were not identified.")
        st.dataframe(gis_df.head(100), use_container_width=True)
    else:
        map_df = gis_df.copy()
        map_df[lat_col] = pd.to_numeric(
            map_df[lat_col], errors="coerce"
        )
        map_df[lon_col] = pd.to_numeric(
            map_df[lon_col], errors="coerce"
        )
        map_df = map_df.dropna(subset=[lat_col, lon_col])

        if map_df.empty:
            st.warning("No valid coordinates are available.")
        else:
            risk_filter = st.multiselect(
                "Filter risk categories",
                ["Low", "Moderate", "High", "Critical"],
                default=["Low", "Moderate", "High", "Critical"]
            )
            map_df = map_df[map_df["Risk"].isin(risk_filter)]

            if not map_df.empty:
                map_df["_map_color"] = map_df["Risk"].map(COLORS)

                hover_cols = [
                    c for c in [
                        "Risk", "_prediction", pred_col
                    ] if c and c in map_df.columns
                ]

                fig = px.scatter_mapbox(
                    map_df,
                    lat=lat_col,
                    lon=lon_col,
                    color="Risk",
                    hover_data=hover_cols,
                    color_discrete_map=COLORS,
                    zoom=1,
                    height=550,
                    mapbox_style="carto-darkmatter"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#eafaff",
                    margin=dict(l=0, r=0, t=5, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Select a point on the map to inspect its available "
                    "details using the chart hover information."
                )

                st.markdown("### Location details")
                location_options = map_df.index.tolist()
                selected = st.selectbox(
                    "Choose a mapped record",
                    location_options,
                    format_func=lambda i: (
                        f"Record {i} · {map_df.loc[i, 'Risk']}"
                    )
                )
                st.dataframe(
                    map_df.loc[[selected]].T.rename(
                        columns={selected: "Value"}
                    ),
                    use_container_width=True
                )
            else:
                st.info("No locations match the selected risk filters.")

# ---------- AI Predictor ----------
elif page == "AI Pollution Predictor":
    st.subheader("🤖 CatBoost AI Pollution Predictor")

    model_path = BASE / "catboost_model.cbm"

    if not CATBOOST_OK:
        st.warning(
            "CatBoost is not installed. Add catboost to requirements.txt."
        )
    elif not model_path.exists():
        st.warning(
            "catboost_model.cbm was not found beside app.py. "
            "Upload your trained model file to enable model predictions."
        )
    else:
        try:
            model = CatBoostRegressor()
            model.load_model(str(model_path))
            st.success("Trained CatBoost model loaded.")

            if test_df.empty:
                st.info(
                    "Upload test_data.csv with the same feature columns "
                    "used during model training."
                )
            else:
                st.write("Preview of available prediction features")
                st.dataframe(test_df.head(10), use_container_width=True)

                row_num = st.number_input(
                    "Select test record",
                    min_value=0,
                    max_value=len(test_df) - 1,
                    value=0,
                    step=1
                )

                if st.button("Predict pollution"):
                    row = test_df.iloc[[int(row_num)]]
                    prediction = float(model.predict(row)[0])
                    st.metric(
                        "Predicted pollution abundance",
                        f"{prediction:,.3f}"
                    )
                    st.caption(
                        "Prediction uses the selected test record and "
                        "the saved CatBoost model."
                    )
        except Exception as exc:
            st.error(f"Model prediction failed: {exc}")
            st.info(
                "Check that the model and test feature columns match."
            )

# ---------- SHAP Explainability ----------
elif page == "AI Explainability":
    st.subheader("🧠 Explainable AI — Feature Importance")

    if shap_df.empty:
        st.warning(
            "SHAP CSV not found. Add top_10_shap_features.csv "
            "or shap_importance.csv beside app.py."
        )
    else:
        feature_col = find_col(
            shap_df,
            ["Feature", "Feature_Name", "feature_name", "name"]
        )
        importance_col = find_col(
            shap_df,
            [
                "SHAP_Importance", "shap_importance", "Mean_SHAP",
                "mean_shap", "Importance", "mean_abs_shap"
            ]
        )

        if feature_col and importance_col:
            view = shap_df[[feature_col, importance_col]].copy()
            view[importance_col] = pd.to_numeric(
                view[importance_col], errors="coerce"
            )
            view = view.dropna().sort_values(
                importance_col, ascending=False
            ).head(20)

            if not view.empty:
                fig = px.bar(
                    view.sort_values(importance_col),
                    x=importance_col,
                    y=feature_col,
                    orientation="h",
                    color=importance_col,
                    color_continuous_scale=["#0c6578", "#1bd4c4"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#eafaff",
                    height=max(400, len(view) * 28),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    view, use_container_width=True, hide_index=True
                )
            else:
                st.info("No valid numeric importance values were found.")
        else:
            st.error("Feature/importance columns could not be identified.")
            st.dataframe(shap_df.head(50), use_container_width=True)

# ---------- Hotspot Analysis ----------
elif page == "Hotspot Analysis":
    st.subheader("🔥 Marine Pollution Hotspot Analysis")

    source = gis_df.copy()

    if source.empty or not source["_prediction"].notna().any():
        if not hotspot_df.empty:
            st.info("Showing the saved hotspot results.")
            st.dataframe(
                hotspot_df, use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download hotspot CSV",
                hotspot_df.to_csv(index=False).encode("utf-8"),
                "marine_pollution_hotspots.csv",
                "text/csv"
            )
        else:
            st.warning("No valid prediction or hotspot data was found.")
    else:
        percentile = st.slider(
            "Hotspot threshold percentile", 80, 99, 95
        )
        valid_values = source["_prediction"].dropna()
        threshold = float(valid_values.quantile(percentile / 100))
        hotspots = source[
            source["_prediction"] >= threshold
        ].copy()

        st.metric("Candidate hotspot locations", len(hotspots))
        st.write(
            f"Threshold at {percentile}th percentile: "
            f"**{threshold:,.3f}**"
        )
        st.caption(
            "Percentile-based candidate hotspots are not a formal "
            "statistical cluster-significance test."
        )

        if lat_col and lon_col and not hotspots.empty:
            hotspots[lat_col] = pd.to_numeric(
                hotspots[lat_col], errors="coerce"
            )
            hotspots[lon_col] = pd.to_numeric(
                hotspots[lon_col], errors="coerce"
            )
            hotspots = hotspots.dropna(subset=[lat_col, lon_col])

            if not hotspots.empty:
                fig = px.scatter_mapbox(
                    hotspots,
                    lat=lat_col,
                    lon=lon_col,
                    color="_prediction",
                    color_continuous_scale=["#fb923c", "#fb334f"],
                    zoom=1,
                    height=450,
                    mapbox_style="carto-darkmatter"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#eafaff"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(hotspots, use_container_width=True)
        st.download_button(
            "Download candidate hotspots",
            hotspots.to_csv(index=False).encode("utf-8"),
            "marine_pollution_hotspots.csv",
            "text/csv"
        )

# ---------- CatBoost vs QML ----------
elif page == "CatBoost vs QML":
    st.subheader("⚛️ Quantum–Classical Model Comparison")

    if not compare_df.empty:
        st.dataframe(
            compare_df, use_container_width=True, hide_index=True
        )

        numeric_cols = [
            c for c in compare_df.columns
            if pd.api.types.is_numeric_dtype(compare_df[c])
        ]
        name_col = find_col(
            compare_df, ["Model", "Algorithm", "Method"]
        )

        if numeric_cols and name_col:
            metric = st.selectbox(
                "Evaluation metric", numeric_cols
            )
            fig = px.bar(
                compare_df,
                x=name_col,
                y=metric,
                color=name_col,
                text=metric
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#eafaff"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        comparison = pd.DataFrame({
            "Model": [
                "CatBoost", "Hybrid Quantum–Classical"
            ],
            "MAE": [629.2820, 1157.3159],
            "RMSE": [9351.6746, 11517.7038],
            "R2": [0.3340, -0.0102]
        })

        st.caption(
            "These are previously recorded evaluation results. "
            "They are not recalculated by this dashboard."
        )
        st.dataframe(comparison, use_container_width=True)
        metric = st.selectbox(
            "Metric to visualize", ["MAE", "RMSE", "R2"]
        )

        fig = px.bar(
            comparison,
            x="Model",
            y=metric,
            color="Model",
            text=metric,
            color_discrete_map={
                "CatBoost": "#27c8c1",
                "Hybrid Quantum–Classical": "#f59e70"
            }
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#eafaff",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "On these recorded results, CatBoost performed better "
            "on MAE, RMSE and R2 for the evaluated split."
        )

# ---------- Research Data ----------
elif page == "Research Data":
    st.subheader("📁 Research Data & Downloads")

    files = [
        ("GIS predictions", gis_df, "gis_predictions.csv"),
        ("Dashboard summary", summary_df, "dashboard_summary.csv"),
        ("SHAP importance", shap_df, "shap_importance.csv"),
        ("Hotspot results", hotspot_df, "hotspot_results.csv"),
        ("Risk results", risk_df, "prediction_risk_results.csv"),
        ("Model comparison", compare_df, "final_model_comparison.csv"),
        ("Test features", test_df, "test_data.csv"),
        ("Actual values", actual_df, "actual_values.csv")
    ]

    for label, frame, filename in files:
        with st.expander(f"{label} · {len(frame):,} rows"):
            if frame.empty:
                st.caption("File not found or empty.")
            else:
                st.dataframe(
                    frame.head(100),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    f"Download {filename}",
                    frame.to_csv(index=False).encode("utf-8"),
                    filename,
                    "text/csv",
                    key=f"download_{filename}"
                )

st.markdown("---")
st.caption(
    "Marine Pollution Intelligence · Research prototype · "
    "No live ocean sensor feed is connected."
)
