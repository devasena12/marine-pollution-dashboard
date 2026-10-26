            "Importance", "importance", "mean_abs_shap", "MeanAbsSHAP"
        ])
        if feature_col and importance_col:
            view = shap_df[[feature_col, importance_col]].copy()
            view[importance_col] = pd.to_numeric(view[importance_col], errors="coerce")
            view = view.dropna().sort_values(importance_col, ascending=False).head(20)
            if not view.empty:
                fig = px.bar(view.sort_values(importance_col), x=importance_col, y=feature_col,
                             orientation="h", color=importance_col,
                             color_continuous_scale=["#0c6578", "#1bd4c4"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#eafaff", height=max(400, len(view)*28),
                                  coloraxis_showscale=False, xaxis_title="Saved feature importance", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(view.sort_values(importance_col, ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("SHAP file has no valid numeric importance values.")
        else:
            st.error("Could not identify feature and importance columns.")
            st.dataframe(shap_df.head(50), use_container_width=True)

# ---------- Hotspots ----------
elif page == "Hotspot Analysis":
    st.subheader("🔥 Marine Pollution Hotspot Analysis")
    source = gis_df.copy() if not gis_df.empty else pd.DataFrame()
    if source.empty or not source["_prediction"].notna().any():
        if not hotspot_df.empty:
            st.info("Showing saved hotspot output because GIS predictions are unavailable.")
            st.dataframe(hotspot_df, use_container_width=True, hide_index=True)
            st.download_button("Download hotspot CSV", hotspot_df.to_csv(index=False).encode("utf-8"),
                               file_name="marine_pollution_hotspots.csv", mime="text/csv")
        else:
            st.warning("No valid pollution prediction column or saved hotspot file was found.")
    else:
        percentile = st.slider("Hotspot threshold percentile", 80, 99, 95, 1)
        values = source["_prediction"].dropna()
        threshold = float(values.quantile(percentile / 100))
        hotspots = source[source["_prediction"] >= threshold].copy()
        st.metric("Candidate hotspot locations", f"{len(hotspots):,}")
        st.write(f"Pollution threshold at the {percentile}th percentile: **{threshold:,.3f}**")
        st.caption("These are candidate hotspots selected by prediction percentile, not a formal statistical cluster-significance test.")
        if lat_col and lon_col and not hotspots.empty:
            hotspots[lat_col] = pd.to_numeric(hotspots[lat_col], errors="coerce")
            hotspots[lon_col] = pd.to_numeric(hotspots[lon_col], errors="coerce")
            map_hot = hotspots.dropna(subset=[lat_col, lon_col])
            if not map_hot.empty:
                fig = px.scatter_mapbox(map_hot, lat=lat_col, lon=lon_col,
                    color="_prediction", color_continuous_scale=["#fb923c", "#fb334f"],
                    hover_data=[c for c in [pred_col, risk_col] if c and c in map_hot.columns],
                    zoom=1, height=450, mapbox_style="carto-darkmatter")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#eafaff", margin=dict(l=0,r=0,t=5,b=0))
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hotspots.sort_values("_prediction", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download hotspot CSV", hotspots.to_csv(index=False).encode("utf-8"),
                           file_name="marine_pollution_hotspots.csv", mime="text/csv")

# ---------- CatBoost vs QML ----------
elif page == "CatBoost vs QML":
    st.subheader("⚛️ Quantum–Classical Model Comparison")
    if not compare_df.empty:
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        numeric_cols = [c for c in compare_df.columns if pd.api.types.is_numeric_dtype(compare_df[c])]
        if numeric_cols:
            metric = st.selectbox("Metric to visualize", numeric_cols)
            name_col = first_existing(compare_df, ["Model", "model", "Algorithm", "algorithm", "Method"])
            if name_col:
                fig = px.bar(compare_df, x=name_col, y=metric, color=name_col, text=metric)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#eafaff", height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Showing the previously recorded evaluation values. Replace them if your latest experiment produced different results.")
        comparison = pd.DataFrame({
            "Model": ["CatBoost", "Hybrid Quantum–Classical"],
            "MAE": [629.2820, 1157.3159],
            "RMSE": [9351.6746, 11517.7038],
            "R²": [0.3340, -0.0102],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        metric = st.selectbox("Metric to visualize", ["MAE", "RMSE", "R²"])
        fig = px.bar(comparison, x="Model", y=metric, color="Model",
                     color_discrete_map={"CatBoost": "#27c8c1", "Hybrid Quantum–Classical": "#f59e70"},
                     text=metric)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eafaff", height=380, showlegend=False)
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("On these recorded results, CatBoost has lower MAE/RMSE and higher R². This is specific to the current evaluation split.")

# ---------- Research data ----------
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
        ("Actual values", actual_df, "actual_values.csv"),
    ]
    for label, frame, filename in files:
        with st.expander(f"{label} · {len(frame):,} rows"):
            if frame.empty:
                st.caption("File not found or empty.")
            else:
                st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {filename}", frame.to_csv(index=False).encode("utf-8"),
                    file_name=filename, mime="text/csv", key=f"download_{filename}"
                )

st.markdown("---")
st.caption("Marine Pollution Intelligence V2 · New color test build · Research prototype · No live ocean sensor feed is connected.")
