import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ============================================================
# MARINE POLLUTION INTELLIGENCE DASHBOARD  (full project)
# Pages (left sidebar): Overview, Hotspot Map, Risk Analysis,
#                       Trends & Categories, Prediction, Data Explorer
# Put this file in the same folder as your CSV.
# ============================================================

st.set_page_config(
    page_title="Marine Pollution Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEVELS = ["Low", "Moderate", "High", "Critical"]
LEVEL_COLORS = {
    "Low": "#22c55e",       # green
    "Moderate": "#facc15",  # yellow
    "High": "#f97316",      # orange
    "Critical": "#ef4444",  # red
}

# ---------------- Theme ----------------
st.markdown("""
<style>
.stApp {
 background: radial-gradient(ellipse at 12% 0%, rgba(0,190,210,.13), transparent 35%),
 linear-gradient(145deg,#04111f 0%,#061d31 48%,#082b40 100%);
 color:#eafaff;
}
[data-testid="stHeader"] {background:rgba(4,17,31,.85);}
[data-testid="stSidebar"] {
 background:linear-gradient(180deg,#03101d 0%,#08283b 100%);
 border-right:1px solid rgba(52,211,214,.25);
}
h1,h2,h3,h4,p,label,span,li {color:#eafaff;}
.eyebrow {color:#50e0dc;font-size:.75rem;font-weight:800;letter-spacing:2px;}
.marine-subtitle {color:#7ddfe4;font-size:1rem;margin-top:-8px;margin-bottom:14px;}
.level-card {
 background:linear-gradient(145deg,rgba(11,52,72,.98),rgba(5,27,43,.98));
 border-radius:16px;padding:16px 18px;min-height:120px;
 box-shadow:0 8px 26px rgba(0,0,0,.15);
}
.level-name {font-size:.85rem;font-weight:800;letter-spacing:1px;text-transform:uppercase;}
.level-pct {color:#ffffff;font-size:2.1rem;font-weight:850;line-height:1.2;}
.level-count {color:#a8cbd6;font-size:.8rem;}
.kpi-card {
 background:linear-gradient(145deg,rgba(11,52,72,.98),rgba(5,27,43,.98));
 border:1px solid rgba(62,218,218,.30);border-radius:14px;
 padding:14px 16px;min-height:96px;
}
.kpi-label {color:#a8cbd6;font-size:.78rem;letter-spacing:.4px;}
.kpi-value {color:#ffffff;font-size:1.55rem;font-weight:800;overflow-wrap:anywhere;}
.kpi-foot {color:#62d9d4;font-size:.72rem;}
.stButton button,.stDownloadButton button {
 background:linear-gradient(90deg,#0d9eaa,#16c5bc);color:#041a26;
 border:0;border-radius:9px;font-weight:750;
}
div[data-testid="stDataFrame"] {border:1px solid rgba(89,202,211,.45);border-radius:10px;}
div[data-testid="stAlert"] {background:#092d43!important;color:#ffffff!important;
 border:1px solid #21a6b4!important;border-radius:12px!important;}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {color:#ffffff!important;}
.stCaption, [data-testid="stCaptionContainer"] {color:#c2e8f2!important;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#092f45,#061d30)!important;
 border:1px solid #1b8292!important;border-radius:14px!important;padding:14px!important;}
div[data-testid="stMetricLabel"] p {color:#c5f3ff!important;font-weight:700!important;}
div[data-testid="stMetricValue"] div {color:#ffffff!important;font-weight:850!important;}
</style>
""", unsafe_allow_html=True)


# ---------------- Helpers ----------------
def show_df(d, **kw):
    try:
        st.dataframe(d, width="stretch", **kw)
    except TypeError:
        st.dataframe(d, use_container_width=True, **kw)


def style_fig(fig, height=340):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"), margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(font=dict(color="#ffffff")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.10)")
    return fig


def kpi(col, label, value, foot=""):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def find_file(candidates):
    for filename in candidates:
        path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def read_csv(path):
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


GIS_PATH = find_file([
    "gis_predictions.csv",
    "marine_pollution_GIS_results.csv",
    "hotspot_results.csv",
    "FINAL_MARINE_POLLUTION_RESULTS.csv",
])


def pick_col(df, names, exclude=()):
    lower = {c.lower().strip(): c for c in df.columns if c not in exclude}
    for n in names:
        if n in lower:
            return lower[n]
    for n in names:
        for lc, orig in lower.items():
            if n in lc:
                return orig
    return None


def normalize_level(value):
    s = str(value).strip().lower()
    if s in ("nan", "none", ""):
        return None
    if "crit" in s or "severe" in s or "extreme" in s or s == "3":
        return "Critical"
    if "high" in s or s == "2":
        return "High"
    if "mod" in s or "med" in s or s == "1":
        return "Moderate"
    if "low" in s or "safe" in s or s == "0":
        return "Low"
    return None


def is_numeric_col(df, col, min_frac=0.5):
    if col is None:
        return False
    return pd.to_numeric(df[col], errors="coerce").notna().mean() >= min_frac


def prepare(df):
    df = df.copy()
    lat_c = pick_col(df, ["latitude", "lat"])
    lon_c = pick_col(df, ["longitude", "lon", "lng", "long"], exclude=(lat_c,))
    level_c = pick_col(df, ["risk_level", "pollution_level", "risk_category",
                            "severity", "hotspot_level", "category", "risk", "level"],
                       exclude=(lat_c, lon_c))
    score_c = pick_col(df, ["predicted", "prediction", "pollution_index",
                            "risk_score", "score", "pollution", "value"],
                       exclude=(lat_c, lon_c, level_c))
    if not is_numeric_col(df, score_c):
        score_c = None

    if lat_c and lon_c:
        df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
        df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")

    risk = None
    if level_c is not None:
        risk = df[level_c].map(normalize_level)
        if risk.notna().sum() == 0:
            risk = None
    if score_c is None:
        num_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in (lat_c, lon_c)]
        score_c = num_cols[-1] if num_cols else None
    if score_c is not None:
        df[score_c] = pd.to_numeric(df[score_c], errors="coerce")

    if risk is None:
        if score_c is not None and df[score_c].notna().sum() > 3:
            q = df[score_c].quantile([0.25, 0.5, 0.75]).values
            edges = sorted(set([-np.inf] + list(q) + [np.inf]))
            labels = LEVELS[:len(edges) - 1]
            risk = pd.cut(df[score_c], bins=edges, labels=labels).astype(object)
        else:
            risk = pd.Series([None] * len(df), index=df.index)

    df["Risk Level"] = risk
    df = df[df["Risk Level"].notna()].copy()
    return df, lat_c, lon_c, score_c


def find_date_col(df):
    for name in ["date", "datetime", "timestamp", "year", "month", "time"]:
        c = pick_col(df, [name])
        if c is not None:
            return c
    return None


# ---------------- Load data ----------------
raw = read_csv(GIS_PATH)
if raw.empty:
    st.markdown('<div class="eyebrow">OCEAN MONITORING</div>', unsafe_allow_html=True)
    st.title("🌊 Marine Pollution Intelligence Dashboard")
    st.warning("No data CSV found next to app.py. Upload your results CSV below.")
    up = st.file_uploader("Upload results CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up, low_memory=False)
if raw.empty:
    st.info("Expected file names: gis_predictions.csv, marine_pollution_GIS_results.csv, "
            "hotspot_results.csv or FINAL_MARINE_POLLUTION_RESULTS.csv")
    st.stop()

df, LAT, LON, SCORE = prepare(raw)
if df.empty:
    st.error("Could not determine pollution levels. Add a column like risk_level "
             "(Low/Moderate/High/Critical) or a numeric score column.")
    st.stop()

# ---------------- Sidebar: navigation + filters ----------------
st.sidebar.markdown("## 🌊 Marine Pollution")
st.sidebar.caption("Intelligence Dashboard")
PAGES = [
    "🏠 Overview",
    "🗺️ Hotspot Map",
    "📊 Risk Analysis",
    "📈 Trends & Categories",
    "🤖 Prediction",
    "🗂️ Data Explorer",
]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
selected = st.sidebar.multiselect("Risk levels", LEVELS, default=LEVELS)
data = df[df["Risk Level"].isin(selected)] if selected else df.iloc[0:0]
st.sidebar.caption(f"{len(data):,} of {len(df):,} records shown")

# Level statistics (always from the filtered data)
counts = data["Risk Level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
total = int(counts.sum())
pcts = (counts / total * 100).round(1) if total else counts.astype(float)

st.markdown('<div class="eyebrow">OCEAN MONITORING</div>', unsafe_allow_html=True)
st.title("🌊 Marine Pollution Intelligence Dashboard")
st.markdown('<div class="marine-subtitle">Pollution risk levels, hotspots, trends and prediction</div>',
            unsafe_allow_html=True)

if data.empty:
    st.warning("No records match the selected risk levels. Pick at least one level in the sidebar.")
    st.stop()


def level_cards():
    cols = st.columns(4)
    for col, lvl in zip(cols, LEVELS):
        color = LEVEL_COLORS[lvl]
        col.markdown(
            f"""<div class="level-card" style="border:1px solid {color};border-left:8px solid {color};">
            <div class="level-name" style="color:{color};">● {lvl}</div>
            <div class="level-pct">{pcts[lvl]}%</div>
            <div class="level-count">{counts[lvl]:,} of {total:,} records</div>
            </div>""",
            unsafe_allow_html=True,
        )


# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == PAGES[0]:
    level_cards()
    st.write("")

    k = st.columns(4)
    kpi(k[0], "TOTAL RECORDS", f"{total:,}", "after filters")
    if LAT and LON:
        n_loc = data[[LAT, LON]].dropna().drop_duplicates().shape[0]
        kpi(k[1], "UNIQUE LOCATIONS", f"{n_loc:,}", "distinct lat/lon points")
    else:
        kpi(k[1], "UNIQUE LOCATIONS", "n/a", "no coordinates")
    if SCORE:
        kpi(k[2], f"AVERAGE {SCORE.upper()}"[:34], f"{data[SCORE].mean():.2f}", "mean value")
        kpi(k[3], f"MAXIMUM {SCORE.upper()}"[:34], f"{data[SCORE].max():.2f}", "worst reading")
    else:
        kpi(k[2], "AVERAGE SCORE", "n/a", "")
        kpi(k[3], "MAXIMUM SCORE", "n/a", "")

    st.subheader("Pollution Flow: Total → Risk Level")
    labels = [f"All Records ({total:,})"] + [f"{l}: {pcts[l]}%" for l in LEVELS]
    sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=["#38bdf8"] + [LEVEL_COLORS[l] for l in LEVELS],
                  pad=28, thickness=26, line=dict(color="#04111f", width=1)),
        link=dict(source=[0] * 4, target=[1, 2, 3, 4],
                  value=[int(v) for v in counts.values],
                  color=[LEVEL_COLORS[l] + "99" for l in LEVELS]),
    ))
    sankey.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                         paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=14))
    st.plotly_chart(sankey)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Share by Level")
        pie = go.Figure(go.Pie(
            labels=LEVELS, values=counts.values, hole=0.55, sort=False,
            marker=dict(colors=[LEVEL_COLORS[l] for l in LEVELS]),
            textinfo="label+percent", textfont=dict(size=14, color="#041a26")))
        st.plotly_chart(style_fig(pie))
    with c2:
        st.subheader("Count by Level")
        bar = go.Figure(go.Bar(
            x=LEVELS, y=counts.values, marker_color=[LEVEL_COLORS[l] for l in LEVELS],
            text=[f"{p}%" for p in pcts.values], textposition="outside",
            textfont=dict(color="#ffffff", size=14)))
        st.plotly_chart(style_fig(bar))

# =====================================================================
# PAGE 2: HOTSPOT MAP
# =====================================================================
elif page == PAGES[1]:
    st.subheader("Hotspot Map")
    if not (LAT and LON):
        st.info("No latitude/longitude columns found, so the map is not available.")
    else:
        mc1, mc2 = st.columns([1, 1])
        mode = mc1.radio("Map style", ["Risk markers", "Heatmap"], horizontal=True)
        max_pts = mc2.slider("Max map points", 200, 5000, 2000, step=200)
        mdf = data.dropna(subset=[LAT, LON])
        if len(mdf) > max_pts:
            mdf = mdf.sample(max_pts, random_state=42)
        if mdf.empty:
            st.info("No points to show for the selected levels.")
        else:
            m = folium.Map(location=[float(mdf[LAT].mean()), float(mdf[LON].mean())],
                           zoom_start=4, tiles="CartoDB dark_matter")
            if mode == "Heatmap":
                if SCORE and mdf[SCORE].notna().any():
                    s = mdf[SCORE].fillna(mdf[SCORE].min())
                    rng = (s.max() - s.min()) or 1.0
                    w = ((s - s.min()) / rng).clip(0.05, 1.0)
                else:
                    w = pd.Series(1.0, index=mdf.index)
                HeatMap(list(zip(mdf[LAT], mdf[LON], w)), radius=16, blur=20,
                        gradient={0.2: "#22c55e", 0.5: "#facc15",
                                  0.75: "#f97316", 1.0: "#ef4444"}).add_to(m)
            else:
                for _, r in mdf.iterrows():
                    lvl = r["Risk Level"]
                    tip = lvl
                    if SCORE and pd.notna(r.get(SCORE)):
                        tip += f" | {SCORE}: {r[SCORE]:.2f}"
                    folium.CircleMarker(
                        location=[r[LAT], r[LON]],
                        radius=9 if lvl == "Critical" else 7 if lvl == "High" else 5,
                        color=LEVEL_COLORS[lvl], fill=True, fill_color=LEVEL_COLORS[lvl],
                        fill_opacity=0.85, weight=1, tooltip=tip).add_to(m)
            legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;'
                      'background:#082b40;color:#fff;padding:10px 14px;border:1px solid #21a6b4;'
                      'border-radius:10px;font-size:13px;"><b>Risk Level</b><br>' +
                      "".join(f'<span style="color:{LEVEL_COLORS[l]};">●</span> {l}<br>'
                              for l in LEVELS) + "</div>")
            m.get_root().html.add_child(folium.Element(legend))
            try:
                st_folium(m, height=560, use_container_width=True, returned_objects=[])
            except TypeError:
                st_folium(m, height=560, width=1100, returned_objects=[])
            st.caption(f"Showing {len(mdf):,} points.")
        level_cards()

# =====================================================================
# PAGE 3: RISK ANALYSIS
# =====================================================================
elif page == PAGES[2]:
    st.subheader("Risk Analysis")
    level_cards()
    st.write("")
    if SCORE:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Distribution of {SCORE}**")
            h = px.histogram(data, x=SCORE, color="Risk Level", nbins=40,
                             color_discrete_map=LEVEL_COLORS,
                             category_orders={"Risk Level": LEVELS})
            st.plotly_chart(style_fig(h))
        with c2:
            st.markdown(f"**{SCORE} by risk level**")
            b = px.box(data, x="Risk Level", y=SCORE, color="Risk Level",
                       color_discrete_map=LEVEL_COLORS,
                       category_orders={"Risk Level": LEVELS})
            st.plotly_chart(style_fig(b))

        st.markdown("**Summary statistics by level**")
        summ = (data.groupby("Risk Level")[SCORE]
                .agg(["count", "mean", "min", "max"]).reindex(LEVELS).round(3))
        show_df(summ)

        st.markdown(f"**Top 10 highest {SCORE} records**")
        show_df(data.sort_values(SCORE, ascending=False).head(10))
    else:
        st.info("No numeric score column detected for detailed analysis.")

    num = data.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        st.markdown("**Correlation between numeric columns**")
        corr = num.corr().round(2)
        hm = px.imshow(corr, text_auto=True, aspect="auto",
                       color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1)
        st.plotly_chart(style_fig(hm, height=460))

# =====================================================================
# PAGE 4: TRENDS & CATEGORIES
# =====================================================================
elif page == PAGES[3]:
    st.subheader("Trends & Categories")

    date_c = find_date_col(data)
    st.markdown("**Risk levels over time**")
    if date_c is None:
        st.info("No date / year / month column found, so the time trend is not available.")
    else:
        t = data.copy()
        if "year" in date_c.lower() and pd.api.types.is_numeric_dtype(t[date_c]):
            t["_period"] = t[date_c].astype("Int64").astype(str)
        else:
            parsed = pd.to_datetime(t[date_c], errors="coerce")
            if parsed.notna().sum() > 0:
                t["_period"] = parsed.dt.to_period("M").astype(str)
            else:
                t["_period"] = t[date_c].astype(str)
        t = t[t["_period"].notna() & (t["_period"] != "NaT")]
        trend = (t.groupby(["_period", "Risk Level"]).size()
                 .reset_index(name="Records").sort_values("_period"))
        if trend.empty:
            st.info("Could not build a time trend from the date column.")
        else:
            fig = px.line(trend, x="_period", y="Records", color="Risk Level", markers=True,
                          color_discrete_map=LEVEL_COLORS,
                          category_orders={"Risk Level": LEVELS},
                          labels={"_period": date_c})
            st.plotly_chart(style_fig(fig, height=380))

    st.markdown("**Risk level by category**")
    cat_cols = [c for c in data.columns
                if c != "Risk Level" and data[c].dtype == object
                and 2 <= data[c].nunique() <= 40]
    if not cat_cols:
        st.info("No suitable category columns (e.g. region, pollutant type) found.")
    else:
        cat = st.selectbox("Group by", cat_cols)
        ct = pd.crosstab(data[cat], data["Risk Level"]).reindex(columns=LEVELS, fill_value=0)
        ct_pct = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
        long = ct_pct.reset_index().melt(id_vars=cat, var_name="Risk Level", value_name="Percent")
        fig = px.bar(long, x=cat, y="Percent", color="Risk Level",
                     color_discrete_map=LEVEL_COLORS,
                     category_orders={"Risk Level": LEVELS}, barmode="stack")
        st.plotly_chart(style_fig(fig, height=420))
        show_df(ct)

# =====================================================================
# PAGE 5: PREDICTION
# =====================================================================
elif page == PAGES[4]:
    st.subheader("Pollution Prediction")
    if not SCORE:
        st.info("A numeric score column is needed to train a prediction model.")
        st.stop()

    feats_all = [c for c in df.select_dtypes(include=np.number).columns if c != SCORE]
    feats_all = feats_all[:12]
    if len(feats_all) < 1:
        st.info("No numeric feature columns available for prediction.")
        st.stop()

    @st.cache_resource(show_spinner="Training model...")
    def train_model(frame, features, target):
        d = frame[features + [target]].dropna()
        if len(d) < 30:
            return None
        if len(d) > 20000:
            d = d.sample(20000, random_state=42)
        X, y = d[features], d[target]
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            return dict(model=mdl, r2=r2_score(yte, pred), mae=mean_absolute_error(yte, pred),
                        imp=mdl.feature_importances_, name="Random Forest")
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor
            n = int(len(d) * 0.8)
            mdl = CatBoostRegressor(iterations=300, verbose=0, random_seed=42)
            mdl.fit(X.iloc[:n], y.iloc[:n])
            pred = mdl.predict(X.iloc[n:])
            yt = y.iloc[n:]
            r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
            return dict(model=mdl, r2=r2, mae=float(np.abs(yt - pred).mean()),
                        imp=mdl.get_feature_importance(), name="CatBoost")
        except ImportError:
            return None

    res = train_model(df, feats_all, SCORE)
    if res is None:
        st.warning("Could not train a model. Install scikit-learn (add it to requirements.txt) "
                   "or ensure there are at least 30 complete rows.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", res["name"])
    m2.metric("R² (test)", f"{res['r2']:.3f}")
    m3.metric("MAE (test)", f"{res['mae']:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Enter values**")
        vals = {}
        for f in feats_all:
            lo, hi = float(df[f].min()), float(df[f].max())
            med = float(df[f].median())
            if lo == hi:
                vals[f] = lo
                st.caption(f"{f}: constant = {lo}")
            else:
                vals[f] = st.slider(f, lo, hi, med)
        pred_val = float(res["model"].predict(pd.DataFrame([vals])[feats_all])[0])

    q1, q2, q3 = df[SCORE].quantile([0.25, 0.5, 0.75]).values
    lvl = ("Low" if pred_val <= q1 else "Moderate" if pred_val <= q2
           else "High" if pred_val <= q3 else "Critical")
    smin, smax = float(df[SCORE].min()), float(df[SCORE].max())
    with right:
        st.markdown(f"**Predicted {SCORE}**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pred_val,
            gauge=dict(axis=dict(range=[smin, smax]),
                       bar=dict(color="#ffffff"),
                       steps=[dict(range=[smin, q1], color=LEVEL_COLORS["Low"]),
                              dict(range=[q1, q2], color=LEVEL_COLORS["Moderate"]),
                              dict(range=[q2, q3], color=LEVEL_COLORS["High"]),
                              dict(range=[q3, smax], color=LEVEL_COLORS["Critical"])])))
        st.plotly_chart(style_fig(gauge, height=300))
        st.markdown(
            f'<div class="level-card" style="border:1px solid {LEVEL_COLORS[lvl]};'
            f'border-left:8px solid {LEVEL_COLORS[lvl]};min-height:80px;">'
            f'<div class="level-name" style="color:{LEVEL_COLORS[lvl]};">Predicted risk level</div>'
            f'<div class="level-pct">{lvl}</div></div>', unsafe_allow_html=True)

    st.markdown("**Feature importance**")
    imp = pd.DataFrame({"Feature": feats_all, "Importance": res["imp"]}).sort_values("Importance")
    fi = px.bar(imp, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=["#16c5bc"])
    st.plotly_chart(style_fig(fi, height=380))

# =====================================================================
# PAGE 6: DATA EXPLORER
# =====================================================================
else:
    st.subheader("Data Explorer")
    search = st.text_input("Search text in any column")
    view = data
    if search:
        mask = view.astype(str).apply(lambda s: s.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.caption(f"{len(view):,} rows")

    def color_level(v):
        c = LEVEL_COLORS.get(v)
        return f"background-color:{c};color:#041a26;font-weight:700;" if c else ""

    show = view.head(1000)
    try:
        show_df(show.style.map(color_level, subset=["Risk Level"]), height=480)
    except Exception:
        try:
            show_df(show.style.applymap(color_level, subset=["Risk Level"]), height=480)
        except Exception:
            show_df(show, height=480)

    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                       "filtered_marine_pollution.csv", "text/csv")
