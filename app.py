import os
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
 
# ============================================================
# MARINE POLLUTION INTELLIGENCE DASHBOARD
# Put this app.py in the same folder as your CSV file(s).
# ============================================================
 
st.set_page_config(
    page_title="Marine Pollution Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# ---------- Risk level colors (clearly different) ----------
LEVELS = ["Low", "Moderate", "High", "Critical"]
LEVEL_COLORS = {
    "Low": "#22c55e",       # green
    "Moderate": "#facc15",  # yellow
    "High": "#f97316",      # orange
    "Critical": "#ef4444",  # red
}
 
# ---------- Theme ----------
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
h1,h2,h3,p,label,span {color:#eafaff;}
.eyebrow {color:#50e0dc;font-size:.75rem;font-weight:800;letter-spacing:2px;}
.marine-subtitle {color:#7ddfe4;font-size:1rem;margin-top:-8px;}
.level-card {
 background:linear-gradient(145deg,rgba(11,52,72,.98),rgba(5,27,43,.98));
 border-radius:16px;padding:16px 18px;min-height:120px;
 box-shadow:0 8px 26px rgba(0,0,0,.15);
}
.level-name {font-size:.85rem;font-weight:800;letter-spacing:1px;text-transform:uppercase;}
.level-pct {color:#ffffff;font-size:2.1rem;font-weight:850;line-height:1.2;}
.level-count {color:#a8cbd6;font-size:.8rem;}
.stButton button,.stDownloadButton button {
 background:linear-gradient(90deg,#0d9eaa,#16c5bc);color:#041a26;
 border:0;border-radius:9px;font-weight:750;
}
div[data-testid="stDataFrame"] {border:1px solid rgba(89,202,211,.45);border-radius:10px;}
div[data-testid="stAlert"] {background:#092d43!important;color:#ffffff!important;
 border:1px solid #21a6b4!important;border-radius:12px!important;}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {color:#ffffff!important;}
.stCaption, [data-testid="stCaptionContainer"] {color:#c2e8f2!important;}
</style>
""", unsafe_allow_html=True)
 
 
# ---------- File helpers ----------
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
 
 
# ---------- Column detection ----------
def pick_col(df, names):
    lower = {c.lower().strip(): c for c in df.columns}
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
 
 
def prepare(df):
    df = df.copy()
    lat_c = pick_col(df, ["latitude", "lat"])
    lon_c = pick_col(df, ["longitude", "lon", "lng", "long"])
    level_c = pick_col(df, ["risk_level", "pollution_level", "risk_category",
                            "severity", "hotspot_level", "category", "risk", "level"])
    score_c = pick_col(df, ["predicted", "prediction", "pollution_index",
                            "risk_score", "score", "pollution", "value"])
 
    # Score column must be numeric
    if score_c is not None:
        df[score_c] = pd.to_numeric(df[score_c], errors="coerce")
 
    # Build Risk Level
    risk = None
    if level_c is not None:
        risk = df[level_c].map(normalize_level)
        if risk.notna().sum() == 0:
            risk = None
    if risk is None:
        if score_c is None:
            num_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in (lat_c, lon_c)]
            score_c = num_cols[-1] if num_cols else None
        if score_c is not None and df[score_c].notna().sum() > 3:
            q = df[score_c].quantile([0.25, 0.5, 0.75]).values
            risk = pd.cut(df[score_c], bins=[-np.inf, q[0], q[1], q[2], np.inf],
                          labels=LEVELS, duplicates="drop")
            risk = risk.astype(object)
        else:
            risk = pd.Series([None] * len(df), index=df.index)
 
    df["Risk Level"] = risk
    df = df[df["Risk Level"].notna()].copy()
 
    if lat_c and lon_c:
        df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
        df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
    return df, lat_c, lon_c, score_c
 
 
# ---------- Header ----------
st.markdown('<div class="eyebrow">OCEAN MONITORING</div>', unsafe_allow_html=True)
st.title("🌊 Marine Pollution Intelligence Dashboard")
st.markdown('<div class="marine-subtitle">Pollution risk levels, hotspots and distribution</div>',
            unsafe_allow_html=True)
 
# ---------- Load data ----------
raw = read_csv(GIS_PATH)
if raw.empty:
    st.warning("No data CSV found next to app.py. Upload one below.")
    up = st.file_uploader("Upload your results CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up, low_memory=False)
 
if raw.empty:
    st.info("Waiting for data. Expected file names: gis_predictions.csv, "
            "marine_pollution_GIS_results.csv, hotspot_results.csv or "
            "FINAL_MARINE_POLLUTION_RESULTS.csv")
    st.stop()
 
df, LAT, LON, SCORE = prepare(raw)
if df.empty:
    st.error("Could not determine pollution levels from this CSV. "
             "Add a column such as risk_level (Low/Moderate/High/Critical) "
             "or a numeric score column.")
    st.stop()
 
# ---------- Sidebar ----------
st.sidebar.header("Filters")
selected = st.sidebar.multiselect("Risk levels", LEVELS, default=LEVELS)
max_pts = st.sidebar.slider("Max map points", 200, 5000, 2000, step=200)
fdf = df[df["Risk Level"].isin(selected)] if selected else df.iloc[0:0]
 
# ---------- Percentage cards at the TOP ----------
counts = df["Risk Level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
total = int(counts.sum())
pcts = (counts / total * 100).round(1)
 
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
 
st.write("")
 
# ---------- Flow chart (Sankey) with percentages ----------
st.subheader("Pollution Flow: Total → Risk Level")
labels = [f"All Records ({total:,})"] + [f"{l}: {pcts[l]}%" for l in LEVELS]
node_colors = ["#38bdf8"] + [LEVEL_COLORS[l] for l in LEVELS]
sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(label=labels, color=node_colors, pad=28, thickness=26,
              line=dict(color="#04111f", width=1)),
    link=dict(source=[0] * 4, target=[1, 2, 3, 4],
              value=[max(int(v), 0) for v in counts.values],
              color=[LEVEL_COLORS[l].replace("#", "#") + "99" for l in LEVELS]),
))
sankey.update_layout(
    height=360, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=14),
)
st.plotly_chart(sankey, use_container_width=True)
 
# ---------- Distribution charts ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Share by Level")
    pie = go.Figure(go.Pie(
        labels=LEVELS, values=counts.values, hole=0.55,
        marker=dict(colors=[LEVEL_COLORS[l] for l in LEVELS]),
        textinfo="label+percent", textfont=dict(size=14, color="#041a26"),
    ))
    pie.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#ffffff"), margin=dict(t=10, b=10),
                      legend=dict(font=dict(color="#ffffff")))
    st.plotly_chart(pie, use_container_width=True)
 
with c2:
    st.subheader("Count by Level")
    bar = go.Figure(go.Bar(
        x=LEVELS, y=counts.values,
        marker_color=[LEVEL_COLORS[l] for l in LEVELS],
        text=[f"{p}%" for p in pcts.values], textposition="outside",
        textfont=dict(color="#ffffff", size=14),
    ))
    bar.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"),
                      margin=dict(t=30, b=10),
                      yaxis=dict(gridcolor="rgba(255,255,255,.12)"))
    st.plotly_chart(bar, use_container_width=True)
 
# ---------- Map ----------
st.subheader("Hotspot Map")
if LAT and LON:
    mdf = fdf.dropna(subset=[LAT, LON])
    if len(mdf) > max_pts:
        mdf = mdf.sample(max_pts, random_state=42)
    if mdf.empty:
        st.info("No points to show for the selected levels.")
    else:
        m = folium.Map(location=[float(mdf[LAT].mean()), float(mdf[LON].mean())],
                       zoom_start=4, tiles="CartoDB dark_matter")
        for _, r in mdf.iterrows():
            lvl = r["Risk Level"]
            tip = f"{lvl}" + (f" | {SCORE}: {r[SCORE]:.2f}"
                              if SCORE and pd.notna(r.get(SCORE)) else "")
            folium.CircleMarker(
                location=[r[LAT], r[LON]],
                radius=9 if lvl == "Critical" else 7 if lvl == "High" else 5,
                color=LEVEL_COLORS[lvl], fill=True, fill_color=LEVEL_COLORS[lvl],
                fill_opacity=0.85, weight=1, tooltip=tip,
            ).add_to(m)
        legend = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:#082b40;
        color:#fff;padding:10px 14px;border:1px solid #21a6b4;border-radius:10px;font-size:13px;">
        <b>Risk Level</b><br>""" + "".join(
            f'<span style="color:{LEVEL_COLORS[l]};">●</span> {l}<br>' for l in LEVELS
        ) + "</div>"
        m.get_root().html.add_child(folium.Element(legend))
        st_folium(m, height=520, use_container_width=True, returned_objects=[])
else:
    st.info("No latitude/longitude columns found, so the map is hidden.")
 
# ---------- Table ----------
st.subheader("Data")
def color_level(v):
    c = LEVEL_COLORS.get(v)
    return f"background-color:{c};color:#041a26;font-weight:700;" if c else ""
 
show = fdf.head(1000)
try:
    st.dataframe(show.style.map(color_level, subset=["Risk Level"]),
                 use_container_width=True, height=360)
except Exception:
    st.dataframe(show, use_container_width=True, height=360)
 
st.download_button("Download filtered CSV", fdf.to_csv(index=False).encode("utf-8"),
                   "filtered_marine_pollution.csv", "text/csv")
 
