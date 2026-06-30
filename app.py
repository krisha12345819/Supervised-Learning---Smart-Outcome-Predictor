"""
Smart Outcome Predictor — Streamlit Web App
Author: Krisha | Ensemble Learning Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
  --bg-deep:    #06091a;
  --bg-card:    #0e1230;
  --bg-card2:   #111a3a;
  --violet:     #7c3aed;
  --violet-lt:  #a78bfa;
  --cyan:       #22d3ee;
  --cyan-lt:    #67e8f9;
  --pink:       #ec4899;
  --green:      #10b981;
  --amber:      #f59e0b;
  --text-hi:    #f1f5f9;
  --text-mid:   #94a3b8;
  --text-lo:    #475569;
  --border:     rgba(124,58,237,0.25);
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-deep);
  color: var(--text-hi);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--violet); border-radius: 3px; }

/* ── Streamlit chrome ── */
.stApp { background: var(--bg-deep); }

header[data-testid="stHeader"] {
  background: rgba(6,9,26,0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

section[data-testid="stSidebar"] {
  background: var(--bg-card);
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Sidebar label ── */
.sidebar-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--violet-lt);
  letter-spacing: 0.04em;
  padding: 0.5rem 0 1.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #0e1230 0%, #1a0a3d 50%, #0a1a2e 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2.5rem 2.8rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
  border-radius: 50%;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 30%;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(34,211,238,0.1) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-eyebrow {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cyan);
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--text-hi);
  line-height: 1.15;
  margin-bottom: 0.8rem;
}
.hero-title span { color: var(--violet-lt); }
.hero-sub {
  color: var(--text-mid);
  font-size: 0.98rem;
  max-width: 680px;
  line-height: 1.65;
}
.hero-badges {
  display: flex;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin-top: 1.2rem;
}
.badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.28rem 0.75rem;
  border-radius: 99px;
  letter-spacing: 0.05em;
  border: 1px solid;
}
.badge-violet { color: var(--violet-lt); border-color: var(--violet); background: rgba(124,58,237,0.12); }
.badge-cyan   { color: var(--cyan-lt);   border-color: var(--cyan);   background: rgba(34,211,238,0.1);  }
.badge-green  { color: #6ee7b7;          border-color: var(--green);  background: rgba(16,185,129,0.1);  }

/* ── KPI cards ── */
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.8rem; }

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.4rem;
  transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--violet-lt); }
.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-mid);
  margin-bottom: 0.3rem;
}
.kpi-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-hi);
}
.kpi-delta { font-size: 0.78rem; margin-top: 0.2rem; color: var(--green); }

/* ── Section headings ── */
.section-head {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-hi);
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* ── Plotly chart container ── */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
}

/* ── Predict card ── */
.predict-panel {
  background: linear-gradient(135deg, #0e1230, #140c2e);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 2rem;
}

/* ── Result box ── */
.result-box {
  background: var(--bg-card2);
  border-radius: 12px;
  padding: 1.5rem 2rem;
  border-left: 4px solid var(--violet);
  margin-top: 1.2rem;
}
.result-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-mid); }
.result-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0; }
.result-pass { color: var(--green); }
.result-fail { color: var(--pink); }
.result-score { color: var(--cyan); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card);
  border-radius: 10px;
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--text-mid);
  border-radius: 7px;
  font-weight: 500;
  font-size: 0.88rem;
  padding: 0.45rem 1rem;
  border: none;
}
.stTabs [aria-selected="true"] {
  background: var(--violet) !important;
  color: white !important;
}

/* ── Sliders + inputs ── */
.stSlider [data-baseweb="slider"] { color: var(--violet-lt); }
.stSelectbox > div > div { background: var(--bg-card); border-color: var(--border) !important; color: var(--text-hi); }
.stNumberInput > div > div > input { background: var(--bg-card); border-color: var(--border) !important; color: var(--text-hi); }

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, var(--violet), #4f46e5);
  color: white;
  border: none;
  border-radius: 10px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 0.95rem;
  padding: 0.65rem 2rem;
  transition: all 0.2s;
  width: 100%;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #6d28d9, #4338ca);
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(124,58,237,0.4);
}

/* ── Metric override ── */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--text-hi); }
[data-testid="stMetricLabel"] { color: var(--text-mid); }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Radio sidebar ── */
.stRadio label { color: var(--text-mid) !important; font-size: 0.9rem; }
.stRadio [aria-checked="true"] + div { color: var(--violet-lt) !important; }

/* ── Info/success boxes ── */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING & MODEL TRAINING ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Smart_Outcome_Predictor_Dataset_5200.csv")
    df["time_spent_hours"]  = df["time_spent_hours"].fillna(df["time_spent_hours"].median())
    df["avg_quiz_score"]    = df["avg_quiz_score"].fillna(df["avg_quiz_score"].median())
    df["attendance_rate"]   = df["attendance_rate"].fillna(df["attendance_rate"].median())
    return df

@st.cache_resource
def train_models(df):
    work = df.drop("student_id", axis=1).copy()
    cat_cols = ["country_region", "device_type", "education_background",
                "course_level", "course_category", "course_start_date"]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        work[c] = le.fit_transform(work[c].astype(str))
        encoders[c] = le

    feat_cols = [c for c in work.columns if c not in ["completion_status", "final_score"]]
    X = work[feat_cols]
    y_cls = work["completion_status"]
    y_reg = work["final_score"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_sc, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_sc, y_reg, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    clf.fit(X_tr_c, y_tr_c)

    reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    reg.fit(X_tr_r, y_tr_r)

    cls_acc  = accuracy_score(y_te_c, clf.predict(X_te_c))
    reg_r2   = r2_score(y_te_r, reg.predict(X_te_r))
    reg_mae  = mean_absolute_error(y_te_r, reg.predict(X_te_r))
    cm       = confusion_matrix(y_te_c, clf.predict(X_te_c))

    fi = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    return dict(
        clf=clf, reg=reg, scaler=scaler, encoders=encoders,
        feat_cols=feat_cols, cat_cols=cat_cols, X=X,
        cls_acc=cls_acc, reg_r2=reg_r2, reg_mae=reg_mae,
        cm=cm, fi=fi
    )

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    title_font=dict(family="Space Grotesk", color="#f1f5f9", size=14),
    xaxis=dict(gridcolor="rgba(124,58,237,0.12)", zerolinecolor="rgba(124,58,237,0.12)"),
    yaxis=dict(gridcolor="rgba(124,58,237,0.12)", zerolinecolor="rgba(124,58,237,0.12)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(l=10, r=10, t=40, b=10)
)
COLORS = ["#7c3aed","#22d3ee","#ec4899","#10b981","#f59e0b","#a78bfa","#67e8f9"]

# ─── LOAD ──────────────────────────────────────────────────────────────────────
df   = load_data()
mods = train_models(df)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎓 Smart Outcome Predictor</div>', unsafe_allow_html=True)
    nav = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Explore Data", "🧠 Model Insights", "🔮 Predict"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#475569;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;">Dataset Info</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.85rem;">5 200 students · 17 features<br>Classification + Regression</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#475569;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;">Model Stack</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:0.85rem;">Gradient Boosting · LabelEncoder · StandardScaler</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Ensemble Learning · Predictive Analytics</div>
      <div class="hero-title">Smart <span>Outcome</span> Predictor</div>
      <div class="hero-sub">
        A dual-task ML system that predicts whether a student will complete their course
        and estimates their final score — powered by Gradient Boosting on 5 200 learner records.
      </div>
      <div class="hero-badges">
        <span class="badge badge-violet">GradientBoosting</span>
        <span class="badge badge-cyan">5 200 Records</span>
        <span class="badge badge-green">Dual-Task ML</span>
        <span class="badge badge-violet">17 Features</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Students", "5,200", "100% cleaned")
    with c2:
        st.metric("Completion Rate", f"{df['completion_status'].mean()*100:.1f}%", "1 952 completed")
    with c3:
        st.metric("Classifier Accuracy", f"{mods['cls_acc']*100:.1f}%", "GBM · 20% test")
    with c4:
        st.metric("Regressor R²", f"{mods['reg_r2']:.3f}", f"MAE {mods['reg_mae']:.2f}")

    st.markdown('<div class="section-head">📌 Data Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(8).style.format(precision=2).set_properties(
            subset=["final_score","attendance_rate","avg_quiz_score"],
            **{"color": "#22d3ee", "font-weight": "600"}
        ),
        use_container_width=True
    )

    # ── Quick charts ──
    st.markdown('<div class="section-head">📈 Key Distributions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="final_score", nbins=40,
                           color_discrete_sequence=["#7c3aed"],
                           title="Final Score Distribution",
                           labels={"final_score": "Final Score"})
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vc = df["completion_status"].value_counts().reset_index()
        vc.columns = ["status","count"]
        vc["label"] = vc["status"].map({0:"Not Completed",1:"Completed"})
        fig = px.pie(vc, names="label", values="count",
                     color_discrete_sequence=["#ec4899","#22d3ee"],
                     title="Completion Status Split",
                     hole=0.5)
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Explore Data":
    st.markdown('<div class="section-head">🔭 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Distributions", "🔗 Correlations", "🌍 Segments"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(df, x="course_level", y="final_score",
                         color="course_level",
                         color_discrete_sequence=COLORS,
                         title="Final Score by Course Level")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.violin(df, x="device_type", y="final_score",
                            color="device_type",
                            color_discrete_sequence=["#22d3ee","#a78bfa","#ec4899"],
                            box=True, title="Score Distribution by Device")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="time_spent_hours", color="completion_status",
                               nbins=35, barmode="overlay",
                               color_discrete_map={0:"#ec4899",1:"#22d3ee"},
                               title="Time Spent (hrs) vs Completion",
                               labels={"completion_status":"Completed?"})
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df.sample(800, random_state=1),
                             x="attendance_rate", y="final_score",
                             color="completion_status",
                             color_discrete_map={0:"#ec4899",1:"#22d3ee"},
                             opacity=0.65, size_max=6,
                             title="Attendance Rate vs Final Score",
                             labels={"completion_status":"Completed"})
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ["age","sessions","time_spent_hours","videos_watched",
                    "quiz_attempts","assignments_submitted","forum_posts",
                    "avg_quiz_score","attendance_rate","final_score"]
        corr = df[num_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#06091a"],[0.5,"#4f46e5"],[1,"#22d3ee"]],
            text=corr.round(2).values, texttemplate="%{text}",
            textfont={"size":9},
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Feature Correlation Heatmap", height=520, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            grp = df.groupby("course_category")["final_score"].mean().reset_index()
            grp.columns = ["Category","Avg Score"]
            fig = px.bar(grp.sort_values("Avg Score", ascending=False),
                         x="Category", y="Avg Score",
                         color="Avg Score",
                         color_continuous_scale=["#4f46e5","#22d3ee"],
                         title="Avg Final Score by Course Category")
            fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            grp2 = df.groupby("country_region")["completion_status"].mean().mul(100).reset_index()
            grp2.columns = ["Region","Completion %"]
            fig = px.bar(grp2.sort_values("Completion %", ascending=True),
                         x="Completion %", y="Region", orientation="h",
                         color="Completion %",
                         color_continuous_scale=["#7c3aed","#22d3ee"],
                         title="Completion Rate by Region (%)")
            fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Age group analysis
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"], bins=[15,20,25,30,35,46],
                                   labels=["16-20","21-25","26-30","31-35","36+"])
        grp3 = df2.groupby("age_group", observed=True).agg(
            Completion_Rate=("completion_status","mean"),
            Avg_Score=("final_score","mean")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=grp3["age_group"], y=grp3["Avg_Score"],
                             name="Avg Score", marker_color="#7c3aed"), secondary_y=False)
        fig.add_trace(go.Scatter(x=grp3["age_group"], y=grp3["Completion_Rate"]*100,
                                 name="Completion %", mode="lines+markers",
                                 line=dict(color="#22d3ee", width=2.5),
                                 marker=dict(size=8)), secondary_y=True)
        fig.update_layout(title="Score & Completion Rate by Age Group",
                          yaxis2=dict(title="Completion %", color="#22d3ee"), **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🧠 Model Insights":
    st.markdown('<div class="section-head">🧠 Model Performance & Interpretability</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Classifier Accuracy", f"{mods['cls_acc']*100:.2f}%")
    with c2: st.metric("Regressor R²", f"{mods['reg_r2']:.4f}")
    with c3: st.metric("Regressor MAE", f"{mods['reg_mae']:.2f} pts")

    tab1, tab2 = st.tabs(["🏆 Feature Importance", "🧩 Confusion Matrix"])

    with tab1:
        fi = mods["fi"].reset_index()
        fi.columns = ["Feature","Importance"]
        fig = px.bar(fi.head(12), x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#4f46e5","#a78bfa","#22d3ee"],
                     title="Top 12 Features — Completion Classifier")
        layout = {**PLOT_LAYOUT, "coloraxis_showscale": False}
        layout["yaxis"] = dict(autorange="reversed", gridcolor="rgba(124,58,237,0.12)",
                               zerolinecolor="rgba(124,58,237,0.12)")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cm = mods["cm"]
        fig = px.imshow(cm,
                        text_auto=True,
                        x=["Predicted: No","Predicted: Yes"],
                        y=["Actual: No","Actual: Yes"],
                        color_continuous_scale=[[0,"#06091a"],[1,"#7c3aed"]],
                        title="Confusion Matrix — Completion Classifier",
                        aspect="auto")
        fig.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.markdown('<div class="section-head">📋 Classification Report</div>', unsafe_allow_html=True)
    work = df.drop("student_id", axis=1).copy()
    for c in mods["cat_cols"]:
        work[c] = mods["encoders"][c].transform(work[c].astype(str))
    X_all = mods["scaler"].transform(work[mods["feat_cols"]])
    y_pred = mods["clf"].predict(X_all)
    report = classification_report(work["completion_status"], y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.drop(["accuracy"], errors="ignore")

    def _violet_bar(s):
        # matplotlib-free "heatmap" using CSS linear-gradient bars, scaled 0-1
        vmin, vmax = s.min(), s.max()
        rng = (vmax - vmin) or 1
        styles = []
        for v in s:
            pct = (v - vmin) / rng
            alpha = 0.12 + 0.55 * pct
            styles.append(f"background-color: rgba(124,58,237,{alpha:.2f}); color: #f1f5f9;")
        return styles

    styled = (
        report_df.style
        .format("{:.3f}")
        .apply(_violet_bar, subset=["precision", "recall", "f1-score"])
    )
    st.dataframe(styled, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Predict":
    st.markdown('<div class="section-head">🔮 Live Student Prediction</div>', unsafe_allow_html=True)
    st.markdown("Fill in the student's profile below to get instant predictions on **course completion** and **final score**.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            age                  = st.slider("Age", 16, 45, 22)
            sessions             = st.slider("Sessions", 1, 60, 12)
            time_spent           = st.slider("Time Spent (hrs)", 0.5, 150.0, 25.0, step=0.5)
            videos_watched       = st.slider("Videos Watched", 0, 100, 20)
        with c2:
            quiz_attempts        = st.slider("Quiz Attempts", 0, 20, 5)
            assignments_submitted= st.slider("Assignments Submitted", 0, 30, 8)
            forum_posts          = st.slider("Forum Posts", 0, 50, 3)
            avg_quiz_score       = st.slider("Avg Quiz Score", 0.0, 100.0, 65.0, step=0.5)
        with c3:
            attendance_rate      = st.slider("Attendance Rate", 0.0, 1.0, 0.75, step=0.01)
            week_of_year         = st.slider("Week of Year", 1, 52, 20)
            country_region       = st.selectbox("Country / Region", df["country_region"].unique())
            device_type          = st.selectbox("Device Type", df["device_type"].unique())
            education_background = st.selectbox("Education Background", df["education_background"].unique())
            course_level         = st.selectbox("Course Level", df["course_level"].unique())
            course_category      = st.selectbox("Course Category", df["course_category"].unique())

        predict_btn = st.button("⚡ Run Prediction")

        if predict_btn:
            row = {
                "age": age, "sessions": sessions, "time_spent_hours": time_spent,
                "videos_watched": videos_watched, "quiz_attempts": quiz_attempts,
                "assignments_submitted": assignments_submitted, "forum_posts": forum_posts,
                "avg_quiz_score": avg_quiz_score, "attendance_rate": attendance_rate,
                "week_of_year": week_of_year,
                "country_region": country_region, "device_type": device_type,
                "education_background": education_background, "course_level": course_level,
                "course_category": course_category, "course_start_date": "2024-01-01"
            }
            inp = pd.DataFrame([row])
            for c in mods["cat_cols"]:
                inp[c] = mods["encoders"][c].transform(inp[c].astype(str))
            inp_sc = mods["scaler"].transform(inp[mods["feat_cols"]])

            completion   = mods["clf"].predict(inp_sc)[0]
            prob         = mods["clf"].predict_proba(inp_sc)[0][1]
            score        = mods["reg"].predict(inp_sc)[0]

            status_label = "✅ Will Complete" if completion == 1 else "❌ At Risk"
            status_cls   = "result-pass" if completion == 1 else "result-fail"

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="result-box">
                  <div class="result-label">Completion Prediction</div>
                  <div class="result-value {status_cls}">{status_label}</div>
                  <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem;">Confidence: <b style="color:#a78bfa">{prob*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="result-box">
                  <div class="result-label">Predicted Final Score</div>
                  <div class="result-value result-score">{score:.1f} / 100</div>
                  <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem;">Gradient Boosting Regressor</div>
                </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": df["final_score"].mean(), "valueformat":".1f"},
                title={"text": "Predicted Score vs Dataset Avg", "font":{"color":"#f1f5f9","family":"Space Grotesk","size":13}},
                number={"font":{"color":"#22d3ee","size":52}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":"#475569"},
                    "bar": {"color":"#7c3aed"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor":"rgba(124,58,237,0.3)",
                    "steps": [
                        {"range":[0,50],  "color":"rgba(236,72,153,0.15)"},
                        {"range":[50,75], "color":"rgba(245,158,11,0.12)"},
                        {"range":[75,100],"color":"rgba(16,185,129,0.12)"}
                    ],
                    "threshold": {"line":{"color":"#22d3ee","width":2},"thickness":0.8,"value":df["final_score"].mean()}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), height=300, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
