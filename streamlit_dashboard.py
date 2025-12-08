# streamlit_dashboard.py
# Robust Streamlit dashboard for CSE303 — Workload vs Mental Health
# Usage: streamlit run streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re

# optional analytics/statistics libs
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Workload & Mental Health Dashboard")
st.title("CSE303 — Workload vs Mental Health (Interactive Dashboard)")
st.markdown("Upload your cleaned CSV or load the default CSV (from repo or Google Drive). The app is defensive and will show messages if data is missing.")

# ------------------ Configuration ------------------
# If you prefer the Drive fallback, make sure the file is shared publicly
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1KLnVKougQauelIE9_83vr9uiZreArfoJ/view?usp=drive_link"
DEFAULT_LOCAL = Path("./data/cleaned_survey_data.csv")  # repo-local copy is best for Streamlit Cloud

# helper to extract file id
def extract_drive_id(url: str):
    m = re.search(r"/d/([^/]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)
    return None

DRIVE_FILE_ID = extract_drive_id(GOOGLE_DRIVE_LINK)

# ------------------ Loading functions ------------------
@st.cache_data
def load_csv_from_file_like(file_like):
    return pd.read_csv(file_like)

@st.cache_data
def load_csv_from_path(path: Path):
    return pd.read_csv(path)

@st.cache_data
def download_from_drive_and_load(path: Path, file_id: str):
    """Download via gdown if available; otherwise raise informative error."""
    try:
        import gdown
    except Exception:
        raise RuntimeError("gdown is not installed. Add 'gdown' to requirements.txt or upload the CSV manually.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(path), quiet=True)
    if not path.exists():
        raise FileNotFoundError("Download failed — check Drive sharing settings (must be 'Anyone with link').")
    return pd.read_csv(path)

# ------------------ Acquire dataframe (uploader -> local -> drive) ------------------
uploaded = st.file_uploader("Upload cleaned CSV (recommended)", type=["csv"])
df = None
load_method_msg = ""

if uploaded is not None:
    try:
        df = load_csv_from_file_like(uploaded)
        load_method_msg = "Loaded from uploaded file."
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif DEFAULT_LOCAL.exists():
    try:
        df = load_csv_from_path(DEFAULT_LOCAL)
        load_method_msg = f"Loaded local default: {DEFAULT_LOCAL}"
    except Exception as e:
        st.error(f"Failed to read local CSV {DEFAULT_LOCAL}: {e}")
        st.stop()
elif DRIVE_FILE_ID:
    # try downloading from Drive (public link required)
    try:
        df = download_from_drive_and_load(DEFAULT_LOCAL, DRIVE_FILE_ID)
        load_method_msg = "Downloaded from Google Drive (cached locally)."
    except Exception as e:
        st.warning(f"Could not download from Drive: {e}")
        st.info("Please either upload the CSV or add a local file at ./data/cleaned_survey_data.csv")
        st.stop()
else:
    st.info("No CSV provided. Upload a CSV or place one at ./data/cleaned_survey_data.csv")
    st.stop()

# Show load method
st.sidebar.success(load_method_msg)
st.sidebar.write(f"Columns found: {list(df.columns)[:20]}")

# Basic sanity
if df is None or df.empty:
    st.error("Dataframe is empty after loading. Please check the CSV.")
    st.stop()

# ------------------ Column mapping (defensive) ------------------
def smart_map(df):
    cmap = {}
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    def find_contains(words):
        for i, lc in enumerate(lower):
            for w in words:
                if w in lc:
                    return cols[i]
        return None

    mapping = {
        "study_hours_per_week": ["study", "hours", "per week", "study_hours"],
        "courses_enrolled": ["course", "courses", "credit"],
        "extracurricular_bin": ["extracurricular", "extra"],
        "extra_hours_per_week": ["extra_hours", "extra hours"],
        "job_bin": ["job", "part-time", "part time"],
        "job_hours_per_week": ["job_hours", "job hours"],
        "year_of_study": ["year of study", "year"],
        "gender": ["gender"],
        "stress_1to5": ["stress"],
        "anxiety_1to5": ["anxiety"],
        "sleep_hours": ["sleep hours", "sleep_hours", "hours of sleep"],
        "sleep_quality_1to10": ["sleep quality", "sleep_quality"]
    }

    for canonical, keywords in mapping.items():
        found = find_contains(keywords)
        if found:
            cmap[found] = canonical

    return cmap

rename_map = smart_map(df)
if rename_map:
    df = df.rename(columns=rename_map)

# Force certain dtypes
num_cols_expected = ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week',
                     'stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10','year_of_study']
for c in num_cols_expected:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create binary columns if not present but derivable
if 'extracurricular_bin' not in df.columns:
    if 'extra_hours_per_week' in df.columns:
        df['extracurricular_bin'] = (df['extra_hours_per_week'].fillna(0) > 0).astype(int)
if 'job_bin' not in df.columns:
    if 'job_hours_per_week' in df.columns:
        df['job_bin'] = (df['job_hours_per_week'].fillna(0) > 0).astype(int)

# Derived
if 'stress_1to5' in df.columns:
    df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)

# ------------------ Sidebar controls & small-sample warning ------------------
n_rows = len(df)
st.sidebar.markdown(f"**Total responses:** {n_rows}")
if n_rows < 80:
    st.sidebar.warning("Sample size < 80. Interpret results with caution.")

st.sidebar.header("Filters")
filters = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique().tolist())
    filters['year_of_study'] = st.sidebar.multiselect("Year of study", options=years, default=years)
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique().tolist())
    filters['gender'] = st.sidebar.multiselect("Gender", options=genders, default=genders)
if 'job_bin' in df.columns:
    filters['job_bin'] = st.sidebar.multiselect("Job status", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No", default=[0,1])

# Apply filters
df_f = df.copy()
for k, v in filters.items():
    if v is not None and len(v) > 0:
        if k in df_f.columns:
            df_f = df_f[df_f[k].isin(v)]

# If filters produce empty data, warn and stop
if df_f.empty:
    st.warning("No rows match the selected filters. Please widen filters or reset.")
    st.stop()

# ------------------ Numeric columns list (after filtering) ------------------
num_cols = [c for c in ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week',
                        'stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]

# Layout
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Interactive Visualizations")

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric variables for a correlation matrix (need >=2).")

    # Scatter with trendline (defensive)
    st.markdown("**Scatter with regression line**")
    if len(num_cols) >= 1:
        x = st.selectbox("X variable", options=num_cols, index=0, key="scatter_x")
        y_options = [c for c in num_cols if c != x]
        if not y_options:
            st.info("Only one numeric variable available — scatter plot needs two different numeric variables.")
        else:
            y = st.selectbox("Y variable", options=y_options, index=0, key="scatter_y")
            color_opt = 'year_of_study' if 'year_of_study' in df_f.columns else None
            try:
                fig_scatter = px.scatter(df_f, x=x, y=y, color=color_opt, trendline="ols", title=f"{y} vs {x}")
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Could not produce scatter: {e}")
    else:
        st.info("No numeric columns available to plot scatter.")

    # Boxplot by group
    st.markdown("**Boxplot by group**")
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular_bin'] if c in df_f.columns]
    if group_options and num_cols:
        group = st.selectbox("Group by", options=group_options, index=0, key="box_group")
        num_for_box = st.selectbox("Numeric for boxplot", options=num_cols, index=0, key="box_numeric")
        df_plot = df_f.copy()
        if group in ['job_bin','extracurricular_bin']:
            df_plot[group] = df_plot[group].map({1:"Yes",0:"No"})
        try:
            fig_box = px.box(df_plot, x=group, y=num_for_box, points="all", title=f"{num_for_box} by {group}")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"Could not produce boxplot: {e}")
    else:
        st.info("Insufficient columns for boxplot (need group + numeric).")

with col2:
    st.subheader("Regression modeling")
    target_candidates = [c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns]
    if not target_candidates:
        st.info("No supported target variables found in dataset (stress/anxiety/sleep).")
    else:
        target = st.selectbox("Target variable", options=target_candidates, index=0)
        default_predictors = [c for c in num_cols if c != target]
        extra_cats = []
        if 'year_of_study' in df.columns: extra_cats.append('C(year_of_study)')
        if 'gender' in df.columns: extra_cats.append('C(gender)')
        predictors = st.multiselect("Predictors", options=default_predictors + extra_cats, default=default_predictors[:3])
        if st.button("Fit model"):
            if not predictors:
                st.error("Choose at least one predictor.")
            else:
                formula = f"{target} ~ " + " + ".join(predictors)
                try:
                    model = smf.ols(formula=formula, data=df_f).fit()
                    st.write("**Formula:**", formula)
                    st.write("**R-squared:**", round(model.rsquared, 3))
                    st.dataframe(model.params.to_frame("coef"))

                    # residuals vs fitted
                    resid = model.resid
                    fitted = model.fittedvalues
                    fig_res = px.scatter(x=fitted, y=resid, labels={"x":"Fitted", "y":"Residuals"}, title="Residuals vs Fitted")
                    fig_res.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig_res, use_container_width=True)

                    # QQ plot using statsmodels (matplotlib)
                    qq = sm.qqplot(resid, line="45", fit=True)
                    st.pyplot(qq)

                    st.subheader("Model summary (statsmodels)")
                    st.text(model.summary().as_text())
                except Exception as e:
                    st.error(f"Model fitting failed: {e}")

# Footer
st.markdown("---")
st.subheader("Short automated interpretation (guidance only)")
if 'stress_1to5' in df.columns:
    st.write("Example: if study_hours_per_week is a positive significant predictor of stress_1to5, this suggests association (not causation).")
st.caption("Ensure dataset is anonymized and ethically collected.")
