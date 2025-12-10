# streamlit_dashboard_final.py
"""
CSE303: Workload vs Mental Health Interactive Dashboard
Features:
- Upload CSV, use local default, or download from Google Drive
- Auto-clean and transform data
- Interactive visualizations: correlation, scatter w/ trendline, boxplots
- Filters with counts, sample warnings
- Regression modeling & diagnostics (OLS, robust SEs, VIF, residuals)
- Export filtered data & model summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------- Streamlit Page ----------------
st.set_page_config(layout="wide", page_title="Workload & Mental Health Dashboard")
st.title("CSE303 — Group 3 project dashboard")


# ---------------- Config ----------------
DEFAULT_LOCAL = Path("./data/cleaned_survey_data.csv")
DEFAULT_LOCAL.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1KLnVKougQauelIE9_83vr9uiZreArfoJ/view?usp=drive_link"

def extract_drive_id(url: str):
    if not isinstance(url, str):
        return None
    m = re.search(r"/d/([^/]+)", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([^&]+)", url)
    if m: return m.group(1)
    return None

@st.cache_data
def read_csv_from_path(path_str: str):
    return pd.read_csv(path_str)

@st.cache_data
def download_drive_and_load(path_str: str, file_id: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown not installed. Add to requirements or upload manually.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, path_str, quiet=True)
    if not Path(path_str).exists():
        raise FileNotFoundError("Failed to download from Google Drive.")
    return pd.read_csv(path_str)

# ---------------- Data Cleaning ----------------
def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if 'year' in low and 'study' in low:
            rename_map[col] = 'year_of_study'
        elif low.startswith('gender'):
            rename_map[col] = 'gender'
        elif low == 'age' or ('age' in low and 'your' in low):
            rename_map[col] = 'age'
        elif 'cgpa' in low:
            rename_map[col] = 'cgpa'
        elif 'study' in low and ('week' in low or 'per' in low):
            rename_map[col] = 'study_hours_per_week'
        elif 'course' in low and any(k in low for k in ('enroll','credit','course','courses')):
            rename_map[col] = 'courses_enrolled'
        elif 'extracurricular' in low or ('extra' in low and 'activity' in low):
            if 'hour' in low or 'per' in low:
                rename_map[col] = 'extra_hours_per_week'
            else:
                rename_map[col] = 'extracurricular'
        elif 'job' in low and 'hour' in low:
            rename_map[col] = 'job_hours_per_week'
        elif 'job' in low:
            rename_map[col] = 'job'
        elif 'stress' in low:
            rename_map[col] = 'stress_1to5'
        elif 'anxiety' in low:
            rename_map[col] = 'anxiety_1to5'
        elif 'sleep' in low and 'quality' in low:
            rename_map[col] = 'sleep_quality_1to10'
        elif 'sleep' in low and 'hour' in low:
            rename_map[col] = 'sleep_hours'
    if rename_map:
        st.sidebar.write("Auto-mapped columns:", rename_map)
    df = df.rename(columns=rename_map)

    # Numeric coercion
    numeric_cols = ['age','cgpa','study_hours_per_week','courses_enrolled','extra_hours_per_week',
                    'job_hours_per_week','sleep_hours','sleep_quality_1to10','stress_1to5','anxiety_1to5']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Normalize yes/no columns
    def norm_yesno(s):
        s = s.astype(str).str.strip().str.lower().replace({'nan': None})
        s = s.replace({'yes.':'yes','yes':'yes','y':'yes','true':'yes','1':'yes',
                       'no.':'no','no':'no','n':'no','false':'no','0':'no'})
        return s

    if 'extracurricular' in df.columns:
        df['extracurricular'] = norm_yesno(df['extracurricular'])
        df['extracurricular_bin'] = (df['extracurricular']=='yes').astype(int)
    elif 'extra_hours_per_week' in df.columns:
        df['extracurricular_bin'] = (df['extra_hours_per_week'].fillna(0)>0).astype(int)

    if 'job' in df.columns:
        df['job'] = norm_yesno(df['job'])
        df['job_bin'] = (df['job']=='yes').astype(int)
    elif 'job_hours_per_week' in df.columns:
        df['job_bin'] = (df['job_hours_per_week'].fillna(0)>0).astype(int)

    if 'year_of_study' in df.columns:
        df['year_of_study'] = pd.to_numeric(df['year_of_study'], errors='coerce').astype('Int64')
    if 'stress_1to5' in df.columns:
        df['high_stress'] = (df['stress_1to5']>=4).astype(int)
    
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.loc[:, df.notna().any()]
    df = df.reset_index(drop=True)
    return df

# ---------------- Load Data ----------------
uploaded = st.file_uploader("Upload CSV file if there is s a issue in drive", type=["csv"])
df = None
load_msg = ""

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        load_msg = "Loaded from upload."
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif DEFAULT_LOCAL.exists():
    try:
        df = read_csv_from_path(str(DEFAULT_LOCAL))
        load_msg = f"Loaded local file: {DEFAULT_LOCAL}"
    except Exception as e:
        st.warning(f"Could not read local file: {e}")
        df = None
else:
    drive_id = extract_drive_id(GOOGLE_DRIVE_LINK)
    if drive_id:
        try:
            df = download_drive_and_load(str(DEFAULT_LOCAL), drive_id)
            load_msg = "Downloaded from Google Drive (cached locally)."
        except Exception as e:
            st.warning(f"Drive download failed: {e}")

if df is None:
    st.info("No CSV loaded. Upload a cleaned CSV.")
    st.stop()

st.sidebar.success(load_msg)
df = clean_transform(df)

# ---------------- Filters ----------------
st.sidebar.header("Filters")
n_total = len(df)
st.sidebar.write("Total rows:", n_total)
if n_total < 80:
    st.sidebar.warning("Sample size < 80 — interpret subgroup inferences with caution.")

filter_values = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect("Year of study", options=years, default=years)
    filter_values['year_of_study'] = selected_years
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
    filter_values['gender'] = selected_genders
if 'job_bin' in df.columns:
    opts = ["Yes","No"]
    counts = { "Yes": int((df['job_bin']==1).sum()), "No": int((df['job_bin']==0).sum()) }
    selected_jobs = st.sidebar.multiselect("Job status", options=opts, default=opts)
    filter_values['job_bin'] = selected_jobs
if 'extracurricular_bin' in df.columns:
    opts2 = ["Yes","No"]
    counts2 = { "Yes": int((df['extracurricular_bin']==1).sum()), "No": int((df['extracurricular_bin']==0).sum()) }
    selected_ex = st.sidebar.multiselect("Extracurricular", options=opts2, default=opts2)
    filter_values['extracurricular_bin'] = selected_ex

# Apply filters
df_f = df.copy()
if 'year_of_study' in filter_values:
    df_f = df_f[df_f['year_of_study'].isin(filter_values['year_of_study'])]
if 'gender' in filter_values:
    df_f = df_f[df_f['gender'].isin(filter_values['gender'])]
if 'job_bin' in filter_values:
    df_f = df_f[df_f['job_bin'].isin([1 if s=='Yes' else 0 for s in filter_values['job_bin']])]
if 'extracurricular_bin' in filter_values:
    df_f = df_f[df_f['extracurricular_bin'].isin([1 if s=='Yes' else 0 for s in filter_values['extracurricular_bin']])]

if df_f.empty:
    st.warning("No rows match the selected filters. Widen filters or upload another CSV.")
    st.stop()

# ---------------- Visualizations ----------------
num_cols = [c for c in ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week',
                        'stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]

col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Visualizations")
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix"), use_container_width=True)
    
    st.markdown("**Scatter with regression line**")
    if len(num_cols) >= 2:
        x = st.selectbox("X variable", options=num_cols, index=0)
        y_options = [c for c in num_cols if c != x]
        y = st.selectbox("Y variable", options=y_options, index=0)
        color_opt = 'year_of_study' if 'year_of_study' in df_f.columns else None
        fig_sc = px.scatter(df_f, x=x, y=y, color=color_opt, trendline="ols", title=f"{y} vs {x}")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("**Boxplot by group**")
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular_bin'] if c in df_f.columns]
    if group_options and num_cols:
        group = st.selectbox("Group by", options=group_options, index=0)
        num_for_box = st.selectbox("Numeric", options=num_cols, index=0)
        df_plot = df_f.copy()
        if group in ['job_bin','extracurricular_bin']:
            df_plot[group] = df_plot[group].map({1:"Yes",0:"No"})
        fig_box = px.box(df_plot, x=group, y=num_for_box, points="all", title=f"{num_for_box} by {group}")
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------- Regression & Diagnostics ----------------
with col2:
    st.subheader("Regression Modeling")
    target_candidates = [c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns]
    if target_candidates:
        target = st.selectbox("Target variable", options=target_candidates, index=0)
        default_preds = [c for c in num_cols if c != target]
        extra_cats = [f"C({c})" for c in ['year_of_study','gender'] if c in df.columns]
        predictors = st.multiselect("Predictors", options=default_preds + extra_cats, default=default_preds[:3])

        if st.button("Fit model"):
            if predictors:
                formula = f"{target} ~ " + " + ".join(predictors)
                model_full = smf.ols(formula=formula, data=df_f).fit()
                st.write("**Formula:**", formula)
                st.write("**R²:**", round(model_full.rsquared,3))
                st.dataframe(model_full.params.to_frame("coef"))
                st.text_area("Model summary", model_full.summary().as_text(), height=300)

# ---------------- Export Filtered Data ----------------
st.markdown("---")
st.subheader("Export / Save")
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered dataset as CSV", data=csv_bytes, file_name="filtered_survey.csv", mime="text/csv")
