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
st.markdown("Upload your cleaned CSV, or let the app load the default (local or Google Drive). If using Drive, file must be shared 'Anyone with link'.")

# ---------------- CONFIG ----------------
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1KLnVKougQauelIE9_83vr9uiZreArfoJ/view?usp=drive_link"
DEFAULT_LOCAL = Path("./data/cleaned_survey_data.csv")  # place a copy here for Streamlit Cloud stability
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- Utilities ----------------
def extract_drive_id(url: str):
    if not isinstance(url, str):
        return None
    m = re.search(r"/d/([^/]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)
    return None

DRIVE_FILE_ID = extract_drive_id(GOOGLE_DRIVE_LINK)

@st.cache_data
def load_csv_from_filelike(filelike):
    return pd.read_csv(filelike)

@st.cache_data
def load_csv_from_path(path: Path):
    return pd.read_csv(path)

@st.cache_data
def download_from_drive(path: Path, file_id: str):
    try:
        import gdown
    except Exception:
        raise RuntimeError("gdown not installed in the environment. Add 'gdown' to requirements.txt or upload the CSV manually.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(path), quiet=True)
    if not path.exists():
        raise FileNotFoundError("Failed to download file from Drive. Ensure file is shared 'Anyone with the link'.")
    return pd.read_csv(path)

# Cleaning function (adapted from your Colab notebook)
def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if 'year' in low and 'study' in low:
            rename_map[col] = 'year_of_study'
        elif low.strip().startswith('gender'):
            rename_map[col] = 'gender'
        elif low.strip() == 'age' or ('age' in low and 'your' in low):
            rename_map[col] = 'age'
        elif 'cgpa' in low:
            rename_map[col] = 'cgpa'
        elif 'study' in low and ('week' in low or 'per' in low):
            rename_map[col] = 'study_hours_per_week'
        elif ('course' in low and ('enroll' in low or 'credit' in low or 'current' in low)):
            rename_map[col] = 'courses_enrolled'
        elif 'extracurricular' in low or ('extra' in low and 'activity' in low):
            if 'hours' in low or 'per' in low:
                rename_map[col] = 'extra_hours_per_week'
            else:
                rename_map[col] = 'extracurricular'
        elif 'job' in low and ('part' in low or 'have' in low or 'tuition' in low):
            # general job yes/no
            rename_map[col] = 'job'
        elif 'job' in low and 'hours' in low:
            rename_map[col] = 'job_hours_per_week'
        elif 'stress' in low and 'level' in low or low.strip().startswith('stress'):
            rename_map[col] = 'stress_1to5'
        elif 'anxiety' in low:
            rename_map[col] = 'anxiety_1to5'
        elif 'how many hours of sleep' in low or ('sleep' in low and 'hours' in low):
            rename_map[col] = 'sleep_hours'
        elif 'sleep' in low and 'quality' in low:
            rename_map[col] = 'sleep_quality_1to10'
        elif 'timestamp' in low or 'time' in low:
            rename_map[col] = 'timestamp'
    df = df.rename(columns=rename_map)

    # Coerce numeric columns
    numeric_cols = ['age','cgpa','study_hours_per_week','courses_enrolled','extra_hours_per_week',
                    'job_hours_per_week','sleep_hours','sleep_quality_1to10','stress_1to5','anxiety_1to5']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Standardize Yes/No fields & binary columns
    if 'extracurricular' in df.columns:
        df['extracurricular'] = df['extracurricular'].astype(str).str.strip().str.title()
        df.loc[df['extracurricular'].isin(['Yes','Y','True','1','Yes.']), 'extracurricular'] = 'Yes'
        df.loc[df['extracurricular'].isin(['No','N','False','0','No.']), 'extracurricular'] = 'No'
        df['extracurricular_bin'] = (df['extracurricular']=='Yes').astype(int)
    else:
        # if there are hours but no Yes/No column, create from hours
        if 'extra_hours_per_week' in df.columns:
            df['extracurricular_bin'] = (df['extra_hours_per_week'].fillna(0) > 0).astype(int)

    if 'job' in df.columns:
        df['job'] = df['job'].astype(str).str.strip().str.title()
        df.loc[df['job'].isin(['Yes','Y','True','1','Yes.']), 'job'] = 'Yes'
        df.loc[df['job'].isin(['No','N','False','0','No.']), 'job'] = 'No'
        # create job_bin integer
        df['job_bin'] = (df['job']=='Yes').astype(int)
    else:
        if 'job_hours_per_week' in df.columns:
            df['job_bin'] = (df['job_hours_per_week'].fillna(0) > 0).astype(int)

    # Year fix: numeric ints where possible
    if 'year_of_study' in df.columns:
        df['year_of_study'] = pd.to_numeric(df['year_of_study'], errors='coerce').astype('Int64')

    # Fill extra hours / job hours with 0 for No responses if those string columns exist
    if 'extra_hours_per_week' in df.columns and 'extracurricular' in df.columns:
        df.loc[df['extracurricular']=='No','extra_hours_per_week'] = df.loc[df['extracurricular']=='No','extra_hours_per_week'].fillna(0)
    if 'job_hours_per_week' in df.columns and 'job' in df.columns:
        df.loc[df['job']=='No','job_hours_per_week'] = df.loc[df['job']=='No','job_hours_per_week'].fillna(0)

    # High stress indicator
    if 'stress_1to5' in df.columns:
        df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)

    # Trim whitespace from string columns
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()

    # Drop columns that are fully NA
    df = df.loc[:, df.notna().any()]

    df = df.reset_index(drop=True)
    return df

# ---------------- Acquire dataframe ----------------
uploaded = st.file_uploader("Upload cleaned CSV (recommended)", type=["csv"])
df = None
load_message = ""

if uploaded is not None:
    try:
        df = load_csv_from_filelike(uploaded)
        load_message = "Loaded from upload."
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif DEFAULT_LOCAL.exists():
    try:
        df = load_csv_from_path(DEFAULT_LOCAL)
        load_message = f"Loaded local file: {DEFAULT_LOCAL}"
    except Exception as e:
        st.error(f"Failed to read local CSV: {e}")
        st.stop()
elif DRIVE_FILE_ID:
    # attempt drive download
    try:
        df = download_from_drive(DEFAULT_LOCAL, DRIVE_FILE_ID)
        load_message = "Downloaded from Google Drive (cached locally)."
    except Exception as e:
        st.warning(f"Drive download failed: {e}")
        st.info("Please upload the CSV or add it to ./data/cleaned_survey_data.csv")
        st.stop()
else:
    st.info("No CSV found. Upload one or place a CSV at ./data/cleaned_survey_data.csv")
    st.stop()

st.sidebar.success(load_message)
st.sidebar.write("Detected columns:", list(df.columns)[:40])

if df is None or df.empty:
    st.error("Loaded dataframe is empty. Please check the CSV.")
    st.stop()

# ---------------- Clean & standardize ----------------
df = clean_transform(df)

# show a short preview option
st.sidebar.header("Preview")
if st.sidebar.checkbox("Show data sample", value=False):
    st.dataframe(df.head(10))

# ---------------- Prepare filters (defensive) ----------------
# Normalize categorical columns for filtering
if 'gender' in df.columns:
    df['gender'] = df['gender'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan}).where(lambda s: s.notna(), other=np.nan)
    df.loc[df['gender'].notna(), 'gender'] = df.loc[df['gender'].notna(), 'gender'].str.title()

if 'job' in df.columns and df['job'].dtype == object:
    df['job'] = df['job'].astype(str).str.strip().str.title()
if 'extracurricular' in df.columns and df['extracurricular'].dtype == object:
    df['extracurricular'] = df['extracurricular'].astype(str).str.strip().str.title()

# Guarantee job_bin/extracurricular_bin are present (0/1)
if 'job_bin' not in df.columns:
    if 'job' in df.columns:
        df['job_bin'] = (df['job'] == 'Yes').astype(int)
    elif 'job_hours_per_week' in df.columns:
        df['job_bin'] = (df['job_hours_per_week'].fillna(0) > 0).astype(int)

if 'extracurricular_bin' not in df.columns:
    if 'extracurricular' in df.columns:
        df['extracurricular_bin'] = (df['extracurricular'] == 'Yes').astype(int)
    elif 'extra_hours_per_week' in df.columns:
        df['extracurricular_bin'] = (df['extra_hours_per_week'].fillna(0) > 0).astype(int)

# Sidebar controls & defaults
st.sidebar.header("Filters")
filter_defaults = {}

if 'year_of_study' in df.columns:
    years = sorted([int(x) for x in df['year_of_study'].dropna().unique().tolist()])
    # include a Missing option if any NA
    year_opts = [y for y in years]
    if df['year_of_study'].isna().any():
        year_opts = year_opts + ["Missing"]
    filter_defaults['year_of_study'] = st.sidebar.multiselect("Year of study", options=year_opts, default=year_opts)

if 'gender' in df.columns:
    genders = sorted([g for g in df['gender'].dropna().unique().tolist()])
    if df['gender'].isna().any():
        genders = genders + ["Missing"]
    filter_defaults['gender'] = st.sidebar.multiselect("Gender", options=genders, default=genders)

if 'job_bin' in df.columns:
    job_opts = ["Yes","No"]
    # if job is numeric bin, map default display values
    # check presence of NaN in job column (rare)
    if df['job_bin'].isna().any():
        job_opts = job_opts + ["Missing"]
    filter_defaults['job_bin'] = st.sidebar.multiselect("Job status", options=job_opts, default=job_opts)

# Reset filters button: when clicked, rerun to restore defaults
if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

# ---------------- Apply filters (robust) ----------------
df_f = df.copy()

# helper to interpret selections (handles "Missing" token)
def apply_filter(df_current, col, selection):
    if selection is None or len(selection) == 0:
        return df_current  # no filtering for this column
    # treat "Missing" specially
    sel = list(selection)
    include_missing = False
    if "Missing" in sel:
        include_missing = True
        sel = [s for s in sel if s != "Missing"]
    if col == 'year_of_study':
        # convert strings like '1' back to numeric if present
        numeric_sel = []
        for s in sel:
            try:
                numeric_sel.append(int(s))
            except Exception:
                # ignore
                pass
        cond = pd.Series(False, index=df_current.index)
        if numeric_sel:
            cond = cond | df_current[col].isin(numeric_sel)
        if include_missing:
            cond = cond | df_current[col].isna()
        return df_current.loc[cond]
    elif col in ['job_bin','extracurricular_bin']:
        # user choices are 'Yes'/'No' strings
        ints = []
        for s in sel:
            if isinstance(s, str):
                if s.lower().startswith('y') or s == "Yes":
                    ints.append(1)
                elif s.lower().startswith('n') or s == "No":
                    ints.append(0)
        cond = df_current[col].isin(ints) if ints else pd.Series(False, index=df_current.index)
        if include_missing:
            cond = cond | df_current[col].isna()
        return df_current.loc[cond]
    else:
        # generic string filter
        cond = pd.Series(False, index=df_current.index)
        if sel:
            cond = cond | df_current[col].isin(sel)
        if include_missing:
            cond = cond | df_current[col].isna()
        return df_current.loc[cond]

# actually apply
for col, sel in filter_defaults.items():
    if col in df_f.columns:
        df_f = apply_filter(df_f, col, sel)

# If filters produce empty data, show diagnostic and stop (but not crash)
if df_f.empty:
    st.warning("No rows match the selected filters. Please widen filters or press 'Reset filters'.")
    st.write("Current filter selections:")
    for k, v in filter_defaults.items():
        st.write(f"- {k}: {v}")
    st.write("Unique values in the dataset for each filter column:")
    for k in filter_defaults.keys():
        if k in df.columns:
            st.write(f"**{k}**:", sorted(df[k].dropna().unique().tolist())[:50], ("+ Missing" if df[k].isna().any() else ""))
    st.stop()

# ---------------- Numeric columns available for plotting ----------------
num_cols = [c for c in ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week',
                        'stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]

# Layout
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Interactive Visualizations")

    # correlation matrix
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric variables for correlation matrix (need >= 2).")

    # Scatter with trendline
    st.markdown("**Scatter with regression line**")
    if len(num_cols) >= 2:
        x = st.selectbox("X variable", options=num_cols, index=0, key="scatter_x")
        y_options = [c for c in num_cols if c != x]
        y = st.selectbox("Y variable", options=y_options, index=0, key="scatter_y")
        color_opt = 'year_of_study' if 'year_of_study' in df_f.columns else None
        try:
            fig_scatter = px.scatter(df_f, x=x, y=y, color=color_opt, trendline="ols", title=f"{y} vs {x}")
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.error(f"Could not produce scatter: {e}")
    elif len(num_cols) == 1:
        st.info("Only one numeric variable available — scatter plot needs two numeric variables.")
    else:
        st.info("No numeric columns available for scatter.")

    # Boxplot by group
    st.markdown("**Boxplot by group**")
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular_bin'] if c in df_f.columns]
    if group_options and len(num_cols) >= 1:
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
        st.info("No target variable (stress/anxiety/sleep) is present in dataset.")
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
                    fig_res = px.scatter(x=fitted, y=resid, labels={"x":"Fitted","y":"Residuals"}, title="Residuals vs Fitted")
                    fig_res.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig_res, use_container_width=True)

                    # QQ plot using statsmodels
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
