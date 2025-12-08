# streamlit_dashboard.py
# CSE303: Workload vs Mental Health Dashboard
# Loads CSV from Google Drive directly (no subprocess), safe for Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import plotly.express as px
from pathlib import Path
import re

# ------------------ Google Drive CSV ------------------
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1KLnVKougQauelIE9_83vr9uiZreArfoJ/view?usp=drive_link"
OUTPUT_FILENAME = "cleaned_survey_data.csv"
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CSV_LOCAL_PATH = DATA_DIR / OUTPUT_FILENAME

# Function to extract file ID from Google Drive link
def extract_drive_id(url):
    match = re.search(r'/d/([^/]+)', url)
    if match: return match.group(1)
    match = re.search(r'[?&]id=([^&]+)', url)
    if match: return match.group(1)
    raise ValueError("Invalid Google Drive link")

DRIVE_FILE_ID = extract_drive_id(GOOGLE_DRIVE_LINK)

# Download CSV if not exists
if not CSV_LOCAL_PATH.exists():
    try:
        import gdown
    except ImportError:
        st.error("Please ensure gdown is installed (add 'gdown' to requirements.txt).")
        st.stop()
    
    st.info("Downloading CSV from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, str(CSV_LOCAL_PATH), quiet=False)

# Load CSV
try:
    df = pd.read_csv(CSV_LOCAL_PATH)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# ------------------ Sidebar: Preview ------------------
st.sidebar.header("Preview & sanitization")
if st.sidebar.checkbox("Show raw preview", value=False):
    st.write(df.head())

# ------------------ Column Mapping ------------------
col_map = {}
cols_lower = [c.lower() for c in df.columns]

def map_column(possible_names, canonical_name):
    for name in possible_names:
        for c in df.columns:
            if name.lower() in c.lower():
                col_map[canonical_name] = c
                return
    return

map_column(["study_hours"], "study_hours_per_week")
map_column(["stress"], "stress_1to5")
map_column(["anxiety"], "anxiety_1to5")
map_column(["sleep_hours"], "sleep_hours")
map_column(["sleep_quality"], "sleep_quality_1to10")
map_column(["extra"], "extracurricular_bin")
map_column(["job"], "job_bin")
map_column(["year"], "year_of_study")
map_column(["gender"], "gender")

rename_actual = {v: k for k, v in col_map.items() if v in df.columns}
df = df.rename(columns=rename_actual)

# Ensure numeric types
num_cols_expected = ['study_hours_per_week','stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10']
for c in num_cols_expected:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Derived variables
if 'stress_1to5' in df.columns:
    df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)

# ------------------ Sidebar: Filters ------------------
st.sidebar.header("Filters")
filters = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique())
    filters['year_of_study'] = st.sidebar.multiselect('Year of study', options=years, default=years)
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique())
    filters['gender'] = st.sidebar.multiselect('Gender', options=genders, default=genders)
if 'job_bin' in df.columns:
    filters['job_bin'] = st.sidebar.multiselect('Job status', options=[0,1], format_func=lambda x: 'Yes' if x==1 else 'No', default=[0,1])

# Apply filters
df_f = df.copy()
for k,v in filters.items():
    if v: df_f = df_f[df_f[k].isin(v)]

st.sidebar.markdown("---")
st.sidebar.write("Rows after filtering:", len(df_f))

# ------------------ Layout: Visualizations ------------------
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Interactive Visualizations")
    num_cols = [c for c in ['study_hours_per_week','stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Regression modeling")
    target = st.selectbox('Target variable', options=[c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns], index=0)
    predictors = st.multiselect('Predictors', options=[c for c in df.columns if c != target])
    if st.button("Fit model"):
        if not predictors:
            st.error("Choose at least one predictor")
        else:
            formula = f"{target} ~ " + " + ".join(predictors)
            try:
                model = smf.ols(formula=formula, data=df_f).fit()
                st.write("**Formula:**", formula)
                st.write("**R-squared:**", round(model.rsquared, 3))
                st.dataframe(model.params.to_frame('coef'))
            except Exception as e:
                st.error(f"Model fitting failed: {e}")

st.caption("This dashboard is a template for the CSE303 project. CSV is loaded directly from Google Drive.")

