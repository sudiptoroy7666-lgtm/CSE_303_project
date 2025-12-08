# streamlit_dashboard.py
# CSE303: Workload vs Mental Health Dashboard
# Loads CSV from Google Drive or uploaded file, fully interactive

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import plotly.express as px
from pathlib import Path
import re

st.set_page_config(layout='wide', page_title='Workload & Mental Health Dashboard')
st.title('CSE303 — Workload vs Mental Health (Interactive Dashboard)')
st.markdown('Upload your cleaned CSV exported from Colab, or use the default CSV from Google Drive.')

# ------------------ Google Drive CSV ------------------
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1KLnVKougQauelIE9_83vr9uiZreArfoJ/view?usp=drive_link"
OUTPUT_FILENAME = "cleaned_survey_data.csv"
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
CSV_LOCAL_PATH = DATA_DIR / OUTPUT_FILENAME

def extract_drive_id(url):
    match = re.search(r'/d/([^/]+)', url)
    if match: return match.group(1)
    match = re.search(r'[?&]id=([^&]+)', url)
    if match: return match.group(1)
    raise ValueError("Invalid Google Drive link")

DRIVE_FILE_ID = extract_drive_id(GOOGLE_DRIVE_LINK)

# ------------------ Load CSV ------------------
@st.cache_data
def load_csv_from_drive(path, file_id):
    if not path.exists():
        try:
            import gdown
        except ImportError:
            st.error("Please add 'gdown' to requirements.txt for Streamlit Cloud.")
            st.stop()
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info("Downloading CSV from Google Drive...")
        gdown.download(url, str(path), quiet=False)
    return pd.read_csv(path)

@st.cache_data
def load_csv_from_file(file):
    return pd.read_csv(file)

# ------------------ File uploader or default ------------------
uploaded = st.file_uploader('Upload cleaned CSV (recommended)', type=['csv'])
if uploaded:
    df = load_csv_from_file(uploaded)
else:
    df = load_csv_from_drive(CSV_LOCAL_PATH, DRIVE_FILE_ID)

# Stop if CSV not loaded
if df is None or df.empty:
    st.error("CSV could not be loaded.")
    st.stop()

# ------------------ Sidebar: Preview & Mapping ------------------
st.sidebar.header('Preview & sanitization')
if st.sidebar.checkbox("Show raw preview", value=False):
    st.write(df.head())

# --- Column mapping heuristics ---
col_map = {}
cols = [c.lower() for c in df.columns]

def map_column(possible_names, canonical_name):
    for name in possible_names:
        for c in df.columns:
            if name.lower() in c.lower():
                col_map[canonical_name] = c
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

# ------------------ Sidebar filters ------------------
st.sidebar.header('Filters')
filters = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique())
    filters['year_of_study'] = st.sidebar.multiselect('Year of study', options=years, default=years)
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique())
    filters['gender'] = st.sidebar.multiselect('Gender', options=genders, default=genders)
if 'job_bin' in df.columns:
    filters['job_bin'] = st.sidebar.multiselect('Job status', options=[0,1], format_func=lambda x:'Yes' if x==1 else 'No', default=[0,1])

# Apply filters
df_f = df.copy()
for k,v in filters.items():
    if v: df_f = df_f[df_f[k].isin(v)]

st.sidebar.markdown('---')
st.sidebar.write("Rows after filtering:", len(df_f))

# ------------------ Layout ------------------
col1, col2 = st.columns((2,1))

with col1:
    st.subheader('Interactive Visualizations')

    # Correlation heatmap
    num_cols = [c for c in ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week','stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)

    # Scatter with trendline
st.markdown('**Scatter with regression line**')

# X variable selection
x = st.selectbox('X variable', options=num_cols, index=0)

# Compute Y options safely (exclude X)
y_options = [c for c in num_cols if c != x]

if not y_options:
    st.warning("Not enough numeric columns to plot scatter. Add more numeric columns.")
    y = None
else:
    # Ensure index is valid
    y_index = 0 if len(y_options) <= 1 else 1
    y = st.selectbox('Y variable', options=y_options, index=y_index)

# Only plot if both x and y exist
if y:
    color_opt = 'year_of_study' if 'year_of_study' in df_f.columns else None
    fig2 = px.scatter(df_f, x=x, y=y, color=color_opt, trendline='ols', title=f'{y} vs {x}')
    st.plotly_chart(fig2, use_container_width=True)


    # Boxplot by group
    st.markdown('**Boxplot by group**')
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular_bin'] if c in df_f.columns]
    if group_options:
        group = st.selectbox('Group by', options=group_options)
        num = st.selectbox('Numeric for boxplot', options=num_cols, index=0)
        if group in ['job_bin', 'extracurricular_bin']:
            df_f[group] = df_f[group].map({1:'Yes',0:'No'})
        fig3 = px.box(df_f, x=group, y=num, points='all', title=f'{num} by {group}')
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader('Regression modeling')
    target = st.selectbox('Target variable', options=[c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns], index=0)
    default_predictors = [c for c in ['study_hours_per_week','courses_enrolled','extracurricular_bin','job_bin','sleep_hours','sleep_quality_1to10'] if c in df.columns and c!=target]
    predictors = st.multiselect('Predictors', options=default_predictors + ['C(year_of_study)','C(gender)'], default=default_predictors[:3])

    if st.button('Fit model'):
        if not predictors:
            st.error("Choose at least one predictor")
        else:
            formula = f"{target} ~ " + ' + '.join(predictors)
            try:
                model = smf.ols(formula=formula, data=df_f).fit()
                st.write('**Formula:**', formula)
                st.write('**R-squared:**', round(model.rsquared,3))
                st.write('**Coefficients:**')
                st.dataframe(model.params.to_frame('coef'))

                # Equation string
                params = model.params
                intercept = params.get('Intercept',0.0)
                eq_parts = [f"{intercept:.3f}"]
                for name, coef in params.items():
                    if name == 'Intercept': continue
                    safe_name = name.replace('C(year_of_study)','year').replace('C(gender)','gender')
                    eq_parts.append(f"{coef:+.3f}*{safe_name}")
                equation = 'Target_hat = ' + ' '.join(eq_parts)
                st.markdown('**Equation:**')
                st.code(equation)

                # Residual plot
                resid = model.resid
                fitted = model.fittedvalues
                fig_res = px.scatter(x=fitted, y=resid, labels={'x':'Fitted','y':'Residuals'}, title='Residuals vs Fitted')
                fig_res.add_hline(y=0, line_dash='dash')
                st.plotly_chart(fig_res, use_container_width=True)

                # QQ plot
                import matplotlib.pyplot as plt
                import statsmodels.api as sm_aux
                qq = sm_aux.qqplot(resid, line='45', fit=True)
                st.pyplot(qq)

                # Model summary
                st.subheader('Model summary (statsmodels)')
                st.text(model.summary().as_text())

            except Exception as e:
                st.error(f'Model fitting failed: {e}')

# Short interpretation panel
st.markdown('---')
st.subheader('Short automated interpretation (guidance only)')
if 'stress_1to5' in df.columns:
    st.write('Example: if study_hours_per_week is a positive significant predictor of stress_1to5, higher study hours are associated with higher stress (association, not causation). Always check residuals and sample size.')

st.markdown('---')
st.caption('Dashboard template for CSE303 project. Ensure dataset is anonymized and follows ethical guidelines.')
