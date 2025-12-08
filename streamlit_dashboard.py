# streamlit_dashboard.py
# Streamlit dashboard for CSE303 project - Workload vs Mental Health
# Usage: streamlit run streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import plotly.express as px
from pathlib import Path
from io import StringIO

st.set_page_config(layout='wide', page_title='Workload & Mental Health Dashboard')

st.title('CSE303 — Workload vs Mental Health (Interactive Dashboard)')
st.markdown('Upload your cleaned CSV exported from the Colab notebook, or use the default `cleaned_survey_data.csv` if present in the same folder as this app.')

# Helper to attempt to find cleaned file from common locations
DEFAULT_CANDIDATES = [
    'cleaned_survey_data.csv',
    './data/cleaned_survey_data.csv',
    './data/cleaned_survey_data.csv',
    './cleaned_survey_data.csv'
]

@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

# File uploader or use default
uploaded = st.file_uploader('Upload cleaned CSV (recommended)', type=['csv'])
use_default = False
if uploaded is None:
    # try to locate default
    found = None
    for p in DEFAULT_CANDIDATES:
        if Path(p).exists():
            found = p
            break
    if found:
        st.sidebar.success(f'Found cleaned file at: {found}')
        use_default = st.sidebar.button('Load default cleaned_survey_data.csv')
        if use_default:
            df = load_csv_from_path(found)
    else:
        st.info('No default CSV found. Upload one using the file uploader on the top.')
        df = None
else:
    # read uploaded
    df = pd.read_csv(uploaded)

if df is None:
    st.stop()

st.sidebar.header('Preview & sanitization')
if st.sidebar.checkbox('Show raw preview', value=False):
    st.write(df.head())

# Map common column name variants to canonical names used by app
col_map = {}
cols = [c.lower() for c in df.columns]
# mapping heuristics
if 'study_hours_per_week' in df.columns:
    col_map['study_hours_per_week'] = 'study_hours_per_week'
else:
    for c in df.columns:
        if 'study' in c.lower() and 'hour' in c.lower():
            col_map['study_hours_per_week'] = c
            break

if 'stress_1to5' in df.columns:
    col_map['stress_1to5'] = 'stress_1to5'
else:
    for c in df.columns:
        if 'stress' in c.lower():
            col_map['stress_1to5'] = c
            break

if 'anxiety_1to5' in df.columns:
    col_map['anxiety_1to5'] = 'anxiety_1to5'
else:
    for c in df.columns:
        if 'anxiety' in c.lower():
            col_map['anxiety_1to5'] = c
            break

if 'sleep_hours' in df.columns:
    col_map['sleep_hours'] = 'sleep_hours'
else:
    for c in df.columns:
        if 'sleep' in c.lower() and 'hour' in c.lower():
            col_map['sleep_hours'] = c
            break

if 'sleep_quality_1to10' in df.columns:
    col_map['sleep_quality_1to10'] = 'sleep_quality_1to10'
else:
    for c in df.columns:
        if 'sleep' in c.lower() and 'qual' in c.lower():
            col_map['sleep_quality_1to10'] = c
            break

if 'extracurricular_bin' in df.columns:
    col_map['extracurricular_bin'] = 'extracurricular_bin'
else:
    for c in df.columns:
        if 'extra' in c.lower() and ('bin' in c.lower() or 'extracurricular' in c.lower()):
            col_map['extracurricular_bin'] = c
            break

if 'job_bin' in df.columns:
    col_map['job_bin'] = 'job_bin'
else:
    for c in df.columns:
        if 'job' in c.lower() and ('bin' in c.lower() or 'job' in c.lower()):
            col_map['job_bin'] = c
            break

if 'year_of_study' in df.columns:
    col_map['year_of_study'] = 'year_of_study'
else:
    for c in df.columns:
        if 'year' in c.lower():
            col_map['year_of_study'] = c
            break

if 'gender' in df.columns:
    col_map['gender'] = 'gender'
else:
    for c in df.columns:
        if 'gender' in c.lower():
            col_map['gender'] = c
            break

# Apply renaming
rename_actual = {v:k for k,v in col_map.items() if v in df.columns}
df = df.rename(columns=rename_actual)

# Ensure numeric types where expected
num_cols_expected = ['study_hours_per_week','stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10']
for c in num_cols_expected:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Derived variables
if 'stress_1to5' in df.columns:
    df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)

# Basic checks & sample size warnings
n = len(df)
st.sidebar.markdown(f'**Total responses:** {n}')
if n < 80:
    st.sidebar.warning('Sample size is < 80. Project guideline requests >= 80 responses — interpret subgroup results with caution.')

# Sidebar filters
st.sidebar.header('Filters')
filters = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique())
    sel_years = st.sidebar.multiselect('Year of study', options=years, default=years)
    filters['year_of_study'] = sel_years
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique())
    sel_genders = st.sidebar.multiselect('Gender', options=genders, default=genders)
    filters['gender'] = sel_genders
if 'job_bin' in df.columns:
    sel_jobs = st.sidebar.multiselect('Job status', options=[0,1], format_func=lambda x: 'Yes' if x==1 else 'No', default=[0,1])
    filters['job_bin'] = sel_jobs

# Apply filters
df_f = df.copy()
for k,v in filters.items():
    if v is None or len(v)==0:
        continue
    df_f = df_f[df_f[k].isin(v)]

st.sidebar.markdown('---')
st.sidebar.write('Rows after filtering: ', len(df_f))

# Layout
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
    x = st.selectbox('X variable', options=num_cols, index=0)
    y = st.selectbox('Y variable', options=[c for c in num_cols if c!=x], index=1 if len(num_cols)>1 else 0)
    color_opt = None
    if 'year_of_study' in df_f.columns:
        color_opt = 'year_of_study'
    fig2 = px.scatter(df_f, x=x, y=y, color=color_opt, trendline='ols', title=f'{y} vs {x}')
    st.plotly_chart(fig2, use_container_width=True)

    # Boxplot by group
    st.markdown('**Boxplot by group**')
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular'] if c in df_f.columns]
    if group_options:
        group = st.selectbox('Group by', options=group_options)
        num = st.selectbox('Numeric for boxplot', options=num_cols, index=0)
        if group in ['job_bin']:
            df_f[group] = df_f[group].map({1:'Yes',0:'No'})
        fig3 = px.box(df_f, x=group, y=num, points='all', title=f'{num} by {group}')
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader('Regression modeling')
    target = st.selectbox('Target variable', options=[c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns], index=0)

    # Predictor selection
    default_predictors = [c for c in ['study_hours_per_week','courses_enrolled','extracurricular_bin','job_bin','sleep_hours','sleep_quality_1to10'] if c in df.columns and c!=target]
    predictors = st.multiselect('Predictors', options=default_predictors + [ 'C(year_of_study)', 'C(gender)' ], default=default_predictors[:3])

    if st.button('Fit model'):
        if len(predictors) == 0:
            st.error('Choose at least one predictor')
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
                intercept = params.get('Intercept', 0.0)
                eq_parts = [f"{intercept:.3f}"]
                for name, coef in params.items():
                    if name == 'Intercept':
                        continue
                    safe_name = name.replace('C(year_of_study)','year').replace('C(gender)','gender')
                    eq_parts.append(f"{coef:+.3f}*{safe_name}")
                equation = 'Target_hat = ' + ' '.join(eq_parts)
                st.markdown('**Equation:**')
                st.code(equation)

                # Diagnostics: residual plot and QQ
                resid = model.resid
                fitted = model.fittedvalues
                fig_res = px.scatter(x=fitted, y=resid, labels={'x':'Fitted','y':'Residuals'}, title='Residuals vs Fitted')
                fig_res.add_hline(y=0, line_dash='dash')
                st.plotly_chart(fig_res, use_container_width=True)

                import matplotlib.pyplot as plt
                import statsmodels.api as sm_aux
                qq = sm_aux.qqplot(resid, line='45', fit=True)
                st.pyplot(qq)

                # Show model summary (text)
                st.subheader('Model summary (statsmodels)')
                st.text(model.summary().as_text())

            except Exception as e:
                st.error(f'Model fitting failed: {e}')

# Short interpretation panel
st.markdown('---')
st.subheader('Short automated interpretation (very quick, for guidance only)')
if 'stress_1to5' in df.columns:
    st.write('Example: if study_hours_per_week is a positive significant predictor of stress_1to5, it suggests higher study hours are associated with higher reported stress score (association, not causation). Always check residuals and sample size before concluding.)')

st.markdown('---')
st.caption('This dashboard is a template for the CSE303 project. Adapt column names if your CSV uses different headers. Ensure your dataset follows privacy and ethical guidelines (anonymized responses).')
