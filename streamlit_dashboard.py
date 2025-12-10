# streamlit_dashboard_full.py
"""
Streamlit dashboard (improved) for CSE303 project.
Features:
- Upload or auto-download cleaned CSV (Drive link)
- Interactive visuals: correlation, scatter w/ trendline, boxplots
- Filters with counts, sample size warnings
- Model fitting with train/test evaluation, CV, robust SEs
- Diagnostics: residuals, QQ, Breusch-Pagan, VIF
- Export filtered CSV and download model summary text
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

st.set_page_config(layout="wide", page_title="Workload & Mental Health Dashboard")
st.title("CSE303 — Workload vs Mental Health (Interactive Dashboard)")
st.markdown("Upload cleaned CSV, or let the app load a default local file. Dashboard shows diagnostics, train/test, CV, and export options.")

# ---------------- Config ----------------
DEFAULT_LOCAL = Path("./data/cleaned_survey_data.csv")
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1OEfj7aU-AkkRbGUl8QgrCpDuWrZig0c1/view?usp=drive_link"

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

@st.cache_data
def read_csv_from_path(path_str: str):
    return pd.read_csv(path_str)

@st.cache_data
def download_drive_and_load(path_str: str, file_id: str):
    try:
        import gdown
    except Exception:
        raise RuntimeError("gdown not installed. Add to requirements or upload manually.")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, path_str, quiet=True)
    if not Path(path_str).exists():
        raise FileNotFoundError("Failed to download from Drive.")
    return pd.read_csv(path_str)

# cleaning helper (same logic as Colab)
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
    # numeric coercion
    numeric_cols = ['age','cgpa','study_hours_per_week','courses_enrolled','extra_hours_per_week',
                    'job_hours_per_week','sleep_hours','sleep_quality_1to10','stress_1to5','anxiety_1to5']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # yes/no normalization
    def norm_yesno(s):
        s = s.astype(str).str.strip().str.lower().replace({'nan': None})
        s = s.replace({'yes.':'yes','yes':'yes','y':'yes','true':'yes','1':'yes',
                       'no.':'no','no':'no','n':'no','false':'no','0':'no'})
        return s
    if 'extracurricular' in df.columns:
        df['extracurricular'] = norm_yesno(df['extracurricular'])
        df['extracurricular_bin'] = (df['extracurricular'] == 'yes').astype(int)
    elif 'extra_hours_per_week' in df.columns:
        df['extracurricular_bin'] = (df['extra_hours_per_week'].fillna(0) > 0).astype(int)
    if 'job' in df.columns:
        df['job'] = norm_yesno(df['job'])
        df['job_bin'] = (df['job'] == 'yes').astype(int)
    elif 'job_hours_per_week' in df.columns:
        df['job_bin'] = (df['job_hours_per_week'].fillna(0) > 0).astype(int)
    if 'year_of_study' in df.columns:
        df['year_of_study'] = pd.to_numeric(df['year_of_study'], errors='coerce').astype('Int64')
    if 'stress_1to5' in df.columns:
        df['high_stress'] = (df['stress_1to5'] >= 4).astype(int)
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.loc[:, df.notna().any()]
    df = df.reset_index(drop=True)
    return df

# ---------------- Acquire dataframe ----------------
uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"])
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

# ---------------- Filters with counts ----------------
st.sidebar.header("Filters")
n_total = len(df)
st.sidebar.write("Total rows:", n_total)
if n_total < 80:
    st.sidebar.warning("Sample size < 80 — interpret subgroup inferences with caution.")

filter_values = {}
if 'year_of_study' in df.columns:
    years = sorted(df['year_of_study'].dropna().unique().tolist())
    year_labels = [f"{y} (n={int((df['year_of_study']==y).sum())})" for y in years]
    selected_years = st.sidebar.multiselect("Year of study", options=years, default=years, format_func=lambda x: f"{x} (n={int((df['year_of_study']==x).sum())})")
    filter_values['year_of_study'] = selected_years
if 'gender' in df.columns:
    genders = sorted(df['gender'].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
    filter_values['gender'] = selected_genders
if 'job_bin' in df.columns:
    opts = ["Yes","No"]
    counts = { "Yes": int((df['job_bin']==1).sum()), "No": int((df['job_bin']==0).sum()) }
    selected_jobs = st.sidebar.multiselect("Job status", options=opts, default=opts, format_func=lambda x: f"{x} (n={counts[x]})")
    filter_values['job_bin'] = selected_jobs
if 'extracurricular_bin' in df.columns:
    opts2 = ["Yes","No"]
    counts2 = { "Yes": int((df['extracurricular_bin']==1).sum()), "No": int((df['extracurricular_bin']==0).sum()) }
    selected_ex = st.sidebar.multiselect("Extracurricular", options=opts2, default=opts2, format_func=lambda x: f"{x} (n={counts2[x]})")
    filter_values['extracurricular_bin'] = selected_ex

# Apply filters
df_f = df.copy()
if 'year_of_study' in filter_values and filter_values['year_of_study']:
    df_f = df_f[df_f['year_of_study'].isin(filter_values['year_of_study'])]
if 'gender' in filter_values and filter_values['gender']:
    df_f = df_f[df_f['gender'].isin(filter_values['gender'])]
if 'job_bin' in filter_values and filter_values['job_bin']:
    ints = [1 if s=="Yes" else 0 for s in filter_values['job_bin']]
    df_f = df_f[df_f['job_bin'].isin(ints)]
if 'extracurricular_bin' in filter_values and filter_values['extracurricular_bin']:
    ints = [1 if s=="Yes" else 0 for s in filter_values['extracurricular_bin']]
    df_f = df_f[df_f['extracurricular_bin'].isin(ints)]

if df_f.empty:
    st.warning("No rows match the selected filters. Widen filters or upload another CSV.")
    st.stop()

# ---------------- Visualizations ----------------
num_cols = [c for c in ['study_hours_per_week','courses_enrolled','extra_hours_per_week','job_hours_per_week',
                        'stress_1to5','anxiety_1to5','sleep_hours','sleep_quality_1to10'] if c in df_f.columns]

col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Interactive Visualizations")
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need >=2 numeric columns for correlation matrix.")

    st.markdown("**Scatter with regression line**")
    if len(num_cols) >= 2:
        x = st.selectbox("X variable", options=num_cols, index=0, key="scatter_x")
        y_options = [c for c in num_cols if c != x]
        y = st.selectbox("Y variable", options=y_options, index=0, key="scatter_y")
        color_opt = 'year_of_study' if 'year_of_study' in df_f.columns else None
        try:
            fig_sc = px.scatter(df_f, x=x, y=y, color=color_opt, trendline="ols", title=f"{y} vs {x}")
            st.plotly_chart(fig_sc, use_container_width=True)
        except Exception as e:
            st.error(f"Could not draw scatter: {e}")
    else:
        st.info("Not enough numeric columns for scatter.")

    st.markdown("**Boxplot by group**")
    group_options = [c for c in ['year_of_study','gender','job_bin','extracurricular_bin'] if c in df_f.columns]
    if group_options and num_cols:
        group = st.selectbox("Group by", options=group_options, index=0, key="box_group")
        num_for_box = st.selectbox("Numeric", options=num_cols, index=0, key="box_numeric")
        df_plot = df_f.copy()
        if group in ['job_bin','extracurricular_bin']:
            df_plot[group] = df_plot[group].map({1:"Yes",0:"No"})
        try:
            fig_box = px.box(df_plot, x=group, y=num_for_box, points="all", title=f"{num_for_box} by {group}")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"Could not produce boxplot: {e}")

with col2:
    st.subheader("Regression modeling & diagnostics")
    target_candidates = [c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns]
    if not target_candidates:
        st.info("No supported target variable present.")
    else:
        target = st.selectbox("Target variable", options=target_candidates, index=0)
        default_preds = [c for c in num_cols if c != target]
        extra_cats = []
        if 'year_of_study' in df.columns: extra_cats.append('C(year_of_study)')
        if 'gender' in df.columns: extra_cats.append('C(gender)')
        predictors = st.multiselect("Predictors", options=default_preds + extra_cats, default=default_preds[:3])

        if st.button("Fit model"):
            if not predictors:
                st.error("Select predictors.")
            else:
                formula = f"{target} ~ " + " + ".join(predictors)
                try:
                    model_full = smf.ols(formula=formula, data=df_f).fit()
                    st.write("**Formula:**", formula)
                    st.write("**R² (full data):**", round(model_full.rsquared,3))
                    st.dataframe(model_full.params.to_frame("coef"))

                    # Model summary text (for download)
                    model_text = model_full.summary().as_text()
                    st.text_area("Model summary (text)", value=model_text, height=300)

                    # Train/test evaluation using dummified features for sklearn
                    # Prepare X for sklearn CV (dummify categorical columns used)
                    cat_cols = [p[2:-1] for p in predictors if p.startswith('C(') and p.endswith(')')]
                    num_preds = [p for p in predictors if not (p.startswith('C(') and p.endswith(')'))]
                    df_model = df_f[[target] + num_preds + cat_cols].dropna()
                    if df_model.shape[0] < 8:
                        st.warning("Small sample for train/test split — results may be unstable.")
                    if df_model.shape[0] >= 3:
                        X_sklearn = pd.get_dummies(df_model[num_preds + cat_cols], drop_first=True)
                        y_sklearn = df_model[target]
                        X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y_sklearn, test_size=0.3, random_state=42)
                        if X_train.shape[0] >= 1:
                            lr = LinearRegression()
                            lr.fit(X_train, y_train)
                            preds = lr.predict(X_test)
                            test_rmse = np.sqrt(mean_squared_error(y_test, preds))
                            test_r2 = r2_score(y_test, preds)
                            st.write(f"Test RMSE: {test_rmse:.3f}, Test R²: {test_r2:.3f}")
                            # CV
                            cv = KFold(n_splits=min(5, max(2, int(np.floor(len(y_sklearn)/2)))), shuffle=True, random_state=42)
                            cv_scores = cross_val_score(lr, X_sklearn, y_sklearn, scoring='r2', cv=cv)
                            st.write(f"CV R² mean = {np.mean(cv_scores):.3f}, std = {np.std(cv_scores):.3f}")
                        else:
                            st.info("Not enough data to perform train/test evaluation after dummification.")
                    else:
                        st.info("Not enough rows for train/test/CV evaluation.")

                    # Residual diagnostics
                    resid = model_full.resid
                    fitted = model_full.fittedvalues
                    fig_res = px.scatter(x=fitted, y=resid, labels={'x':'Fitted','y':'Residuals'}, title="Residuals vs Fitted")
                    fig_res.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig_res, use_container_width=True)
                    fig_qq = sm.qqplot(resid, line='45', fit=True).get_figure()
                    st.pyplot(fig_qq)
                    plt.close(fig_qq)

                    # Breusch-Pagan
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        lm, lm_p, fval, f_p = het_breuschpagan(resid, model_full.model.exog)
                        st.write("Breusch-Pagan p (heteroscedasticity):", round(f_p,4))
                        if f_p < 0.05:
                            st.warning("Heteroscedasticity detected (p < 0.05). Consider robust SEs.")
                            robust = model_full.get_robustcov_results(cov_type='HC3')
                            st.text("Robust SE summary (HC3):\n" + robust.summary().as_text())
                    except Exception as e:
                        st.info("Breusch-Pagan not available:", e)

                    # VIF for numeric predictors
                    numeric_for_vif = [p for p in num_preds if p in df_f.columns]
                    if numeric_for_vif:
                        Xv = sm.add_constant(df_f[numeric_for_vif].dropna())
                        vif_df = pd.DataFrame({'variable': Xv.columns, 'VIF': [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]})
                        st.write("VIF (includes intercept):")
                        st.dataframe(vif_df)
                        if vif_df['VIF'].max() > 5:
                            st.warning("Some VIFs > 5: potential multicollinearity. Consider dropping or combining predictors.")

                    # Allow download of model summary text
                    st.download_button("Download model summary (text)", data=model_text, file_name="model_summary.txt", mime="text/plain")

                except Exception as e:
                    st.error(f"Model fitting failed: {e}")

# ---------------- Export filtered data ----------------
st.markdown("---")
st.subheader("Export / Save")
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered dataset as CSV", data=csv_bytes, file_name="filtered_survey.csv", mime="text/csv")

st.caption("Ensure dataset is anonymized and ethically collected. This dashboard is for educational analysis only.")
