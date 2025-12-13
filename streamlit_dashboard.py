# streamlit_dashboard_final.py
"""
CSE303: Workload vs Mental Health Interactive Dashboard
Features:
- Upload CSV, use local default, or download from Google Drive
- Auto-clean and transform data
- Interactive visualizations: correlation, scatter w/ trendline, boxplots
- Filters with counts, sample warnings
- Regression modeling & diagnostics (OLS, robust SEs, VIF, residuals)
- Advanced modeling with feature engineering for better R² (Ridge, Lasso, interaction terms)
- Export filtered data & model summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ---------------- Streamlit Page ----------------
st.set_page_config(layout="wide", page_title="Workload & Mental Health Dashboard")
st.title("CSE303 — Group 3 project dashboard")


# ---------------- Config ----------------
DEFAULT_LOCAL = Path("./data/cleaned_survey_data.csv")
DEFAULT_LOCAL.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/14NusTm-yg5Wex-EJkF2DHwvGcxRmtZtU/view?usp=drive_link"

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

# ============ FEATURE ENGINEERING FOR BETTER R² ============
def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and polynomial features to improve model fit"""
    df = df.copy()
    
    # Interaction terms
    if 'study_hours_per_week' in df.columns and 'sleep_hours' in df.columns:
        df['study_sleep_interaction'] = df['study_hours_per_week'] * df['sleep_hours']
    
    if 'job_hours_per_week' in df.columns and 'study_hours_per_week' in df.columns:
        df['job_study_interaction'] = df['job_hours_per_week'] * df['study_hours_per_week']
    
    # Sleep quality interactions
    if 'sleep_hours' in df.columns and 'sleep_quality_1to10' in df.columns:
        df['sleep_quality_hours_interaction'] = df['sleep_hours'] * df['sleep_quality_1to10']
    
    # Sleep deficit indicator
    if 'sleep_hours' in df.columns:
        df['sleep_deficit'] = (df['sleep_hours'] < 6).astype(int)
        df['sleep_squared'] = df['sleep_hours'] ** 2
    
    # Total workload
    workload_cols = ['study_hours_per_week', 'job_hours_per_week', 'extra_hours_per_week']
    available_workload = [c for c in workload_cols if c in df.columns]
    if len(available_workload) >= 2:
        df['total_workload'] = df[available_workload].sum(axis=1)
    
    # Low sleep quality indicator
    if 'sleep_quality_1to10' in df.columns:
        df['low_sleep_quality'] = (df['sleep_quality_1to10'] < 6).astype(int)
    
    return df

# ============ FEATURE SELECTION FOR BETTER R² ============
def select_best_predictors(df: pd.DataFrame, target: str, candidate_predictors: list, max_features: int = 8) -> list:
    """Use correlation to select best predictors and remove multicollinearity"""
    df_clean = df[[target] + candidate_predictors].dropna()
    
    if df_clean.shape[0] < 10:
        return candidate_predictors
    
    # Calculate correlations with target
    correlations = df_clean[candidate_predictors].corrwith(df_clean[target]).abs().sort_values(ascending=False)
    
    # Select features by correlation and VIF
    selected = []
    for feat in correlations.index:
        if len(selected) >= max_features:
            break
        test_features = selected + [feat]
        if len(test_features) <= 2:
            selected.append(feat)
        else:
            test_df = df_clean[test_features].fillna(0)
            test_x = sm.add_constant(test_df)
            try:
                vifs = [variance_inflation_factor(test_x.values, i) for i in range(test_x.shape[1])]
                if max(vifs[1:]) < 5:
                    selected.append(feat)
            except:
                selected.append(feat)
    
    return selected

# ============ ADVANCED MULTI-MODEL FITTING ============
def fit_multiple_models_advanced(df: pd.DataFrame, target: str, predictors: list):
    """Fit OLS, Ridge, and Lasso models with scaling"""
    df_clean = df[[target] + predictors].dropna()
    
    if df_clean.shape[0] < (len(predictors) + 5):
        return None
    
    X = df_clean[predictors].values
    y = df_clean[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # OLS Model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_ols = lr.predict(X_test_scaled)
    results['OLS'] = {
        'r2': r2_score(y_test, y_pred_ols),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ols)),
        'mae': mean_absolute_error(y_test, y_pred_ols),
        'model': lr,
        'y_pred': y_pred_ols
    }
    
    # Ridge Regression with GridSearch
    try:
        ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        ridge = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        results['Ridge'] = {
            'r2': r2_score(y_test, y_pred_ridge),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'mae': mean_absolute_error(y_test, y_pred_ridge),
            'best_alpha': ridge.best_params_['alpha'],
            'model': ridge.best_estimator_,
            'y_pred': y_pred_ridge
        }
    except:
        pass
    
    # Lasso Regression with GridSearch
    try:
        lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
        lasso = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring='r2')
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        results['Lasso'] = {
            'r2': r2_score(y_test, y_pred_lasso),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            'mae': mean_absolute_error(y_test, y_pred_lasso),
            'best_alpha': lasso.best_params_['alpha'],
            'model': lasso.best_estimator_,
            'y_pred': y_pred_lasso
        }
    except:
        pass
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(LinearRegression(), X, y, scoring='r2', cv=cv)
    
    return {
        'models': results,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred_ols': y_pred_ols,
        'cv_scores': cv_scores,
        'scaler': scaler
    }

# ============ VIF COMPUTATION ============
def compute_vif_df(X_df: pd.DataFrame):
    Xc = sm.add_constant(X_df.fillna(0))
    vif_list = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    return pd.DataFrame({'variable': Xc.columns, 'VIF': vif_list})

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
uploaded = st.file_uploader("Upload CSV file if there is a issue in drive", type=["csv"])
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

# ============ CREATE ADVANCED FEATURES ============
df = create_advanced_features(df)

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

# ============ ADVANCED REGRESSION & DIAGNOSTICS ============
with col2:
    st.subheader("Regression Modeling")
    target_candidates = [c for c in ['stress_1to5','anxiety_1to5','sleep_hours'] if c in df.columns]
    if target_candidates:
        target = st.selectbox("Target variable", options=target_candidates, index=0)
        default_preds = [c for c in num_cols if c != target]
        extra_cats = [f"C({c})" for c in ['year_of_study','gender'] if c in df.columns]
        predictors = st.multiselect("Predictors", options=default_preds + extra_cats, default=default_preds[:3])

        modeling_mode = st.radio("Model Type", ["Standard OLS", "Advanced (Multiple Models)"], horizontal=True)

        if st.button("Fit model"):
            if predictors:
                if modeling_mode == "Standard OLS":
                    # Original OLS functionality
                    formula = f"{target} ~ " + " + ".join(predictors)
                    model_full = smf.ols(formula=formula, data=df_f).fit()
                    st.write("**Formula:**", formula)
                    st.write("**R²:**", round(model_full.rsquared,3))
                    st.dataframe(model_full.params.to_frame("coef"))
                    st.text_area("Model summary", model_full.summary().as_text(), height=300)
                else:
                    # Advanced multi-model approach
                    numeric_preds = [p for p in predictors if not p.startswith('C(')]
                    if numeric_preds:
                        adv_results = fit_multiple_models_advanced(df_f, target, numeric_preds)
                        if adv_results:
                            st.write("### Advanced Model Comparison")
                            
                            # Model metrics table
                            metrics_data = []
                            for model_name, metrics in adv_results['models'].items():
                                metrics_data.append({
                                    'Model': model_name,
                                    'R²': round(metrics['r2'], 4),
                                    'RMSE': round(metrics['rmse'], 4),
                                    'MAE': round(metrics['mae'], 4)
                                })
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # CV Scores
                            st.write(f"**Cross-validation R² (mean ± std):** {adv_results['cv_scores'].mean():.4f} ± {adv_results['cv_scores'].std():.4f}")
                            
                            # Residual diagnostics
                            st.write("### Diagnostics")
                            residuals = adv_results['y_test'] - adv_results['y_pred_ols']
                            
                            fig_diag = go.Figure()
                            fig_diag.add_trace(go.Scatter(x=adv_results['y_pred_ols'], y=residuals, mode='markers', 
                                                          name='Residuals', marker=dict(size=6, opacity=0.6)))
                            fig_diag.add_hline(y=0, line_dash="dash", line_color="red")
                            fig_diag.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted Values", yaxis_title="Residuals")
                            st.plotly_chart(fig_diag, use_container_width=True)
                        else:
                            st.error("Not enough data for advanced modeling. Try fewer predictors or get more samples.")
                    else:
                        st.warning("Advanced mode requires numeric predictors only (no categorical).")

# ============ ORIGINAL EXPORT FUNCTIONALITY ============
st.markdown("---")
st.subheader("Export / Save")
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered dataset as CSV", data=csv_bytes, file_name="filtered_survey.csv", mime="text/csv")
