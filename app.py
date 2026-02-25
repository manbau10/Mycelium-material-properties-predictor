# app.py
# Single-file Streamlit app with two pages:
# - About the app
# - Optimized models with synthetic data
#
# Final models used by the prediction page:
#   Tensile Strength:  lightgbm_tensile_strength_bootstrap_sampling.pkl
#   Elongation @ Break: xgboost_elongation_at_break_gaussian_noise.pkl
#   Young's Modulus:   xgboost_young_modulus_gaussian_noise.pkl
#
# Encoders expected under encoders/encoder_<feature>.pkl
# Optional models_metadata.json can define feature_columns order.

import streamlit as st
import pickle
import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import LightGBM so unpickling LightGBM models works
try:
    import lightgbm  # noqa: F401
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# ---------------------------- SIMPLE LABEL ENCODER ----------------------------
# Pure-Python drop-in for sklearn LabelEncoder.
# MUST be defined here so pickle.load() can find the class when loading
# encoder_<feature>.pkl files (which were created using this same class).
# This avoids any numpy version compatibility issues entirely.
class SimpleLabelEncoder:
    """Drop-in replacement for sklearn LabelEncoder using only Python built-ins.
    Compatible with any numpy/sklearn version because it never touches numpy."""
    def __init__(self, classes):
        self.classes_ = sorted([str(c) for c in classes])
        self._lookup = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, values):
        return [self._lookup[v] for v in values]

    def inverse_transform(self, codes):
        return [self.classes_[i] for i in codes]

# ---------------------------- PAGE CONFIGURATION ----------------------------
st.set_page_config(
    page_title="Mycelium Material Properties Predictor",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------- GLOBAL STYLES ----------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #2E8B57; text-align: center; margin: 0.5rem 0 1rem 0; }
    .sub-header { font-size: 1.2rem; color: #1F4E79; margin: 0.8rem 0 0.6rem 0; font-weight: 700; }
    .note { background: #FFF7E6; border-left: 4px solid #F59E0B; padding: 0.8rem 1rem; border-radius: 4px; margin: 0.6rem 0 1rem 0; color: #5A3B00; }
    .good { background: #ECFDF5; border-left: 4px solid #10B981; padding: 0.8rem 1rem; border-radius: 4px; margin: 0.6rem 0 1rem 0; color: #065F46; }
    .danger { background: #FEF2F2; border-left: 4px solid #EF4444; padding: 0.8rem 1rem; border-radius: 4px; margin: 0.6rem 0 1rem 0; color: #7F1D1D; }
    .img-caption { text-align:center; font-size:0.9rem; color:#374151; margin-top:0.4rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- UTILITIES ----------------------------
def get_feature_options_from_encoders(encoders):
    options = {}
    for name, encoder in encoders.items():
        if hasattr(encoder, 'classes_'):
            options[name] = list(encoder.classes_)
    return options

def infer_feature_order(metadata):
    if metadata and 'feature_columns' in metadata:
        return metadata['feature_columns']
    return [
        'fungi_type', 'substrate', 'inoculation_state', 'incubation_temperature',
        'incubation_condition', 'growth_condition', 'reinforcement', 'crosslinking', 'plasticizing'
    ]

def prepare_input_data(input_data, encoders, feature_order, *, report_unseen=False):
    try:
        processed = input_data.copy()
        categorical = [
            'fungi_type', 'substrate', 'inoculation_state', 'incubation_condition',
            'growth_condition', 'reinforcement', 'crosslinking', 'plasticizing'
        ]
        unseen = []
        for col in categorical:
            if col in encoders:
                val = processed.get(col, None)
                if val in encoders[col].classes_:
                    processed[col] = int(encoders[col].transform([val])[0])
                else:
                    processed[col] = 0
                    unseen.append((col, val))
            else:
                processed[col] = 0
        if report_unseen and unseen:
            st.warning("Some categories are not in the encoders and were mapped to a default code: " +
                       ", ".join([f"{c}: {v}" for c, v in unseen if v is not None]))

        processed['incubation_temperature'] = float(processed.get('incubation_temperature', 0.0))

        df_dict = {}
        for col in feature_order:
            df_dict[col] = float(processed.get(col, 0.0))
        df = pd.DataFrame([df_dict]).reindex(columns=feature_order, fill_value=0.0)
        return df
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def create_visual(predictions):
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        column_widths=[0.31, 0.31, 0.31],
        subplot_titles=("Tensile Strength (MPa)", "Elongation at Break (%)", "Young's Modulus (MPa)")
    )
    inset_domain = [0.05, 0.95]

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predictions['tensile_strength'],
        domain={'x': inset_domain, 'y': [0, 1]},
        number={'font': {'size': 28}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 50], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "green"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
        }
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predictions['elongation_at_break'],
        domain={'x': inset_domain, 'y': [0, 1]},
        number={'font': {'size': 28}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "blue"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
        }
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predictions['young_modulus'],
        domain={'x': inset_domain, 'y': [0, 1]},
        number={'font': {'size': 28}},
        gauge={
            'shape': 'angular',
            'axis': {
                'range': [0, 3000],
                'tickmode': 'array',
                'tickvals': [0, 500, 1000, 1500, 2000, 2500, 3000],
                'ticktext': ['0', '500', '1000', '1500', '2000', '2500', '3000'],
                'tickwidth': 1,
                'tickcolor': "black",
                'tickformat': ",d",
                'separatethousands': True
            },
            'bar': {'color': "orange"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
        }
    ), row=1, col=3)

    fig.update_layout(
        height=320,
        width=880,
        margin=dict(t=70, b=30, l=20, r=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12),
        showlegend=False
    )
    return fig

def sidebar_inputs(encoders):
    st.sidebar.markdown("### 🧬 Input Parameters")
    options = get_feature_options_from_encoders(encoders)
    fungi_type = st.sidebar.selectbox("Fungi Type", options.get('fungi_type', ['Unknown']))
    substrate = st.sidebar.selectbox("Substrate", options.get('substrate', ['Unknown']))
    inoculation_state = st.sidebar.selectbox("Inoculation State", options.get('inoculation_state', ['Unknown']))
    incubation_condition = st.sidebar.selectbox("Incubation Condition", options.get('incubation_condition', ['Unknown']))
    growth_condition = st.sidebar.selectbox("Growth Condition", options.get('growth_condition', ['Unknown']))
    reinforcement = st.sidebar.selectbox("Reinforcement", options.get('reinforcement', ['Unknown']))
    crosslinking = st.sidebar.selectbox("Crosslinking", options.get('crosslinking', ['Unknown']))
    plasticizing = st.sidebar.selectbox("Plasticizing", options.get('plasticizing', ['Unknown']))
    incubation_temperature = st.sidebar.number_input("Incubation Temperature (°C)", min_value=0.0, max_value=100.0, value=28.0)
    return {
        'fungi_type': fungi_type,
        'substrate': substrate,
        'inoculation_state': inoculation_state,
        'incubation_condition': incubation_condition,
        'growth_condition': growth_condition,
        'reinforcement': reinforcement,
        'crosslinking': crosslinking,
        'plasticizing': plasticizing,
        'incubation_temperature': incubation_temperature
    }

# ---------------------------- LOADERS ----------------------------
@st.cache_data
def load_encoders_and_metadata():
    encoders, metadata = {}, {}
    encoder_features = [
        'fungi_type', 'substrate', 'inoculation_state', 'incubation_condition',
        'growth_condition', 'reinforcement', 'crosslinking', 'plasticizing'
    ]
    for feature in encoder_features:
        path = f'encoders/encoder_{feature}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                encoders[feature] = pickle.load(f)
    if os.path.exists('models_metadata.json'):
        with open('models_metadata.json', 'r') as f:
            metadata = json.load(f)
    return encoders, metadata

@st.cache_data
def load_models_augmented():
    if not _HAS_LGBM:
        st.error("LightGBM is required for the augmented models. Please install it with: pip install lightgbm")
        return None
    models = {}
    try:
        with open('Models/lightgbm_tensile_strength_bootstrap_sampling.pkl', 'rb') as f:
            models['tensile_strength'] = pickle.load(f)
        with open('Models/xgboost_elongation_at_break_gaussian_noise.pkl', 'rb') as f:
            models['elongation_at_break'] = pickle.load(f)
        with open('Models/xgboost_young_modulus_gaussian_noise.pkl', 'rb') as f:
            models['young_modulus'] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading augmented models: {e}")
        return None

# ---------------------------- PAGES ----------------------------
def page_about():
    st.markdown('<h1 class="main-header">🍄 Mycelium Material Properties Predictor</h1>', unsafe_allow_html=True)
    st.write("Estimate mechanical properties of mycelium-based materials from process parameters using robust, validated machine learning models.")

    st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
- Targets predicted:
  - Tensile Strength (MPa)
  - Elongation at Break (%)
  - Young's Modulus (MPa)
- Model selection process:
  - We evaluated five model families for each target: ANN, LightGBM, Random Forest, XGBoost, and CatBoost.
  - We also evaluated three data augmentation strategies suitable for regression: SMOTE for regression, Bootstrap Sampling, and Gaussian Noise.
  - For each target, we selected the best-performing combination (model + augmentation) based on validation metrics and generalization behavior.
- The app uses the final selected models trained with the best augmentation per target.
""")

    st.markdown('<div class="sub-header">Final models used in this app</div>', unsafe_allow_html=True)
    st.markdown("""
- Tensile Strength: LightGBM with Bootstrap Sampling
- Elongation at Break: XGBoost with Gaussian Noise
- Young's Modulus: XGBoost with Gaussian Noise
""")

    st.markdown('<div class="sub-header">Why augmentation?</div>', unsafe_allow_html=True)
    st.markdown("""
- Mycelium datasets are often small and heterogeneous.
- Augmentation helps reduce overfitting and better reflect measurement/process variability:
  - SMOTE for regression: synthesizes new samples in feature space.
  - Bootstrap sampling: resamples with replacement to simulate multiple draws from the data distribution.
  - Gaussian noise: perturbs inputs or targets slightly to promote robustness.
""")

    st.markdown('<div class="sub-header">Validation snapshots (all models, train vs. test)</div>', unsafe_allow_html=True)
    st.caption("These figures illustrate how different model families performed under the most suitable augmentation strategy for each target. The final choices for this app are those with the most consistent generalization on the test splits.")
    col1 = st.container()
    with col1:
        st.image("images/ts_bootstrap_all_models.png", use_column_width=True, caption="Bootstrap Sampling – Tensile Strength: All evaluated models (train/test).")
        st.markdown('<div class="img-caption">We selected LightGBM + Bootstrap for Tensile Strength.</div>', unsafe_allow_html=True)

    col2 = st.container()
    with col2:
        st.image("images/eab_gaussian_all_models.png", use_column_width=True, caption="Gaussian Noise – Elongation at Break: All evaluated models (train/test).")
        st.markdown('<div class="img-caption">We selected XGBoost + Gaussian Noise for Elongation at Break.</div>', unsafe_allow_html=True)

    col3 = st.container()
    with col3:
        st.image("images/ym_gaussian_all_models.png", use_column_width=True, caption="Gaussian Noise – Young's Modulus: All evaluated models (train/test).")
        st.markdown('<div class="img-caption">We selected XGBoost + Gaussian Noise for Young\u2019s Modulus.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">How to use</div>', unsafe_allow_html=True)
    st.markdown("""
1) Go to "Optimized models with synthetic data" using the sidebar.
2) Select your process parameters in the sidebar.
3) Click "Predict Properties".
4) Review the metrics and gauges. Iterate on inputs to explore sensitivity.
""")

    st.markdown('<div class="note">Interpreting predictions</div>', unsafe_allow_html=True)
    st.markdown("""
- Predictions are point estimates; treat them as guidance for experiment planning.
- Consider experimental uncertainty and domain knowledge when acting on results.
""")

    st.markdown('<div class="danger">Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
- Reliability decreases outside the training distribution.
- Unseen categorical values are mapped to a default code; updating encoders and retraining is recommended when adding new categories.
""")

def page_predict():
    title = "Optimized models with synthetic data"
    st.markdown(f'<h1 class="main-header">🍄 {title}</h1>', unsafe_allow_html=True)

    encoders, metadata = load_encoders_and_metadata()
    feature_order = infer_feature_order(metadata)

    if not _HAS_LGBM:
        st.warning("This page requires LightGBM. Install it with: pip install lightgbm")

    models = load_models_augmented()
    if models is None:
        st.stop()
    st.success("✅ Models loaded successfully.")

    input_data = sidebar_inputs(encoders)

    if st.button("🚀 Predict Properties", use_container_width=True):
        df = prepare_input_data(input_data, encoders, feature_order, report_unseen=True)
        if df is not None:
            try:
                preds = {
                    'tensile_strength': float(models['tensile_strength'].predict(df)[0]),
                    'elongation_at_break': float(models['elongation_at_break'].predict(df)[0]),
                    'young_modulus': float(models['young_modulus'].predict(df)[0]),
                }
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

            st.markdown('<h3 class="sub-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tensile Strength (MPa)", f"{preds['tensile_strength']:.2f}")
            c2.metric("Elongation at Break (%)", f"{preds['elongation_at_break']:.2f}")
            c3.metric("Young's Modulus (MPa)", f"{preds['young_modulus']:.0f}")
            st.plotly_chart(create_visual(preds), use_container_width=True)

# ---------------------------- SIDEBAR NAVIGATION ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=["About the app", "Optimized models with synthetic data"],
    index=0
)

# ---------------------------- ROUTER ----------------------------
if page == "About the app":
    page_about()
elif page == "Optimized models with synthetic data":
    page_predict()
