import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

from pathlib import Path

# =========================
# 1. CONFIG & LOADING
# =========================

st.set_page_config(
    page_title="CVD Risk – Stacking GenAI v3.0 (16 features)",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    """Load scaler, stacking models, and CNN model."""
    # Adjust file names here if needed
    stacking_pkl_path = Path("stacking_genai_v3_16.pkl")
    scaler_pkl_path   = Path("scaler_16.pkl")
    cnn_path          = Path("cnn_genai_v3_16.h5")

    if not stacking_pkl_path.exists():
        st.error(f"Missing file: {stacking_pkl_path}")
        st.stop()
    if not scaler_pkl_path.exists():
        st.error(f"Missing file: {scaler_pkl_path}")
        st.stop()
    if not cnn_path.exists():
        st.error(f"Missing file: {cnn_path}")
        st.stop()

    stacking_artifacts = joblib.load(stacking_pkl_path)
    scaler = joblib.load(scaler_pkl_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    meta_model = stacking_artifacts["meta_model"]
    rf_model   = stacking_artifacts["rf_gen"]
    xgb_model  = stacking_artifacts["xgb_gen"]
    features_16 = stacking_artifacts["features_16"]

    return scaler, meta_model, rf_model, xgb_model, cnn_model, features_16

scaler, meta_model, rf_model, xgb_model, cnn_model, FEATURES_16 = load_artifacts()

DEFAULT_THRESHOLD = 0.40


# =========================
# 2. HELPERS
# =========================

def build_input_df():
    """
    Render the UI for the 16 features and return a single-row DataFrame.
    FEATURE ORDER MUST MATCH training pipeline.
    """
    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=95, value=55, step=1)
        sex = st.selectbox("Sex", options=["Female", "Male"])  # SEX: 0 = Female, 1 = Male
        sysbp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=260.0, value=130.0, step=1.0)
        diabp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=28.0, step=0.1)
        heartrate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=150.0, value=75.0, step=1.0)
        educ = st.selectbox(
            "Education level",
            options=[
                "1 – Some High School",
                "2 – High School/GED",
                "3 – Some College/Vocational",
                "4 – College Graduate+"
            ],
            index=2
        )

    with col2:
        totchol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=220.0, step=1.0)
        hdlc = st.number_input("HDL Cholesterol (mg/dL)", min_value=10.0, max_value=150.0, value=45.0, step=1.0)
        ldlc = st.number_input("LDL Cholesterol (mg/dL)", min_value=30.0, max_value=300.0, value=120.0, step=1.0)
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=60.0, max_value=400.0, value=100.0, step=1.0)
        cigs_per_day = st.number_input("Cigarettes per day", min_value=0.0, max_value=80.0, value=0.0, step=1.0)

        diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
        bpmeds   = st.selectbox("On BP medications", options=["No", "Yes"])
        angina   = st.selectbox("Angina / Chest pain", options=["No", "Yes"])
        hyperten = st.selectbox("Hypertension diagnosed", options=["No", "Yes"])

    # Map categorical → numeric to match training
    sex_val      = 1 if sex == "Male" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    bpmeds_val   = 1 if bpmeds == "Yes" else 0
    angina_val   = 1 if angina == "Yes" else 0
    hyperten_val = 1 if hyperten == "Yes" else 0

    # Education is 1–4
    educ_val = int(educ.split("–")[0].strip())

    # Build row in exact feature order
    row = {
        'SEX': sex_val,
        'TOTCHOL': totchol,
        'AGE': age,
        'SYSBP': sysbp,
        'DIABP': diabp,
        'CIGPDAY': cigs_per_day,
        'BMI': bmi,
        'DIABETES': diabetes_val,
        'BPMEDS': bpmeds_val,
        'HEARTRTE': heartrate,
        'GLUCOSE': glucose,
        'educ': educ_val,
        'HDLC': hdlc,
        'LDLC': ldlc,
        'ANGINA': angina_val,
        'HYPERTEN': hyperten_val
    }

    # Ensure ordering matches FEATURES_16 from training
    row_ordered = {feat: row[feat] for feat in FEATURES_16}
    df_input = pd.DataFrame([row_ordered])

    return df_input


def stacking_predict_proba(df_input: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD):
    """
    Run the full v3 Stacking GenAI pipeline:
    - scale features
    - RF / XGB / CNN+GRU probabilities
    - meta LR stacking
    Returns:
      final_prob, final_label, component_probs (dict)
    """
    # Scale
    X_scaled = scaler.transform(df_input.values)

    # CNN input shape: (batch, features, 1)
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Component probabilities
    p_rf  = rf_model.predict_proba(X_scaled)[:, 1]
    p_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
    p_cnn = cnn_model.predict(X_cnn).ravel()

    # Stack
    stack_input = np.column_stack([p_rf, p_xgb, p_cnn])
    p_final = meta_model.predict_proba(stack_input)[:, 1]

    final_prob = float(p_final[0])
    final_label = int(final_prob >= threshold)

    component_probs = {
        "RF (GenAI)": float(p_rf[0]),
        "XGB (GenAI)": float(p_xgb[0]),
        "CNN+GRU (GenAI)": float(p_cnn[0])
    }

    return final_prob, final_label, component_probs


def interpret_risk(prob):
    """Simple textual interpretation for a 10-year CVD risk probability."""
    if prob < 0.05:
        category = "Low risk"
        color = "🟢"
    elif prob < 0.10:
        category = "Borderline risk"
        color = "🟡"
    elif prob < 0.20:
        category = "Intermediate risk"
        color = "🟠"
    else:
        category = "High risk"
        color = "🔴"
    return category, color


# =========================
# 3. UI LAYOUT
# =========================

st.title("CVD Risk Prediction – Stacking Generative AI v3.0")
st.markdown(
    """
This tool estimates **10-year cardiovascular disease (CVD) risk** using a
**Stacking Generative AI model (RF + XGBoost + CNN+GRU + CTGAN)** trained on
the **Framingham Heart Study** with **16 clinically available features**.

> ⚠️ **Disclaimer:**  
> This application is for **research and educational purposes only** and is not
> a substitute for professional medical advice, diagnosis, or treatment.
"""
)

st.sidebar.header("Model Settings")

threshold = st.sidebar.slider(
    "Alert threshold (probability of CVD)",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="If predicted risk ≥ threshold, the model flags the patient as 'At Risk'."
)

show_components = st.sidebar.checkbox(
    "Show component model probabilities (RF/XGB/CNN)", value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model version:** v3.0 – 16 features (no leakage)")

# Input form
df_input = build_input_df()

st.markdown("### Review Encoded Features")
st.dataframe(df_input.style.format(precision=2), use_container_width=True)

if st.button("Run CVD Risk Prediction"):
    with st.spinner("Running Stacking GenAI model..."):
        final_prob, final_label, component_probs = stacking_predict_proba(df_input, threshold=threshold)

    category, color = interpret_risk(final_prob)

    st.markdown("## Prediction Result")
    st.metric(
        label="Estimated 10-year CVD risk (Stacking GenAI v3.0)",
        value=f"{final_prob*100:.1f} %"
    )
    st.markdown(f"**Risk category:** {color} **{category}**")
    st.markdown(
        f"**Model decision at threshold {threshold:.2f}:** "
        f"{'⚠️ At Risk (1)' if final_label == 1 else '✅ Not Flagged (0)'}"
    )

    st.markdown("---")

    if show_components:
        st.subheader("Component Model Probabilities")
        comp_df = pd.DataFrame(
            {
                "Model": list(component_probs.keys()),
                "Predicted CVD risk": [p * 100 for p in component_probs.values()]
            }
        )
        st.bar_chart(
            comp_df.set_index("Model")
        )

    st.markdown("### How to interpret this result")
    st.markdown(
        """
- This probability represents the **estimated 10-year risk** of a major CVD event.
- The model learns from **traditional risk factors** (age, BP, cholesterol, diabetes, smoking, etc.).
- Use this as a **decision-support tool**, not a final diagnosis.
"""
    )
else:
    st.info("Fill in the patient information above and click **'Run CVD Risk Prediction'**.")
