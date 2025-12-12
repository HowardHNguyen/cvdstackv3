import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf
from pathlib import Path

# =========================
# 1. PAGE CONFIG
# =========================

st.set_page_config(
    page_title="CVD Risk – Stacking GenAI v3.0 (16 Features)",
    page_icon="❤️",
    layout="wide"
)

# =========================
# 2. LOAD ARTIFACTS
# =========================

@st.cache_resource
def load_artifacts():
    """Load scaler, stacking models, and CNN model."""
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

    meta_model  = stacking_artifacts["meta_model"]
    rf_model    = stacking_artifacts["rf_gen"]
    xgb_model   = stacking_artifacts["xgb_gen"]
    features_16 = stacking_artifacts["features_16"]

    return scaler, meta_model, rf_model, xgb_model, cnn_model, features_16

scaler, meta_model, rf_model, xgb_model, cnn_model, FEATURES_16 = load_artifacts()

DEFAULT_THRESHOLD = 0.40

# =========================
# 3. HELPER FUNCTIONS
# =========================

def build_input_df():
    """
    Render the UI for the 16 features and return a single-row DataFrame.
    FEATURE ORDER MUST MATCH training pipeline.
    """
    st.subheader("Patient Profile & Clinical Risk Factors")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=95, value=55, step=1)
        sex = st.selectbox("Sex", options=["Female", "Male"])  # SEX: 0 = Female, 1 = Male
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=27.0, step=0.1)
        cigs_per_day = st.number_input("Cigarettes per day", min_value=0.0, max_value=80.0, value=10.0, step=1.0)
        diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
        bpmeds   = st.selectbox("On BP medications", options=["No", "Yes"])

    with col2:
        sysbp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=260.0, value=130.0, step=1.0)
        diabp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
        heartrate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=150.0, value=72.0, step=1.0)
        hyperten = st.selectbox("Hypertension diagnosed", options=["No", "Yes"])
        angina   = st.selectbox("Angina / chest pain", options=["No", "Yes"])

    with col3:
        totchol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=210.0, step=1.0)
        hdlc    = st.number_input("HDL Cholesterol (mg/dL)",   min_value=10.0,  max_value=150.0, value=45.0, step=1.0)
        ldlc    = st.number_input("LDL Cholesterol (mg/dL)",   min_value=30.0,  max_value=300.0, value=120.0, step=1.0)
        glucose = st.number_input("Fasting Glucose (mg/dL)",   min_value=60.0,  max_value=400.0, value=100.0, step=1.0)
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

    # Map categorical → numeric to match training
    sex_val      = 1 if sex == "Male" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    bpmeds_val   = 1 if bpmeds == "Yes" else 0
    angina_val   = 1 if angina == "Yes" else 0
    hyperten_val = 1 if hyperten == "Yes" else 0
    educ_val     = int(educ.split("–")[0].strip())

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
    X_scaled = scaler.transform(df_input.values)
    X_cnn    = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Component probabilities
    p_rf  = rf_model.predict_proba(X_scaled)[:, 1]
    p_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
    p_cnn = cnn_model.predict(X_cnn).ravel()

    stack_input = np.column_stack([p_rf, p_xgb, p_cnn])
    p_final = meta_model.predict_proba(stack_input)[:, 1]

    final_prob  = float(p_final[0])
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


def bie_scenario(df_patient: pd.DataFrame, threshold: float):
    """
    Behavioral Impact Engine (BIE):
    - Baseline risk
    - Scenario: remove smoking (CIGPDAY=0) if patient smokes
    - Scenario: lower systolic BP by 10 mmHg (down to min 90)
    Returns a dict with scenario results.
    """
    df0 = df_patient.copy()

    # Baseline
    base_prob, _, _ = stacking_predict_proba(df0, threshold=threshold)

    results = {
        "baseline_prob": base_prob,
        "smoking": None,
        "bp": None
    }

    # Scenario 1 – smoking → non-smoking (if currently smoking)
    cigs = float(df0["CIGPDAY"].iloc[0])
    if cigs > 0:
        df_no_smoke = df0.copy()
        df_no_smoke["CIGPDAY"] = 0.0
        prob_no_smoke, _, _ = stacking_predict_proba(df_no_smoke, threshold=threshold)

        abs_drop = base_prob - prob_no_smoke
        rel_drop = abs_drop / base_prob if base_prob > 0 else 0.0

        results["smoking"] = {
            "current_cigs": cigs,
            "prob_with_smoke": base_prob,
            "prob_without_smoke": prob_no_smoke,
            "abs_drop": abs_drop,
            "rel_drop": rel_drop
        }

    # Scenario 2 – lower systolic BP by 10 mmHg (down to min 90)
    sysbp = float(df0["SYSBP"].iloc[0])
    new_sysbp = max(sysbp - 10.0, 90.0)
    if new_sysbp != sysbp:
        df_bp = df0.copy()
        df_bp["SYSBP"] = new_sysbp
        prob_bp, _, _ = stacking_predict_proba(df_bp, threshold=threshold)

        abs_drop_bp = base_prob - prob_bp
        rel_drop_bp = abs_drop_bp / base_prob if base_prob > 0 else 0.0

        results["bp"] = {
            "current_sysbp": sysbp,
            "new_sysbp": new_sysbp,
            "prob_current": base_prob,
            "prob_lower_bp": prob_bp,
            "abs_drop": abs_drop_bp,
            "rel_drop": rel_drop_bp
        }

    return results

# =========================
# 4. SIDEBAR
# =========================

with st.sidebar:
    st.markdown(
        """
        <h2 style='margin-bottom:0;'>❤️ CVD Stacking GenAI v3.0</h2>
        <p style='margin-top:4px;font-size:13px;'>
        16-feature, non-leakage model • Framingham-based • CTGAN-balanced
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    threshold = st.slider(
        "Alert threshold (probability of CVD)",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="If predicted risk ≥ threshold, the model flags the patient as 'At Risk'."
    )

    show_components = st.checkbox(
        "Show component model probabilities (RF/XGB/CNN)",
        value=True
    )

    st.markdown("---")
    st.markdown(
        """
        **Model version:** v3.0  
        **Features:** 16 (no STROKE / MI_FCHD leakage)  
        **Engine:** RF + XGB + CNN+GRU + CTGAN + Stacking
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **Disclaimer:**  
        This tool is for **research & education** only and must not be used as
        a standalone diagnostic system.
        """
    )

# =========================
# 5. HEADER / HERO SECTION
# =========================

st.markdown(
    """
    <div style="background-color:#0f4c75;padding:18px;border-radius:8px;margin-bottom:16px;">
      <h1 style="color:white;margin-bottom:4px;">CVD Risk Prediction – Stacking Generative AI v3.0</h1>
      <p style="color:#e0f2f1;margin:0;font-size:14px;">
        10-year cardiovascular disease risk estimation using a 16-feature, non-leakage,
        CTGAN-balanced Stacking GenAI model (RF + XGB + CNN+GRU).
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 6. MAIN TABS
# =========================

tab_calc, tab_bie, tab_model, tab_faq = st.tabs(
    [
        "🧮 Risk Calculator",
        "🧠 Behavioral Impact Engine (BIE)",
        "🧬 Model & Data",
        "❓ FAQ & Notes"
    ]
)

# -------------------------
# TAB 1 – RISK CALCULATOR
# -------------------------
with tab_calc:
    st.markdown(
        """
        Use this tab to enter patient information and obtain a **10-year CVD risk estimate**
        from the Stacking Generative AI v3.0 model.
        """
    )

    df_input = build_input_df()

    with st.expander("View encoded feature vector (for experts)", expanded=False):
        st.dataframe(df_input.style.format(precision=2), use_container_width=True)

    run_btn = st.button("Run CVD Risk Prediction", type="primary")

    if run_btn:
        with st.spinner("Running Stacking GenAI model..."):
            final_prob, final_label, component_probs = stacking_predict_proba(
                df_input, threshold=threshold
            )

        # Store in session for BIE tab
        st.session_state["last_input_df"] = df_input
        st.session_state["last_prob"] = final_prob
        st.session_state["last_label"] = final_label
        st.session_state["last_components"] = component_probs
        st.session_state["last_threshold"] = threshold

        category, color = interpret_risk(final_prob)

        st.markdown("### Prediction Result")
        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            st.metric(
                label="Estimated 10-year CVD risk",
                value=f"{final_prob*100:.1f} %",
                help="Probability of a major cardiovascular event within 10 years."
            )
            st.markdown(f"**Risk category:** {color} **{category}**")

            st.markdown(
                f"**Model decision at threshold {threshold:.2f}:** "
                f"{'⚠️ At Risk (1)' if final_label == 1 else '✅ Not Flagged (0)'}"
            )

        with col_res2:
            st.markdown(
                """
                **Interpretation guide**  
                - <5%: Low risk  
                - 5–9%: Borderline  
                - 10–19%: Intermediate  
                - ≥20%: High risk  
                """
            )

        if show_components:
            st.markdown("### Component Model Contributions")
            comp_df = pd.DataFrame(
                {
                    "Model": list(component_probs.keys()),
                    "Predicted CVD risk (%)": [p * 100 for p in component_probs.values()]
                }
            )
            st.bar_chart(comp_df.set_index("Model"))

        st.info(
            "You can now click on the **Behavioral Impact Engine (BIE)** tab to see "
            "how changes in smoking or blood pressure might alter this patient’s risk."
        )
    else:
        st.info("Fill in the patient information and click **Run CVD Risk Prediction**.")

# -------------------------
# TAB 2 – BIE
# -------------------------
with tab_bie:
    st.subheader("Behavioral Impact Engine (BIE)")

    st.markdown(
        """
        The **Behavioral Impact Engine (BIE)** analyzes **which risk factor matters most**
        for a specific patient and simulates **“what-if” scenarios** (e.g., stopping smoking,
        lowering blood pressure) to show how risk can change.

        This goes beyond traditional calculators that only show a risk score—BIE helps answer:
        **“What should we do next?”**
        """
    )

    # Show static example from your description
    with st.expander("Illustrative Example (Fixed)", expanded=False):
        st.markdown(
            """
            **Example patient (from our internal experiments):**
            - Age 55, BP 130/80, Chol 190, Glucose 100, BMI 27  
            - **Risk with smoking** = 21.9%  
            - **Risk without smoking** = 13.8%  

            ➜ Smoking increases their risk by **+59% (relative)**  
            ➜ or **+8.1 percentage points (absolute)**  

            Based on this, the BIE suggests:
            - Smoking cessation advice  
            - Expected risk reduction  
            - Professional recommendations from cardiologists  
            - Scenario tables, e.g., “Lower BP by 10 mmHg → risk down to 21.0%”  
            """
        )

    st.markdown("---")

    # Dynamic BIE for last predicted patient
    if "last_input_df" not in st.session_state:
        st.warning(
            "No patient context available yet. Please go to **Risk Calculator** tab, "
            "enter a patient, and click **Run CVD Risk Prediction** first."
        )
    else:
        df_patient = st.session_state["last_input_df"]
        used_threshold = st.session_state.get("last_threshold", threshold)

        bie_results = bie_scenario(df_patient, threshold=used_threshold)
        base_prob = bie_results["baseline_prob"]

        st.markdown("### Patient-Specific BIE Summary")

        category, color = interpret_risk(base_prob)
        st.markdown(
            f"**Baseline 10-year risk:** {base_prob*100:.1f}%  ({color} {category})"
        )

        rows = []
        rows.append({
            "Scenario": "Baseline",
            "Description": "Current profile",
            "Predicted risk (%)": base_prob * 100,
            "Absolute change (points)": 0.0,
            "Relative change (%)": 0.0
        })

        # Smoking scenario, if available
        if bie_results["smoking"] is not None:
            s = bie_results["smoking"]
            rows.append({
                "Scenario": "No smoking",
                "Description": "Set cigarettes per day = 0",
                "Predicted risk (%)": s["prob_without_smoke"] * 100,
                "Absolute change (points)": s["abs_drop"] * 100,
                "Relative change (%)": s["rel_drop"] * 100
            })

        # BP scenario, if available
        if bie_results["bp"] is not None:
            b = bie_results["bp"]
            rows.append({
                "Scenario": "Lower SBP by 10 mmHg",
                "Description": f"SYSBP: {b['current_sysbp']:.0f} → {b['new_sysbp']:.0f}",
                "Predicted risk (%)": b["prob_lower_bp"] * 100,
                "Absolute change (points)": b["abs_drop"] * 100,
                "Relative change (%)": b["rel_drop"] * 100
            })

        bie_df = pd.DataFrame(rows)
        st.dataframe(
            bie_df.style.format(
                {
                    "Predicted risk (%)": "{:.1f}",
                    "Absolute change (points)": "{:.1f}",
                    "Relative change (%)": "{:.1f}"
                }
            ),
            use_container_width=True
        )

        # Narrative recommendations
        st.markdown("### Recommended Focus Areas")

        if bie_results["smoking"] is not None:
            s = bie_results["smoking"]
            st.markdown(
                f"""
                **1. Smoking Cessation**

                - Current estimated risk: **{s['prob_with_smoke']*100:.1f}%**  
                - If the patient **stops smoking** (CIGPDAY → 0): **{s['prob_without_smoke']*100:.1f}%**  
                - Absolute risk reduction: **{s['abs_drop']*100:.1f} points**  
                - Relative risk reduction: **{s['rel_drop']*100:.1f}%**

                **Clinical message (for discussion with the patient):**
                > Stopping smoking could meaningfully reduce your 10-year CVD risk.
                > Combining smoking cessation with blood pressure and lipid control
                > may further lower your overall risk.
                """
            )
        else:
            st.markdown(
                """
                **1. Smoking Cessation**

                - This patient is recorded as **non-smoking** (CIGPDAY = 0).  
                - Reinforce maintenance of **smoke-free lifestyle**.
                """
            )

        if bie_results["bp"] is not None:
            b = bie_results["bp"]
            st.markdown(
                f"""
                **2. Blood Pressure Optimization**

                - Current SBP: **{b['current_sysbp']:.0f} mmHg**  
                - Scenario: reduce SBP to **{b['new_sysbp']:.0f} mmHg**  
                - Risk today: **{b['prob_current']*100:.1f}%**  
                - Risk if SBP lowered: **{b['prob_lower_bp']*100:.1f}%**  
                - Absolute reduction: **{b['abs_drop']*100:.1f} points**  
                - Relative reduction: **{b['rel_drop']*100:.1f}%**

                **Clinical message:**
                > Achieving tighter blood pressure control (e.g., via lifestyle changes,
                > diet, exercise, and/or medications as per clinician judgment) is
                > expected to reduce long-term CVD risk.
                """
            )

        st.markdown(
            """
            ---
            ⚠️ **Important:** BIE outputs are **scenario simulations** from the model, not
            prescriptive treatment recommendations. Clinical decisions must always be
            made by qualified healthcare professionals using full clinical context.
            """
        )

# -------------------------
# TAB 3 – MODEL & DATA
# -------------------------
with tab_model:
    st.subheader("Model & Data Overview")

    st.markdown(
        """
        ### Data Source

        - Based on the **Framingham Heart Study** dataset (`frmgham2` variant).
        - Target: 10-year cardiovascular disease (**CVD**) outcome (`CVD` variable).
        - Features used (16, non-leakage):
          - **Demographics:** AGE, SEX, educ  
          - **Blood Pressure:** SYSBP, DIABP  
          - **Lipids:** TOTCHOL, HDLC, LDLC  
          - **Metabolic:** BMI, GLUCOSE, DIABETES  
          - **Treatment:** BPMEDS  
          - **Symptoms/History:** ANGINA, HYPERTEN  
          - **Lifestyle:** CIGPDAY  
          - **Cardiac:** HEARTRTE  

        **Note:** Features such as `STROKE`, `MI_FCHD`, or prior MI-related variables
        are intentionally excluded to avoid data leakage and to mimic a **true
        prospective risk prediction** scenario.
        """
    )

    st.markdown(
        """
        ### Generative AI (CTGAN)

        - Real dataset exhibits **class imbalance**: fewer CVD events than non-events.
        - We use **CTGAN** to generate **synthetic minority (CVD=1) samples**.
        - Training pipeline:
          1. Impute missing values (median for continuous, mode/median for binary/ordinal).
          2. Fit CTGAN on the full dataset (16 features + CVD label).
          3. Generate synthetic rows where `CVD = 1` until the classes are balanced.
          4. Combine real + synthetic data, shuffle, and use as the training set.
        """
    )

    st.markdown(
        """
        ### Stacking Generative AI Architecture

        1. **Base Learners:**
           - RandomForestClassifier (GenAI RF)  
           - XGBClassifier (GenAI XGB)  
           - CNN+GRU deep model (Keras/TensorFlow)  

        2. For each base model, we compute:
           - \\( p_{RF}(CVD=1 \\mid x) \\)  
           - \\( p_{XGB}(CVD=1 \\mid x) \\)  
           - \\( p_{CNN}(CVD=1 \\mid x) \\)  

        3. We then stack these into a meta-feature:
           - \\( z = [p_{RF}, p_{XGB}, p_{CNN}] \\)

        4. A **Logistic Regression** meta-learner is trained on \\(z\\) to produce
           the final probability:
           - \\( p_{stack}(CVD=1 \\mid x) \\)

        This design allows each base learner to capture different aspects of the data
        (tree-based patterns, boosting dynamics, temporal-style interactions via CNN/GRU),
        while the meta-learner learns an optimal combination.
        """
    )

    st.markdown(
        """
        ### Performance (v3.0, 16 Features, CTGAN-Balanced Test Set)

        - **AUC (ROC):** ~0.886  
        - **Accuracy:** ~0.81  
        - **Precision (CVD=1):** ~0.82  
        - **Recall (CVD=1):** ~0.68 (at threshold 0.50), ~0.70 (at threshold 0.40)  

        These results are in line with or above many traditional CVD risk engines,
        while being built on **non-leakage features** and a **balanced training signal**.
        """
    )

# -------------------------
# TAB 4 – FAQ & NOTES
# -------------------------
with tab_faq:
    st.subheader("FAQ & Notes")

    st.markdown(
        """
        **Q1. Is this an FDA-approved clinical decision tool?**  
        No. This is a **research and educational** prototype and is **not approved**
        for clinical use. It must not replace physician judgment.

        **Q2. How is this different from traditional risk calculators (e.g., Pooled Cohort Equations)?**  
        - Uses **ensemble ML/DL** (RF, XGB, CNN+GRU) instead of fixed equations.  
        - Uses **CTGAN-generated synthetic data** to balance CVD vs non-CVD events.  
        - Uses **stacking** to combine strengths of multiple models.  
        - Includes a **Behavioral Impact Engine (BIE)** to simulate risk reduction
          when modifying key risk factors (e.g., smoking, blood pressure).

        **Q3. What does the risk percentage mean?**  
        It is the model’s estimated **probability of a major cardiovascular event
        within 10 years** based on the input risk factors.

        **Q4. How should clinicians interpret the BIE scenarios?**  
        - BIE scenarios are **model-based simulations**, not treatment prescriptions.  
        - They are meant to facilitate discussions about **risk factor modification**.  
        - Any actual treatment decisions must consider patient comorbidities,
          preferences, guidelines, and physician judgment.

        **Q5. Why are some features excluded (e.g., STROKE, MI_FCHD)?**  
        Including post-event or strongly post-hoc variables can cause **data leakage**—
        the model “cheats” by seeing information that would not be available at the
        time of prediction. v3.0 focuses on **pre-event, prospectively available**
        features only.

        **Q6. Is the model calibrated?**  
        The model is primarily optimized for **discrimination (AUC)**.  
        Calibration analysis (e.g., calibration curves, Brier score) can be added
        as part of future work or in a clinical deployment pathway.
        """
    )

# =========================
# 7. FOOTER
# =========================

st.markdown(
    """
    <hr style="margin-top:32px;margin-bottom:8px;">
    <div style="text-align:center;font-size:12px;color:gray;">
      Stacking Generative AI CVD Risk Model v3.0 • 16 features • Research & Education Only<br>
      This application does not provide medical advice, diagnosis, or treatment.<br>
      © 2025 Howard Nguyen, PhD. For demonstration only — not for clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)
