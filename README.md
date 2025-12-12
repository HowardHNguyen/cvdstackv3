# 🫀 CVD Risk Prediction – Stacking Generative AI v3.0 (16 Features)

This repository contains the code and models for a **Stacking Generative AI** system
that predicts **10-year cardiovascular disease (CVD) risk** using data derived from
the **Framingham Heart Study**.

The final v3.0 model:

- Uses **16 clinically available features** (no data leakage)
- Balances the dataset with **CTGAN-based synthetic minority generation**
- Stacks three base learners:
  - Random Forest (GenAI RF)
  - XGBoost (GenAI XGB)
  - CNN+GRU (GenAI deep learner)
- Trains a **Logistic Regression meta-learner** on the base model probabilities

> ⚠️ **Disclaimer:**  
> This project is for **research and educational purposes only** and is **not** a
> substitute for professional medical advice, diagnosis, or treatment.

---

## 🌟 Model Summary (v3.0 – 16 Features)

- **Dataset:** Framingham Heart Study (cleaned, imputed, no leakage)
- **Features (16):**

  - Demographics: `AGE`, `SEX`, `educ`
  - Blood Pressure: `SYSBP`, `DIABP`
  - Lipids: `TOTCHOL`, `HDLC`, `LDLC`
  - Metabolic: `BMI`, `GLUCOSE`, `DIABETES`
  - Treatment: `BPMEDS`
  - Symptoms/History: `ANGINA`, `HYPERTEN`
  - Lifestyle: `CIGPDAY`
  - Cardiac: `HEARTRTE`

- **Target:** `CVD` (10-year cardiovascular disease event)

- **Final Stacking Performance (balanced test set):**

  - **AUC (ROC):** ~0.886
  - **Accuracy:** ~0.81
  - **Precision (CVD=1):** ~0.82 (at threshold 0.50)
  - **Recall (CVD=1):** ~0.68 (threshold 0.50), ~0.70 (threshold 0.40)

A threshold of **0.40** is recommended for a screening use-case, to modestly increase
recall for CVD while maintaining good precision.

---

## 🧠 Architecture

1. **Preprocessing**
   - Impute missing continuous values with median.
   - Impute missing binary/ordinal values with mode/median.
   - Standardize features using `StandardScaler`.

2. **Generative Balancing (CTGAN)**
   - Train `CTGAN` on all 16 features + `CVD`.
   - Generate synthetic minority samples (`CVD = 1`) to match the majority class.
   - Concatenate real + synthetic → **balanced training dataset**.

3. **Base Models**
   - **RandomForestClassifier**
   - **XGBClassifier**
   - **CNN+GRU** deep network (Keras / TensorFlow)

4. **Stacking Meta-Learner**
   - For each base model, compute `P(CVD=1 | x)` on the training and test sets.
   - Stack these probabilities into a 3D feature vector:
     - `[p_RF, p_XGB, p_CNN]`
   - Train a **Logistic Regression** meta-model on stacked probabilities.
   - Final prediction is `P(CVD=1)` from the meta-learner.

---

## 📂 Repository Structure

Example structure for v3.0:

```text
.
├── app.py                           # Streamlit app (v3.0, 16 features)
├── stacking_genai_v3_16.pkl         # Meta LR + RF + XGB + feature list
├── scaler_16.pkl                    # StandardScaler for 16 features
├── cnn_genai_v3_16.h5               # CNN+GRU Keras model
├── frmgham2.csv                     # Framingham dataset (not included here)
├── notebooks/
│   ├── 01_preprocessing_v3.ipynb    # Imputation, feature selection
│   ├── 02_ctgan_training_v3.ipynb   # CTGAN balancing
│   ├── 03_models_stacking_v3.ipynb  # RF/XGB/CNN + stacking
│   └── 04_evaluation_plots_v3.ipynb # ROC curves, feature importance, tables
└── README.md







### 📂 Data Upload
Supports drag-and-drop of new CSV files, allowing external researchers or clinicians to test their own patient data and instantly visualize predictions.

### 💬 Interpretability Layer
Includes SHAP explainers and calibration metrics to translate raw model outputs into transparent, actionable insights.

## 🧬 Research Significance

> This project demonstrates that **Stacking Generative AI** can achieve medical-grade predictive accuracy and enhanced fairness for underrepresented patient groups.

It contributes to the next generation of **AI Health Diagnostics** and aligns with the vision of **AICardioHealth Inc.**, a startup dedicated to advancing AI-driven cardiovascular prevention.

By integrating Generative AI with ensemble learning, **CVDStack** represents a paradigm shift in how healthcare systems can predict, explain, and prevent heart failure through data science.

## 🏁 Summary

**CVDStack = Generative AI + Stacked Learning + Explainable Healthcare.**  
It moves beyond prediction to personalized intervention — turning clinical data into life-saving insights.

> *Predict early. Explain clearly. Act precisely.*

## 🧾 License

**Copyright © 2025 Howard Nguyen**  
*(MaxAIS · AICardioHealth)*

Permission is hereby granted **with explicit written approval from Howard Nguyen** to use, copy, modify, and distribute this software and its associated documentation files.  
Unauthorized reproduction, redistribution, or modification without prior approval is strictly prohibited.

For commercial use, research collaboration, or licensing inquiries, please contact **info@howardnguyen.com**.

## 💬 Contact

📧 **Email:** info@howardnguyen.com  
🌐 **Website:** [www.maxais.com](https://www.maxais.com)  
🔗 **LinkedIn:** [Howard H. Nguyen](https://www.linkedin.com/in/howardhnguyen/)

## ⭐ Acknowledgments

Special thanks to **Harrisburg University** faculty and research mentors for academic guidance.  
This project also draws inspiration from global healthcare AI initiatives advancing cardiovascular prediction and early intervention.
