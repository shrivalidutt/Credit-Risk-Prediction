#pip install streamlit

import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load pipeline
model = joblib.load("credit_risk_model.pkl")

st.title("📊 Credit Risk Prediction App")
st.write("This app predicts the probability of loan default and explains the decision with SHAP.")

# --- User inputs ---
loan_amount = st.number_input("Loan Amount", min_value=100, value=2000, step=100)
duration = st.number_input("Duration (months)", min_value=1, value=24, step=1)
checking_status = st.selectbox("Checking Account Status", ["A11","A12","A13","A14"])
credit_history = st.selectbox("Credit History", ["A30","A31","A32","A33","A34"])
savings_status = st.selectbox("Savings Status", ["A61","A62","A63","A64","A65"])
employment = st.selectbox("Employment", ["A71","A72","A73","A74","A75"])

# --- Build input row ---
input_dict = {
    "duration": duration,
    "credit_amount": loan_amount,
    "checking_status": checking_status,
    "credit_history": credit_history,
    "savings_status": savings_status,
    "employment": employment
}
X_new = pd.DataFrame([input_dict])

# --- Fill missing columns with defaults ---
expected_cols = model.named_steps['prep'].feature_names_in_

default_values = {
    "age": 35,
    "housing": "A151",
    "purpose": "A40",
    "property": "A121",
    "residence_since": 2,
    "job": "A171",
    "existing_credits": 1,
    "num_dependents": 1,
    "other_debtors": "A101",
    "other_installment": "A141",
    "telephone": "A191",
    "personal_status": "A92",
    "foreign_worker": "A201"
}

for col in expected_cols:
    if col not in X_new.columns:
        X_new[col] = default_values.get(col, 0)  # fallback = 0

# --- Prediction ---
proba = model.predict_proba(X_new)[0, 1]
st.metric("Predicted Default Probability", f"{proba:.2%}")

if proba < 0.2:
    st.success("✅ Approve Loan")
elif proba < 0.5:
    st.warning("⚠️ Manual Review Needed")
else:
    st.error("❌ Reject Loan")

# --- SHAP Explainability ---
st.subheader("🔎 Feature Importance (SHAP)")

prep = model.named_steps['prep']

# Original numeric + categorical columns
num_cols = prep.transformers_[0][2]
cat_cols = prep.transformers_[1][2]

# One-hot expanded names for categorical features
ohe = prep.named_transformers_['cat'].named_steps['ohe']
cat_feature_names = ohe.get_feature_names_out(cat_cols)

# Final list of feature names
feature_names = np.concatenate([num_cols, cat_feature_names])

# Transform the new input
X_new_trans = prep.transform(X_new)

# SHAP with feature names
explainer = shap.Explainer(model.named_steps['clf'], X_new_trans, feature_names=feature_names)
shap_values = explainer(X_new_trans)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

