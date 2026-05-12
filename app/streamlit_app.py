import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

POP_MODEL  = os.path.join(BASE_DIR, "models", "xgb_model.joblib")
POP_ENC    = os.path.join(BASE_DIR, "models", "label_encoder.joblib")
PROG_MODEL = os.path.join(BASE_DIR, "models", "progression_model.joblib")
PROG_ENC   = os.path.join(BASE_DIR, "models", "progression_encoder.joblib")

# Load models
@st.cache_resource
def load_models():
    pop_model = joblib.load(POP_MODEL)
    pop_enc   = joblib.load(POP_ENC)
    prog_model = joblib.load(PROG_MODEL)
    prog_enc   = joblib.load(PROG_ENC)
    return pop_model, pop_enc, prog_model, prog_enc

# Load reference data for percentile calc
@st.cache_data
def load_reference_data():
    ref_path = os.path.join(BASE_DIR, "data", "reference_totals.csv")
    return pd.read_csv(ref_path)

# Page Config
st.set_page_config(
    page_title="Powerlifting Predictor",
    page_icon="🏋️‍♂️",
    layout="centered"
)

st.title("🏋️‍♂️ Powerlifting Performance Predictor")
st.caption(
    "Built on 900k+ competition results from "
    "[OpenPowerlifting](https://www.openpowerlifting.org)"
)

pop_model, pop_enc, prog_model, prog_enc = load_models()
ref_df = load_reference_data()

tab1, tab2 = st.tabs(["📊 How do I compare?", "📈 What's my next total?"])

# TAB 1: Population Model
with tab1:
    st.header("Compare yourself to the field")
    st.write(
        "Enter your details to see a predicted total and how you'd rank "
        "against historical competition results."
    )

    col1, col2 = st.columns(2)
    with col1:
        sex_1 = st.selectbox("Sex at Birth", ["M", "F"], key = "sex_1")
        age_1 = st.number_input("Age", min_value=14, max_value=80, value=25, key = "age_1")
        bodyweight = st.number_input("Bodyweight (kg)", min_value=30.0, max_value=300.0, value=80.0, step = 0.5, key = "bodyweight")

    with col2:
        actual_total = st.number_input("Your actual Total (kg)", min_value=0.0, max_value=2000.0, value=0.0, step = 2.5,
                                       help = "Enter your best total to see where you rank. Leave at 0 to skip.", key = "actual_total")
        
    if st.button("Predict", key="btn_1"):
        sex_encoded = pop_enc.transform([sex_1])[0]
        pred_total = pop_model.predict(
            pd.DataFrame([[sex_encoded, age_1, bodyweight]],
                          columns=["Sex", "Age", "BodyweightKg"])
        )[0]

        # Percentile vs same sex
        same_sex = ref_df[ref_df['Sex'] == sex_1]['TotalKg']
        percentile = (same_sex < pred_total).mean() * 100

        st.divider()
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted Total (kg)", f"{pred_total:.1f} kg")
        col_b.metric("Percentile Rank (same sex)", f"{percentile:.1f}th")

        if actual_total > 0:
            actual_pct = (same_sex < actual_total).mean() * 100
            diff = actual_total - pred_total
            st.metric(
                "Your actual total",
                f"{actual_total:.1f} kg",
                delta = f"{diff:+.1f} kg vs predicted"
            )
            st.caption(f"Your actual total is at the **{actual_pct:.1f}th percentile**")

        # Distribution plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(same_sex, bins=80, color="#4a9eff", alpha=0.6, edgecolor='none')
        ax.axvline(pred_total, color="#ff4a4a", linewidth=2, label=f"Predicted: {pred_total:.1f}kg")
        if actual_total > 0:
            ax.axvline(actual_total, color="#4aff4a", linewidth=2, linestyle='--', label=f"Actual: {actual_total:.0f}kg")
        ax.set_xlabel("Total (kg)")
        ax.set_ylabel("Count")
        ax.set_title(f"Total distribution - {'Male' if sex_1=='M' else 'Female'} Raw SBD")
        ax.legend()
        st.pyplot(fig)

        st.caption(
            "⚠️ Predicted total is based on sex, age, and bodyweight alone. "
            "Average error: ±61kg. Individual performance varies significantly."
        )
    
# TAB 2: Progression Model
with tab2:
    st.header("Predict your next total")
    st.write(
        "Enter your last meet total and your next meet details "
        "to get a prediction for your next total."
    )

    st.subheader("Your last meet")
    col3, col4 = st.columns(2)
    with col3:
        sex_2      = st.selectbox("Sex", ["M", "F"], key="sex_2")
        prev_total = st.number_input("Total (kg)", 0.0, 2000.0, 400.0, step=2.5)
        prev_gl    = st.number_input("GL Points", 0.0, 200.0, 70.0, step=0.1)
    with col4:
        prev_bw    = st.number_input("Bodyweight (kg)", 40.0, 300.0, 80.0,
                                     step=0.5, key="bw_prev")
        age_prev   = st.number_input("Age at last meet", 14, 80, 25,
                                     key="age_prev")
        meet_num   = st.number_input("Career meet number (e.g. 3rd meet = 3)",
                                     1, 100, 1)
        
    st.subheader("Your next meet")
    col5, col6 = st.columns(2)
    with col5:
        next_bw    = st.number_input("Planned bodyweight (kg)", 40.0, 300.0,
                                     80.0, step=0.5, key="bw_next")
    with col6:
        months_gap = st.number_input("Months until next meet", 1, 60, 6)
        age_next   = st.number_input("Age at next meet", 14, 80, 25,
                                     key="age_next")
        
    if st.button("Predict", key="btn_2"):
        bw_change = next_bw - prev_bw
        sex_encoded = prog_enc.transform([sex_2])[0]

        features = pd.DataFrame([[
            sex_encoded, prev_total, prev_gl,
            prev_bw, next_bw, bw_change,
            age_next, months_gap, meet_num
        ]], columns=[
                'Sex', 'prev_total', 'prev_gl',
                'prev_bodyweight', 'next_bodyweight', 'bw_change',
                'age_at_next', 'months_between', 'meet_number'
            ])

        pred_next = prog_model.predict(features)[0]

        # Percentile
        same_sex = ref_df[ref_df['Sex'] == sex_2]['TotalKg']
        pct = (same_sex < pred_next).mean() * 100
        diff = pred_next - prev_total

        st.divider()
        col_c, col_d, col_e = st.columns(3)
        col_c.metric("Predicted Next Total (kg)", f"{pred_next:.1f} kg")
        col_d.metric("Change from last meet", f"{diff:+.1f} kg")
        col_e.metric("Percentile (same sex)", f"{pct:.0f}th")

        # Distribution plot
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.hist(same_sex, bins=80, color="#4a9eff", alpha=0.6, edgecolor='none')
        ax2.axvline(prev_total, color="#ffaa00", linewidth=2,
                    linestyle='--', label=f"Last total: {prev_total:.0f}kg")
        ax2.axvline(pred_next, color="#ff4a4a", linewidth=2,
                    label=f"Predicted next: {pred_next:.0f}kg")
        ax2.set_xlabel("Total (kg)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Where you'll sit — {'Male' if sex_2 == 'M' else 'Female'} Raw SBD")
        ax2.legend()
        st.pyplot(fig2)

        st.caption(
            "⚠️ Prediction based on competition history patterns. "
            "Average error: ±20kg."
        )

# Footer
st.divider()
st.caption("Data from the [OpenPowerlifting project](https://www.openpowerlifting.org). "
    "Models trained on Raw SBD lifters aged 14–80. "
    "This page uses data from the OpenPowerlifting project, "
    "https://www.openpowerlifting.org."
    "Made by Lawrence Kenworthy"
)