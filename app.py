import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
bundle = joblib.load("mlr_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# Page settings
st.set_page_config(page_title="House Price Estimator", page_icon=":house:", layout="centered")

# Inject Material Icons and style
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
    <style>
    .icon-title {
        display: flex;
        align-items: center;
        font-size: 28px;
        font-weight: bold; 
        margin-bottom: 1rem;
    }
    .material-symbols-outlined {
        font-variation-settings:
        'FILL' 0,
        'wght' 400,
        'GRAD' 0,
        'opsz' 48;
        margin-right: 10px;
        font-size: 36px;
        color: #4a90e2;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="icon-title"><span class="material-symbols-outlined">real_estate_agent</span> House Price Estimator</div>', unsafe_allow_html=True)
st.caption("Enter details to estimate **SalePrice** using a trained Multiple Linear Regression model.")

# Input fields with tooltips
col1, col2 = st.columns(2)

with col1:
    overallqual = st.slider(
        "Overall Quality", 1, 10, 5,
        help="Rates overall material and finish quality of the house (1 = Very Poor, 10 = Excellent)"
    )
    garagecars = st.slider(
        "Garage Capacity (in cars)", 0, 5, 2,
        help="Number of cars that can fit in the garage"
    )

with col2:
    grlivarea = st.number_input(
        "Above Ground Living Area (sq ft)", min_value=100, max_value=5000, value=1500,
        help="Total finished living area above ground level"
    )
    totalbsmtsf = st.number_input(
        "Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800,
        help="Total square footage of the basement (finished + unfinished)"
    )

# Prediction
if st.button("🔍 Predict Sale Price"):
    input_features = np.array([[overallqual, grlivarea, garagecars, totalbsmtsf]])
    input_scaled = scaler.transform(input_features)
    log_pred = model.predict(input_scaled)
    final_pred = np.exp(log_pred[0])  # reverse log

    st.success(f"💰 **Estimated Sale Price:** ${final_pred:,.0f}")
