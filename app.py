import os
import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model

# -------------------- LOAD FILES --------------------
model = load_model("churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

geo_encoder = encoders["geo"]
gender_encoder = encoders["gender"]

with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Churn Predictor", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #60a5fa;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5f5;
        margin-bottom: 30px;
    }
    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a customer will leave or stay</div>', unsafe_allow_html=True)


# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Customer Profile")

    credit_score = st.number_input("Credit Score", 300, 900, 600)
    geography = st.selectbox("Geography", geo_encoder.classes_)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Account Details")

    balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICT BUTTON --------------------
st.markdown("###")

if st.button("Predict Churn", use_container_width=True):

    # Encode
    geo_encoded = geo_encoder.transform([geography])[0]
    gender_encoded = gender_encoder.transform([gender])[0]

    input_dict = {
        "CreditScore": credit_score,
        "Geography": geo_encoded,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }

    input_data = [input_dict[feature] for feature in feature_order]
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0][0]

    # -------------------- RESULT --------------------
    st.markdown("## Prediction Result")

    if prediction > 0.5:
        st.error(f"⚠️ High Risk of Churn ({prediction:.2f})")
        st.progress(float(prediction))
    else:
        st.success(f"Customer Likely to Stay ({prediction:.2f})")
        st.progress(float(prediction))

    st.markdown("---")