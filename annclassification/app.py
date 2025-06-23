import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load trained model and transformers
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔍 Customer Churn Prediction")
st.markdown("Use this tool to predict if a customer is likely to churn based on their profile information.")

st.markdown("---")
st.subheader("📝 Input Customer Details")

# Organize input into columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850)

with col2:
    balance = st.number_input('💰 Account Balance', min_value=0.0)
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0)
    tenure = st.slider('📅 Tenure (Years)', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('🟢 Is Active Member', [0, 1])

# ---- Preprocess Inputs ----
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

# ---- Predict ----
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.markdown("---")
st.subheader("📊 Prediction Results")

st.write(f"**Churn Probability:** `{prediction_proba:.2%}`")

st.progress(prediction_proba)

if prediction_proba > 0.5:
    st.error('⚠️ The customer is **likely to churn**.')
else:
    st.success('✅ The customer is **not likely to churn**.')

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
