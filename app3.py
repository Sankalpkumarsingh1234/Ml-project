import streamlit as st
import joblib
import numpy as np

model = joblib.load("health_risk.pkl")
le_gender = joblib.load("le_gender.pkl")
le_smoking = joblib.load("le_smoking.pkl")
le_alcohol = joblib.load("le_alcohol.pkl")
le_physical = joblib.load("le_physical.pkl")
le_family = joblib.load("le_family.pkl")
le_stress = joblib.load("le_stress.pkl")
le_target = joblib.load("target_label.pkl")

st.title(" Smart Health Risk Prediction System")

gender = st.selectbox("Gender", le_gender.classes_)
smoking = st.selectbox("Smoking", le_smoking.classes_)
alcohol = st.selectbox("Alcohol Consumption", le_alcohol.classes_)
physical = st.selectbox("Physical Activity", le_physical.classes_)
family = st.selectbox("Family History", le_family.classes_)
stress = st.selectbox("Stress Level", le_stress.classes_)
age = st.number_input("Age", min_value=0, max_value=120, value=25)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=400, value=180)
glucose = st.number_input("Glucose", min_value=0, max_value=400, value=100)
heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=70)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=22.5)

if st.button("Predict Health Risk"):
    try:
        features = np.array([
            le_gender.transform([gender])[0],
            le_smoking.transform([smoking])[0],
            le_alcohol.transform([alcohol])[0],
            le_physical.transform([physical])[0],
            le_family.transform([family])[0],
            le_stress.transform([stress])[0],
            age,
            blood_pressure,
            cholesterol,
            glucose,
            heart_rate,
            bmi
        ]).reshape(1, -1)

        pred = model.predict(features)
        risk_label = le_target.inverse_transform(pred)[0]
        st.success(f" Predicted Health Risk Level: **{risk_label}**")

    except Exception as e:
        st.error(f" Error during prediction: {e}")
