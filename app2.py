import streamlit as st
import joblib as jb
st.title("Blood Donation Eligibility Prediction")
model=jb.load('blood_donation_model.pkl')
age=st.number_input("Enter Age",min_value=18,max_value=65)
weight=st.number_input("Enter Weight in kg",min_value=45)
hemoglobin=st.number_input("Enter Hemoglobin in g/dL",min_value=12.5,max_value=17.5)
last_donation=st.number_input("Enter Last Donation in months",min_value=0)
disease_history=st.selectbox("Do you have any disease history?",("No","Yes"))
disease_history=1 if disease_history=="Yes" else 0
if st.button("Predict Eligibility"):
    result=model.predict([[age,weight,hemoglobin,last_donation,disease_history]])
    if result[0]==1:
        st.success("You are eligible for blood donation.")
    else:
        st.error("You are not eligible for blood donation.")
st.write("Note: Eligibility criteria may vary based on local regulations and health conditions. Please consult with a healthcare professional for personalized advice.")    