import streamlit as st
import joblib as jb

model = jb.load("drug_interaction_model.pkl")
le_d1 = jb.load("le_drug1.pkl")        
le_d2 = jb.load("le_drug2.pkl")       
le_target = jb.load("le_target.pkl")   
st.title("Drug Interaction Predictor")
drug1 = st.text_input("Enter Drug 1:")
drug2 = st.text_input("Enter Drug 2:")

if st.button("Predict Interaction"):
    if drug1 and drug2:
        try:
           
            if drug1 not in le_d1.classes_:
                st.error(f"Drug 1 not recognized: {drug1}")
            elif drug2 not in le_d2.classes_:
                st.error(f"Drug 2 not recognized: {drug2}")
            else:
                d1_enc = le_d1.transform([drug1])[0]
                d2_enc = le_d2.transform([drug2])[0]
                pred = model.predict([[d1_enc, d2_enc]])
                interaction = le_target.inverse_transform(pred)[0]
                st.success(f"Predicted Interaction: {interaction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter both drug names.")