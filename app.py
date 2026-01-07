
import streamlit as st
from ml import get_predicted_value, helper, symptoms_dict

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Disease Prediction System")

st.write("Select your symptoms from the list below:")

user_symptoms = st.multiselect(
    "Select Symptoms:",
    options=list(symptoms_dict.keys())
)

if st.button("Predict Disease"):
    if not user_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        disease = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(disease)

        st.subheader("🧾 Prediction Result")
        st.markdown(f"**Predicted Disease:** `{disease}`")

        st.markdown("### 📝 Description")
        st.info(desc)

        st.markdown("### 🛡️ Precautions")
        for p in pre:
            st.markdown(f"- {p}")

        st.markdown("### 💊 Medications")
        for m in med:
            st.markdown(f"- {m}")

        st.markdown("### 🥗 Diet")
        for d in die:
            st.markdown(f"- {d}")

        st.markdown("### 🏃 Workout")
        for w in wrkout:
            st.markdown(f"- {w}")

