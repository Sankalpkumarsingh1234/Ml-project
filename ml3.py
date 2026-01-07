import streamlit as st
import joblib as jb

model = jb.load("chatbot.pkl")
le_x = jb.load( "label_encoder_x.pkl") # Save and load your LabelEncoder for User_Input
le_y = jb.load("label_encoder_y.pkl") # Save and load your LabelEncoder for Response

st.title("Chatbot Assistant")
user_input = st.text_input("Enter your text:")

if st.button("Get Response"):
    if user_input:
        try:
            x_encoded = le_x.transform([user_input]).reshape(-1, 1)
            pred = model.predict(x_encoded)
            response = le_y.inverse_transform(pred)[0]
            st.success(f"Chatbot Response: {response}")
        except ValueError:
            st.error("Input not recognized. Please enter a phrase from the training data.")
    else:
        st.warning("Please enter your text.")
