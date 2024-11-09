import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load model
model = joblib.load(filename='training/model/model.pkl')

st.title('ML Prediciton App')

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stFileUploader {
            border: 2px dashed #4CAF50;
        }
        h1, h2 {
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

# Page Title and Header Image
st.image("res/header-image.png", use_column_width=True)  # Replace with actual header image if available
st.title("ML Prediction App")

# File uploader with styled border
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.write(data.head())  # Display a preview of uploaded data

    if st.button("Predict"):
        predictions = model.predict(data)
        data['Prediction'] = predictions

        # Display predictions with color-coded cells
        st.write("### Prediction Results")
        st.dataframe(
            data.style.applymap(lambda x: 'background-color: #d2f5d1' if x == "good" else 'background-color: #f5d1d1'))

        # Summary Visualization
        st.write("### Prediction Summary")
        fig, ax = plt.subplots()
        data['Prediction'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#4CAF50', '#FF6347'], ax=ax)
        st.pyplot(fig)

        # Download button
        st.download_button("Download predictions", data.to_csv(index=False), "predictions.csv")
