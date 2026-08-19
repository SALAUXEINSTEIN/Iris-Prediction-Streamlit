import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Get the directory containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Locate classifier.pkl in the same directory as app.py
MODEL_PATH = BASE_DIR / "classifier.pkl"

# Load the trained model
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

st.title("Iris Species Classifier")

st.write("Enter the flower measurements to classify the species.")

sepal_length = st.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    step=0.1
)

sepal_width = st.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=5.0,
    step=0.1
)

petal_length = st.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    step=0.1
)

petal_width = st.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    step=0.1
)

if st.button("Predict"):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(features)

    st.write(f"Predicted Iris Species: {prediction[0]}")