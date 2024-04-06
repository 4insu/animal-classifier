import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import animal_classes, load_trained_model, predict_image_class_from_bytes

# Paths and model loading
base_path = os.getcwd()
model_folder = 'models'
model_filename = 'animal_classification.h5'

@st.cache_resource
def load_model():
    return load_trained_model(base_path, model_folder, model_filename)

model = load_model()

# Streamlit app
st.title('Animal Classifier')
st.write("Upload an image and let's predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    prediction = predict_image_class_from_bytes(model, uploaded_file)
    st.markdown(f'<p style="font-size:30px; color:#ffff00; text-align:center;"><b>This is a {prediction}!</b></p>', 
                unsafe_allow_html = True)
    st.image(uploaded_file, caption = 'Uploaded Image', use_column_width = True)