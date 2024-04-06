# import streamlit as st
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from src.utils import animal_classes, load_trained_model, predict_image_class_from_bytes

# # Paths and model loading
# base_path = os.getcwd()
# model_folder = 'models'
# model_filename = 'animal_classification.h5'

# model = load_trained_model(base_path, model_folder, model_filename)

# # Streamlit app
# st.title('Animal Classifier')
# st.write("Upload an image and let's predict its class!")

# uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     prediction = predict_image_class_from_bytes(model, uploaded_file)
#     st.markdown(f'<p style="font-size:30px; color:#ffff00; text-align:center;"><b>This is a {prediction}!</b></p>', 
#                 unsafe_allow_html = True)
#     st.image(uploaded_file, caption = 'Uploaded Image', use_column_width = True)


import os
import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from src.utils import load_trained_model, predict_image_class_from_bytes

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set logging level to ERROR
tf.get_logger().setLevel(logging.ERROR)

app = Flask(__name__, static_url_path = '/static')

# Paths and model loading
base_path = os.getcwd()
model_folder = 'models'
model_filename = 'animal_classification.h5'

model = load_trained_model(base_path, model_folder, model_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        prediction = predict_image_class_from_bytes(model, uploaded_file)
        return jsonify({"prediction": prediction})
    else:
        return jsonify({"error": "No file uploaded"})

if __name__ == '__main__':
    app.run(debug = True)