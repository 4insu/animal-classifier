import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths and model loading
base_path = 'C:\\Users\\SUPRIYO AIN\\Desktop\\animal-classifier'
model_folder = 'models'
model_filename = 'animal_classification.h5'
model_path = os.path.join(base_path, model_folder, model_filename)
model = load_model(model_path)

animal_classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer",
    "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly",
    "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala",
    "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus",
    "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes",
    "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
    "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]

# Function to make predictions
def predict_animal_class(image):
    img = cv2.imdecode(np.asarray(bytearray(image.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, (224, 224)), axis=0)
    predictions = np.argmax(model.predict(img), axis=1)
    return animal_classes[predictions[0]]

# Streamlit app
st.title('Animal Classifier')
st.write("Upload an image and let's predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    prediction = predict_animal_class(uploaded_file)
    st.markdown(f'<p style="font-size:30px; color:#ffff00; text-align:center;"><b>This is a {prediction}!</b></p>', 
                unsafe_allow_html = True)
    st.image(uploaded_file, caption = 'Uploaded Image', use_column_width = True)

