import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

MODEL_PATH = r'C:\Users\preet\DataVisualization_UCONN\BigDataProject\BirdSpecies-Resnet101V2-model.h5'
model = tf.keras.models.load_model(MODEL_PATH)


class_names = {
    0: 'ABBOTTS BABBLER',
    1: 'ABBOTTS BOOBY',
    2: 'ABYSSINIAN GROUND HORNBILL',
    3: 'AFRICAN CROWNED CRANE',
    4: 'AFRICAN EMERALD CUCKOO',
    5: 'AFRICAN FIREFINCH',
    6: 'AFRICAN OYSTER CATCHER',
    7: 'AFRICAN PIED HORNBILL',
    8: 'AFRICAN PYGMY GOOSE',
    9: 'ALBATROSS',
    10: 'ALBERTS TOWHEE',
    11: 'ALEXANDRINE PARAKEET',
    12: 'ALPINE CHOUGH',
    13: 'ALTAMIRA YELLOWTHROAT',
    14: 'AMERICAN AVOCET',
    15: 'AMERICAN BITTERN',
    16: 'AMERICAN COOT',
    17: 'AMERICAN FLAMINGO',
    18: 'AMERICAN GOLDFINCH',
    19: 'AMERICAN KESTREL'
}


def predict(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_probability = np.max(predictions[0])
    return class_names[class_idx], class_probability


st.title('Bird Species Prediction Web App')
st.write('Welcome to Feather Finder, your gateway to the avian world! Just upload an image and let our deep learning model reveal the bird\'s species. It\'s Big Data Analytics making nature more accessible one click at a time!')
st.write("Ready to discover the species of birds around you? Upload an image to start the identification!")

uploaded_file = st.file_uploader("Upload an image of a bird", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, probability = predict(uploaded_file)
    probability_percent = probability * 100
    st.write(f"Prediction: {label}")
    st.write(f"Probability: {probability_percent:.2f}%")

#streamlit run WebApp-Birds.py