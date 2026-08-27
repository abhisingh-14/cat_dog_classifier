import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model_best.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (64, 64)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload a picture of a cat or a dog and the model will classify it!")

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying..."):
        arr = preprocess_image(image)
        pred = model.predict(arr, verbose=0)[0][0]
        
        dog_prob = float(pred)
        cat_prob = 1.0 - dog_prob
        
        label = "Dog 🐶" if dog_prob >= 0.5 else "Cat 🐱"
        confidence = dog_prob if dog_prob >= 0.5 else cat_prob
        
        st.success(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
        
        st.write("### Probabilities")
        st.write(f"- **Dog**: {dog_prob * 100:.2f}%")
        st.write(f"- **Cat**: {cat_prob * 100:.2f}%")
