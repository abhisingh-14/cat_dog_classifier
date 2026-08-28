import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱", layout="centered")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #ff4b4b;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #fafafa;
        font-size: 1.2rem;
        margin-bottom: 40px;
        opacity: 0.8;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background-color: #262730;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
        border: 1px solid #333;
    }
    
    .pred-label {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        margin: 10px 0;
    }
    
    .confidence {
        font-size: 1.1rem;
        color: #aaa;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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

# Header
st.markdown("<h1 class='main-header'>🐱 Cat vs Dog 🐶</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an image and let our AI classify it!</p>", unsafe_allow_html=True)

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout
st.markdown("---")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### 🔍 Analysis")
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            arr = preprocess_image(image)
            pred = model.predict(arr, verbose=0)[0][0]
            
            dog_prob = float(pred)
            cat_prob = 1.0 - dog_prob
            
            is_dog = dog_prob >= 0.5
            label = "Dog 🐶" if is_dog else "Cat 🐱"
            confidence = dog_prob if is_dog else cat_prob
            
            # Custom prediction box
            st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem; color: #aaa;">Result</div>
                    <div class="pred-label">{label}</div>
                    <div class="confidence">{confidence * 100:.2f}% Confidence</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Probability Breakdown")
            
            # Progress bars for probabilities
            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.write("🐶 **Dog**")
            with col_b:
                st.progress(dog_prob)
                
            col_c, col_d = st.columns([1, 4])
            with col_c:
                st.write("🐱 **Cat**")
            with col_d:
                st.progress(cat_prob)
    else:
        st.info("👈 Please upload an image on the left to see the prediction results.")
        
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem;'>Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
