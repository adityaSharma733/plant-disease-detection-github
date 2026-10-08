import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to classify it using a trained deep learning model.")

MODEL_PATH = Path("models/plant_disease_model.keras")

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if MODEL_PATH.exists():
        model = load_model()
        image_array = np.array(image.resize((224, 224))) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        class_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100

        st.success(f"Predicted class: {class_index}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning(
            "The trained model is not available yet. "
            "Run src/train.py after adding the dataset."
        )
else:
    st.info("Upload a JPG, JPEG, or PNG leaf image to begin.")
