import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Load model
model = tf.keras.models.load_model("custom_cnn_best.h5")

# Class labels (order used during training)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# UI Header
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image and get tumor classification with confidence scores.")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

# Run prediction when file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Display results
    st.markdown(f"### 🧠 Predicted Tumor Type: `{predicted_label.upper()}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    st.markdown("#### 🔬 All Class Confidence Scores:")
    for i, score in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {score * 100:.2f}%")
