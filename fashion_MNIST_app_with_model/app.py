import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Cache the model loading
@st.cache_resource
def load_my_model():
    return load_model("fashion_mnist_87.keras")

model = load_my_model()

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess(image):
    img = image.convert("L")  # Grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = img.reshape(1, 784)  # Flatten
    return img

st.title("👕 Fashion Classifier")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    processed_img = preprocess(image)
    pred = model.predict(processed_img)

    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence:.2%}**") # Display as percentage