import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

@st.cache_resource
def load_my_model():
    return load_model("fashion_MNIST_app_with_model/mnist.keras")

model = load_my_model()

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess(image):
    # 1. Convert to grayscale
    img = image.convert("L")
    
    # 2. Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 3. Convert to numpy array and normalize to [0, 1]
    img = np.array(img) / 255.0
    
    # 4. 🚀 CRITICAL FIX: Invert colors if background is lighter than the object
    # Fashion-MNIST expects Black background (0) and White object (1).
    # If the average pixel is > 0.5, the image is mostly white (light background), so we invert!
    if np.mean(img) > 0.5:
        img = 1.0 - img  # Flip black <-> white
    
    # 5. Flatten to (784,) and reshape to (1, 784) for the model
    img = img.reshape(1, 784)
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
    st.info(f"Confidence: **{confidence:.2%}**")
