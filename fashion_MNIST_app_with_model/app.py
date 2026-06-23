import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("fashion_MNIST_app_with_model/mnist.keras")

classes = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

def preprocess(image):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 784)
    return img

st.title("👕 Fashion Classifier")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = preprocess(image)
    pred = model.predict(img)

    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")