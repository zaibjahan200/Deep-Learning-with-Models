import os
import cv2
import numpy as np
import tensorflow as tf

# ============================================
# CONFIG
# ============================================

MODEL_PATH = os.environ.get("MODEL_PATH", "app/best_model_phase2_10.keras")
IMAGE_SIZE = (384, 384)
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
BLUR_KERNEL = (51, 51)
BLUR_SIGMA = 20


# ============================================
# BLOOD DETECTOR
# ============================================

class BloodDetector:

    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded.")

    def preprocess(self, image: np.ndarray) -> tf.Tensor:
        """BGR (OpenCV) → RGB → resize → normalize → batch"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, IMAGE_SIZE)
        image_normalized = image_resized.astype(np.float32) / 255.0
        return tf.expand_dims(image_normalized, axis=0)

    def predict_raw(self, image: np.ndarray) -> tuple:
        """Run model, return (confidence, mask)"""
        tensor = self.preprocess(image)
        outputs = self.model(tensor, training=False)

        confidence = float(outputs["classification"].numpy()[0][0])
        mask = outputs["segmentation"].numpy()[0, :, :, 0]

        return confidence, mask

    def classify(self, image: np.ndarray) -> tuple:
        """Returns (confidence, has_blood)"""
        confidence, _ = self.predict_raw(image)
        has_blood = confidence >= CLASSIFICATION_THRESHOLD
        return confidence, has_blood

    def blur(self, image: np.ndarray) -> tuple:
        confidence, mask = self.predict_raw(image)
        has_blood = confidence >= CLASSIFICATION_THRESHOLD

        # Cast to float32 — OpenCV resize doesn't support float16
        mask = mask.astype(np.float32)

        # Resize mask back to original image size
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binary mask
        binary_mask = (mask_resized > MASK_THRESHOLD).astype(np.uint8)

        # Apply Gaussian blur to entire image
        blurred = cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMA)

        # Composite: use blurred only where mask is 1
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        result = np.where(mask_3ch == 1, blurred, image)

        return confidence, has_blood, result.astype(np.uint8)