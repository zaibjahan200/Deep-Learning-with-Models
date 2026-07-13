import os
import cv2
import numpy as np
import tensorflow as tf

# ============================================
# CONFIG
# ============================================

MODEL_PATH = os.environ.get("MODEL_PATH", "app/best_model_phase2_25.keras")
IMAGE_SIZE = (384, 384)
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.8
MIN_MASK_AREA_RATIO = 0.005
MIN_COMPONENT_AREA = 100
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


        if not has_blood:
            return confidence, False, image.astype(np.uint8)


        mask = mask.astype(np.float32)

        h, w = image.shape[:2]

        mask_resized = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )


        binary_mask = (
            mask_resized > MASK_THRESHOLD
        ).astype(np.uint8)


        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )


        cleaned_mask = np.zeros_like(binary_mask)


        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area >= MIN_COMPONENT_AREA:
                cleaned_mask[labels == i] = 1


        mask_ratio = (
            np.sum(cleaned_mask) /
            cleaned_mask.size
        )


        if mask_ratio < MIN_MASK_AREA_RATIO:

            return confidence, True, image.astype(np.uint8)


        blurred = cv2.GaussianBlur(
            image,
            BLUR_KERNEL,
            BLUR_SIGMA
        )


        mask_3ch = np.stack(
            [cleaned_mask] * 3,
            axis=-1
        )


        result = np.where(
            mask_3ch == 1,
            blurred,
            image
        )


        return confidence, True, result.astype(np.uint8)