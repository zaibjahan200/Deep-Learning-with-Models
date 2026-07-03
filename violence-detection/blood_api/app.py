import os
import io
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
import keras_cv  # <-- necessary for custom layer deserialization
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
import uvicorn

# Force CPU if no GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- Load model ----
MODEL_PATH = "hybrid_blood_detector.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")
# ... rest of app.py unchanged ...

# ---- Constants ----
IMAGE_SIZE = (224, 224)
CLS_THRESHOLD = 0.5

# ---- Preprocessing function ----
def preprocess_image_bytes(image_bytes):
    """
    Convert uploaded bytes to model input.
    Returns normalized numpy array of shape (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---- Prediction function ----
def predict_blood(image_bytes):
    """
    Run inference and return (cls_pred, reg_pred)
    """
    inp = preprocess_image_bytes(image_bytes)
    cls_pred, reg_pred = model.predict(inp, verbose=0)
    # cls_pred: (1,1), reg_pred: (1,4)
    return cls_pred[0][0], reg_pred[0]

# ---- Blur function ----
def blur_blood(image_bytes, cls_conf, reg_coords, blur_strength=(51,51)):
    """
    Blur the detected region on the original image.
    Returns blurred image as bytes (JPEG).
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    xc, yc, bw, bh = reg_coords
    # Convert normalized to pixel coordinates
    xc_px = int(xc * w)
    yc_px = int(yc * h)
    w_px = int(bw * w)
    h_px = int(bh * h)
    x1 = max(0, xc_px - w_px // 2)
    y1 = max(0, yc_px - h_px // 2)
    x2 = min(w, xc_px + w_px // 2)
    y2 = min(h, yc_px + h_px // 2)
    roi = img[y1:y2, x1:x2]
    if roi.size > 0:
        blurred_roi = cv2.GaussianBlur(roi, blur_strength, 0)
        img[y1:y2, x1:x2] = blurred_roi
    _, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

# ---- FastAPI app ----
app = FastAPI(
    title="Blood Detection API",
    description="Hybrid model (YOLO backbone + custom head) for blood detection and blurring.",
    version="1.0"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body>
    <h2>Blood Detection API</h2>
    <p>Use <code>/predict</code> to get classification + bbox, or <code>/blur</code> to get blurred image.</p>
    <p>Swagger docs at <a href="/docs">/docs</a></p>
    </body></html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict blood presence and bounding box.
    Returns JSON with:
      - prediction: "Blood" or "No Blood"
      - confidence: float
      - bbox: {x_center, y_center, width, height} (normalized)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        image_bytes = await file.read()
        cls_conf, reg_coords = predict_blood(image_bytes)
        is_blood = cls_conf >= CLS_THRESHOLD
        return {
            "prediction": "Blood" if is_blood else "No Blood",
            "confidence": float(cls_conf),
            "bbox": {
                "x_center": float(reg_coords[0]),
                "y_center": float(reg_coords[1]),
                "width": float(reg_coords[2]),
                "height": float(reg_coords[3])
            }
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/blur")
async def blur(file: UploadFile = File(...), blur_strength: int = 51):
    """
    Return the image with the detected blood region blurred.
    If no blood detected, returns the original image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        image_bytes = await file.read()
        cls_conf, reg_coords = predict_blood(image_bytes)
        if cls_conf < CLS_THRESHOLD:
            # No blood – return original image
            return Response(content=image_bytes, media_type="image/jpeg")
        # Blur the region
        blurred_bytes = blur_blood(image_bytes, cls_conf, reg_coords, blur_strength=(blur_strength, blur_strength))
        return Response(content=blurred_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)