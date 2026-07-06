import os
import io
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
import keras_cv  # required for custom layer deserialization
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Force CPU (use if no GPU in production)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- Load model ----
MODEL_PATH = "phase2_checkpoint_best.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ---- Constants ----
IMAGE_SIZE = (224, 224)
CLS_THRESHOLD = 0.5

# ---- Preprocessing ----
def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---- Prediction ----
def predict_blood(image_bytes):
    inp = preprocess_image_bytes(image_bytes)
    cls_pred, reg_pred = model.predict(inp, verbose=0)
    return cls_pred[0][0], reg_pred[0]

# ---- Blur function ----
def blur_blood(image_bytes, cls_conf, reg_coords, blur_strength=(71,71)):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    xc, yc, bw, bh = reg_coords
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
    description="Hybrid model for blood detection and blurring.",
    version="1.0"
)

# ---- Test Webpage (HTML) ----
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Blood Detection Tester</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f5f5; }
        .container { max-width: 700px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        input[type="file"] { margin: 20px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; border-top: 1px solid #ddd; padding-top: 20px; }
        .bbox-info { background: #e9ecef; padding: 10px; border-radius: 5px; }
        img { max-width: 100%; border: 1px solid #ddd; margin-top: 10px; }
        .flex { display: flex; gap: 20px; flex-wrap: wrap; }
        .flex > div { flex: 1; min-width: 200px; }
        .badge { display: inline-block; padding: 5px 10px; border-radius: 20px; color: white; }
        .badge-blood { background: #dc3545; }
        .badge-noblood { background: #28a745; }
    </style>
</head>
<body>
<div class="container">
    <h1>🧪 Blood Detection Tester</h1>
    <p>Upload an image to detect blood regions and optionally blur them.</p>
    <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br>
        <button type="submit">🔍 Predict</button>
        <button type="button" id="blurBtn">🌀 Blur & Show</button>
    </form>
    <div class="result" id="result" style="display:none;">
        <h3>Result</h3>
        <div id="prediction"></div>
        <div id="bbox" class="bbox-info"></div>
        <div id="imageContainer"></div>
    </div>
</div>
<script>
    const form = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const predDiv = document.getElementById('prediction');
    const bboxDiv = document.getElementById('bbox');
    const imgContainer = document.getElementById('imageContainer');

    async function handleSubmit(action) {
        const fileInput = form.querySelector('input[type="file"]');
        if (!fileInput.files.length) {
            alert('Please select an image first.');
            return;
        }
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        let url = action;
        let isBlur = action === '/blur';

        try {
            const response = await fetch(url, { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json();
                alert('Error: ' + (err.detail || 'Unknown error'));
                return;
            }
            if (isBlur) {
                // Show blurred image
                const blob = await response.blob();
                const imgUrl = URL.createObjectURL(blob);
                imgContainer.innerHTML = `<img src="${imgUrl}" alt="Blurred result">`;
                resultDiv.style.display = 'block';
                predDiv.innerHTML = '';
                bboxDiv.innerHTML = '';
            } else {
                const data = await response.json();
                resultDiv.style.display = 'block';
                const isBlood = data.prediction === 'Blood';
                predDiv.innerHTML = `
                    <p><span class="badge ${isBlood ? 'badge-blood' : 'badge-noblood'}">${data.prediction}</span></p>
                    <p><strong>Confidence:</strong> ${data.confidence.toFixed(4)}</p>
                `;
                bboxDiv.innerHTML = `
                    <p><strong>Bounding Box (normalized):</strong></p>
                    <ul>
                        <li>x_center: ${data.bbox.x_center.toFixed(4)}</li>
                        <li>y_center: ${data.bbox.y_center.toFixed(4)}</li>
                        <li>width: ${data.bbox.width.toFixed(4)}</li>
                        <li>height: ${data.bbox.height.toFixed(4)}</li>
                    </ul>
                `;
                // Show the original image with a box overlay (optional)
                // For simplicity, we'll just display the uploaded image.
                const reader = new FileReader();
                reader.onload = function(e) {
                    imgContainer.innerHTML = `<img src="${e.target.result}" alt="Uploaded image">`;
                };
                reader.readAsDataURL(file);
            }
        } catch (err) {
            alert('Request failed: ' + err.message);
        }
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSubmit('/predict');
    });

    document.getElementById('blurBtn').addEventListener('click', () => {
        handleSubmit('/blur');
    });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE

# ---- API Endpoints ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        image_bytes = await file.read()
        cls_conf, reg_coords = predict_blood(image_bytes)
        if cls_conf < CLS_THRESHOLD:
            return Response(content=image_bytes, media_type="image/jpeg")
        blurred_bytes = blur_blood(image_bytes, cls_conf, reg_coords, blur_strength=(blur_strength, blur_strength))
        return Response(content=blurred_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)