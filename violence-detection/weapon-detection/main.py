import base64
import os
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

# ============================================================
# Config — mirrors the training notebook (Model Configs / Model Architecture)
# ============================================================
CLASS_NAMES = ['Pistol', 'Rifle', 'Heavy Weapon', 'Gun', 'Shotgun', 'Knife']
NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 for background (label 0)

MODEL_PATH = os.environ.get("MODEL_PATH", "weapon_detector_best.pth")
DEVICE = torch.device("cpu")

NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
torch.set_num_threads(NUM_THREADS)

# Same preprocessing as training/detect_and_blur — the model's own internal
# transform is a no-op (image_mean=(0,0,0), image_std=(1,1,1)), so normalization
# has to happen here, not inside the model.
TRANSFORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def get_model(num_classes):
    # pretrained=False: this is inference-only, loading our own fine-tuned
    # checkpoint below — no need to also download ImageNet backbone weights
    # that immediately get overwritten by load_state_dict.
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        'resnet50', pretrained=False
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
    )
    return model


model = None  # populated at startup via lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    m = get_model(NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(state_dict)
    m.to(DEVICE)
    m.eval()
    model = m
    print(f"Model loaded from {MODEL_PATH} on {DEVICE}, threads={NUM_THREADS}")
    yield


app = FastAPI(title="Weapon Detector API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weapon Detector</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  .panel { border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin-top: 16px; }
  label { display: block; margin-bottom: 6px; font-weight: 600; }
  input[type="file"] { margin-bottom: 12px; }
  input[type="number"] { width: 80px; margin-left: 8px; }
  button { padding: 8px 16px; border: none; border-radius: 6px; background: #1a1a1a; color: #fff; cursor: pointer; }
  button:disabled { background: #888; cursor: not-allowed; }
  #status { margin-top: 10px; font-style: italic; }
  #result-img { max-width: 100%; margin-top: 16px; border: 1px solid #ccc; border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #ddd; font-size: 0.9rem; }
</style>
</head>
<body>
  <h1>Weapon Detector — upload &amp; blur</h1>
  <div class="panel">
    <form id="upload-form">
      <label for="file">Image file</label>
      <input type="file" id="file" name="file" accept="image/*" required>
      <label for="threshold">Confidence threshold</label>
      <input type="number" id="threshold" name="threshold" value="0.5" min="0" max="1" step="0.05">
      <div style="margin-top: 14px;">
        <button type="submit" id="submit-btn">Detect &amp; blur</button>
      </div>
    </form>
    <div id="status"></div>
  </div>
  <div class="panel" id="result-panel" style="display:none;">
    <img id="result-img" alt="Blurred result">
    <table id="detections-table">
      <thead><tr><th>Class</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead>
      <tbody id="detections-body"></tbody>
    </table>
    <div id="detections-empty" style="display:none;">No detections above threshold.</div>
  </div>
<script>
const form = document.getElementById('upload-form');
const statusEl = document.getElementById('status');
const submitBtn = document.getElementById('submit-btn');
const resultPanel = document.getElementById('result-panel');
const resultImg = document.getElementById('result-img');
const detectionsBody = document.getElementById('detections-body');
const detectionsEmpty = document.getElementById('detections-empty');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('file');
  const threshold = document.getElementById('threshold').value;
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  submitBtn.disabled = true;
  statusEl.textContent = 'Running detection... (CPU inference can take a few seconds)';
  resultPanel.style.display = 'none';

  try {
    const url = `/detect-and-blur?conf_threshold=${encodeURIComponent(threshold)}`;
    const resp = await fetch(url, { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${resp.status})`);
    }
    const data = await resp.json();
    resultImg.src = `data:image/jpeg;base64,${data.image_base64}`;
    detectionsBody.innerHTML = '';
    if (data.detections.length === 0) {
      detectionsEmpty.style.display = 'block';
    } else {
      detectionsEmpty.style.display = 'none';
      for (const d of data.detections) {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${d.class_name}</td><td>${(d.confidence * 100).toFixed(1)}%</td><td>${d.box.join(', ')}</td>`;
        detectionsBody.appendChild(row);
      }
    }
    resultPanel.style.display = 'block';
    statusEl.textContent = `Done — ${data.detections.length} detection(s) above threshold.`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

def blur_detections(img_bgr, boxes, scores, labels, threshold):
    """Blur every detected region above `threshold`. Blur-only — no box/label
    drawn on the image; class + confidence are returned separately in `detections`."""
    height, width = img_bgr.shape[:2]
    detections = []

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        img_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

        # labels are 1..6 for weapon classes, 0 is background (see WeaponDataset)
        class_idx = int(label) - 1
        class_name = CLASS_NAMES[class_idx] if 0 <= class_idx < len(CLASS_NAMES) else "Unknown"

        detections.append({
            "class_name": class_name,
            "confidence": float(score),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })

    return img_bgr, detections


@app.post("/detect-and-blur")
async def detect_and_blur_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded yet"})

    contents = await file.read()
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image"})

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transformed = TRANSFORM(image=img_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(img_tensor)

    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    blurred_bgr, detections = blur_detections(img_bgr, boxes, scores, labels, conf_threshold)

    success, encoded = cv2.imencode('.jpg', blurred_bgr)
    if not success:
        return JSONResponse(status_code=500, content={"error": "Failed to encode output image"})

    image_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

    return {
        "detections": detections,
        "image_base64": image_base64,
    }
