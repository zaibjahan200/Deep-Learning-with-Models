import base64
import os
from contextlib import asynccontextmanager

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
