import base64
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.model import BloodDetector

# ============================================
# APP SETUP
# ============================================

app = FastAPI(title="Blood Detection API")
templates = Jinja2Templates(directory="app/templates")

# Load model once at startup
detector = BloodDetector()


# ============================================
# ROUTES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify image for blood presence.
    Returns confidence score and has_blood flag.
    """
    image = await _read_image(file)

    confidence, has_blood = detector.classify(image)

    return JSONResponse({
        "confidence": round(float(confidence), 4),
        "has_blood": bool(has_blood),
        "label": "Blood Detected" if has_blood else "No Blood"
    })


@app.post("/blur")
async def blur(file: UploadFile = File(...)):
    """
    Segment and blur blood regions in the image.
    Returns blurred image as base64 + confidence score.
    """
    image = await _read_image(file)

    confidence, has_blood, blurred_image = detector.blur(image)

    # Encode blurred image to base64
    _, buffer = cv2.imencode(".jpg", blurred_image)
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "confidence": round(float(confidence), 4),
        "has_blood": bool(has_blood),
        "label": "Blood Detected" if has_blood else "No Blood",
        "blurred_image": b64_image
    })


# ============================================
# HELPERS
# ============================================

async def _read_image(file: UploadFile) -> np.ndarray:
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read image. Make sure the file is a valid image."
        )

    return image
