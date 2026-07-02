import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf

# Force CPU (optional, for Docker)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------------ 1. FIND AND LOAD MODEL ------------------
# Automatically pick the .keras file in the current directory
keras_files = [f for f in os.listdir(".") if f.endswith(".keras")]
if not keras_files:
    raise FileNotFoundError("No .keras model file found in the current directory.")
MODEL_PATH = keras_files[0]  # use the first one
print(f"📂 Using model file: {MODEL_PATH}")

# Load the model (this will print any TensorFlow warnings)
print("⏳ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------ 2. PREPROCESSING FUNCTION ------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))          # match training size
    img_array = np.array(img) / 255.0     # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array.astype(np.float32)

# ------------------ 3. FASTAPI APP ------------------
app = FastAPI(
    title="Blood Detection API",
    description="Predicts if an image contains blood",
    version="1.0"
)

@app.get("/")
async def root():
    return {"message": "Blood Detection API is running. Use /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"File must be an image. Received: {file.content_type}")
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        pred = model.predict(input_tensor, verbose=0)[0][0]
        prob_blood = float(pred)
        prob_no_blood = 1.0 - prob_blood
        class_label = "Blood" if prob_blood >= 0.5 else "No Blood"
        return {
            "prediction": class_label,
            "confidence": {
                "blood": round(prob_blood, 4),
                "no_blood": round(prob_no_blood, 4)
            }
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {str(e)}")
    finally:
        await file.close()

@app.get("/test", response_class=HTMLResponse)
async def test_form():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Blood Tester</title></head>
    <body>
        <h2>Upload an image</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# ------------------ 4. START THE SERVER (only when run directly) ------------------
if __name__ == "__main__":
    import uvicorn
    # reload=False is recommended for production
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)