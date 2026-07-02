import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf

# Force CPU (optional – if you want to avoid CUDA issues inside Docker)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Print all .keras files in the current directory
print("📂 Files in /app:")
print(os.listdir("."))
keras_files = [f for f in os.listdir(".") if f.endswith(".keras")]
if keras_files:
    print(f"🔍 Found .keras files: {keras_files}")
    MODEL_PATH = keras_files[0]   # pick the first one (or you can hardcode)
else:
    raise FileNotFoundError("No .keras file found in /app")
# ------------------ LOAD MODEL ------------------
MODEL_PATH = "Blood_model.keras"   # your .keras file
print(f"⏳ Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------ PREPROCESSING ------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))          # match training input shape
    img_array = np.array(img) / 255.0     # normalize
    img_array = np.expand_dims(img_array, axis=0)   # add batch dim
    return img_array.astype(np.float32)

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Blood Detection API", version="1.0")

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
        pred = model.predict(input_tensor, verbose=0)[0][0]   # probability
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
        raise HTTPException(500, f"Error: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)