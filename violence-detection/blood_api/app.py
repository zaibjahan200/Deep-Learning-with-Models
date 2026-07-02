import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import model_from_json

# ---------- 1. Load the model correctly ----------
def load_model_from_files():
    # Load architecture
    with open("config.json", "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    # Load weights
    model.load_weights("model.weights.h5")
    
    # Compile (required for inference even though we don't train)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model_from_files()
print("✅ Model loaded successfully!")

# ---------- 2. Rest of your FastAPI app ----------
app = FastAPI(
    title="Blood Detection API",
    description="Predicts whether an image contains blood.",
    version="1.0"
)

# ... (your preprocessing and prediction functions remain the same) ...
# ---------- 2. Define image preprocessing ----------
def preprocess_image(image_bytes):
    """
    Convert uploaded bytes to a preprocessed tensor ready for the model.
    This must match the preprocessing used during training.
    """
    # Read image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to the same size used during training (224x224)
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img) / 255.0
    
    # Add batch dimension (model expects shape (batch_size, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)

# ---------- 3. Create FastAPI instance ----------
app = FastAPI(
    title="Blood Detection API",
    description="Predicts whether an image contains blood.",
    version="1.0"
)

# ---------- 4. Health check endpoint ----------
@app.get("/")
async def root():
    return {"message": "Blood Detection API is running. Use /predict to upload an image."}

# ---------- 5. Prediction endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a prediction.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # Run inference
        prediction = model.predict(input_tensor, verbose=0)
        
        # Extract probability (sigmoid output)
        prob_blood = float(prediction[0][0])
        prob_no_blood = 1.0 - prob_blood
        
        # Class decision with threshold 0.5
        class_label = "Blood" if prob_blood >= 0.5 else "No Blood"
        
        return {
            "prediction": class_label,
            "confidence": {
                "blood": round(prob_blood, 4),
                "no_blood": round(prob_no_blood, 4)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up
        await file.close()

# ---------- 6. Optional: Endpoint for testing with a simple HTML form ----------
from fastapi.responses import HTMLResponse

@app.get("/test", response_class=HTMLResponse)
async def test_form():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Blood Detection Tester</title></head>
    <body>
        <h2>Upload an image to test</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# ---------- 7. Run the server (for development) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)