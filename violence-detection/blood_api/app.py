import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------------------------------------------
# 1. DISABLE GPU (Force CPU inference in Docker)
#    This prevents CUDA errors and warnings in the container.
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------------------------------------------------------
# 2. DEFINE THE MODEL ARCHITECTURE (EXACT MATCH TO TRAINING)
#    Your 4-block architecture: 64 -> 128 -> 256 -> 512 filters
# ------------------------------------------------------------
def build_model(
    num_filters=64,
    kernel_size=3,
    input_shape=(224, 224, 3),
    pool_size=(2, 2)
):
    model = Sequential([
        # Block 1: 64 filters -> 112x112
        Conv2D(num_filters, (kernel_size, kernel_size), padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size),

        # Block 2: 128 filters -> 56x56
        Conv2D(num_filters * 2, (kernel_size, kernel_size), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size),

        # Block 3: 256 filters -> 28x28
        Conv2D(num_filters * 4, (kernel_size, kernel_size), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size),

        # Block 4: 512 filters -> 14x14
        Conv2D(num_filters * 8, (kernel_size, kernel_size), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    return model

# ------------------------------------------------------------
# 3. LOAD THE MODEL
#    - Build architecture from code (exact match to training)
#    - Load weights from the .h5 file
# ------------------------------------------------------------
def load_model_from_files():
    print("⏳ Building model architecture (4 blocks: 64→128→256→512 filters)...")
    model = build_model()
    
    print("⏳ Loading weights from model.weights.h5...")
    model.load_weights("model.weights.h5")
    
    print("⏳ Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    
    print("✅ Model loaded successfully!")
    return model

# Load the model when the app starts
model = load_model_from_files()

# ------------------------------------------------------------
# 4. PREPROCESSING FUNCTION
#    Resizes and normalizes the image exactly like training
# ------------------------------------------------------------
def preprocess_image(image_bytes):
    """
    Convert uploaded bytes to a preprocessed tensor ready for the model.
    """
    # Read image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 224x224 (must match training input shape)
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img) / 255.0
    
    # Add batch dimension -> shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)

# ------------------------------------------------------------
# 5. FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(
    title="Blood Detection API",
    description="Predicts whether an image contains blood using a 4-block CNN.",
    version="1.0"
)

# ------------------------------------------------------------
# 6. ENDPOINTS
# ------------------------------------------------------------

# Root endpoint - health check
@app.get("/")
async def root():
    return {"message": "Blood Detection API is running. Use /predict to upload an image."}

# Prediction endpoint
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
        await file.close()

# Test HTML form endpoint
@app.get("/test", response_class=HTMLResponse)
async def test_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Blood Detection Tester</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            input[type="file"] { margin: 10px 0; padding: 10px; border: 1px solid #ccc; width: 100%; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            #result { margin-top: 20px; padding: 20px; border: 1px solid #28a745; border-radius: 5px; display: none; background: #f8f9fa; }
            .blood { color: #dc3545; font-weight: bold; }
            .no-blood { color: #28a745; font-weight: bold; }
            .confidence { font-family: monospace; }
        </style>
    </head>
    <body>
        <h2>🧪 Blood Detection API Tester</h2>
        <p>Upload an image to check if it contains blood.</p>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">🔍 Predict</button>
        </form>
        <div id="result">
            <h3>📊 Result:</h3>
            <p><strong>Prediction:</strong> <span id="prediction"></span></p>
            <p><strong>Blood Confidence:</strong> <span id="blood_conf" class="confidence"></span></p>
            <p><strong>No Blood Confidence:</strong> <span id="no_blood_conf" class="confidence"></span></p>
        </div>
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                const predSpan = document.getElementById('prediction');
                predSpan.textContent = data.prediction;
                predSpan.className = data.prediction === 'Blood' ? 'blood' : 'no-blood';
                document.getElementById('blood_conf').textContent = data.confidence.blood;
                document.getElementById('no_blood_conf').textContent = data.confidence.no_blood;
                document.getElementById('result').style.display = 'block';
            };
        </script>
    </body>
    </html>
    """

# ------------------------------------------------------------
# 7. RUN THE SERVER (for local development)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)