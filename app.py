import os
import io
import pickle
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image # Used for initial format check, cv2 handles main processing

# --- Configuration ---
MODEL_PATH = "image_classification_model_cv2.h5" # Make sure this matches the saved model file
CLASS_DICT_PATH = "class_dict_cv2.pkl"      # Make sure this matches the saved class dictionary file
IMAGE_SIZE = (128, 128) # Must match the size used during training

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Image Classification API (OpenCV)",
    description="Upload an image for classification using a model trained with OpenCV preprocessing.",
    version="1.1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Model and Class Dictionary ---
model = None
class_dict = None
class_names = None

# --- Startup Event Handler ---
@app.on_event("startup")
async def load_resources():
    """Load the trained model and class dictionary when the FastAPI application starts."""
    global model, class_dict, class_names
    print("--- Loading ML Model and Class Dictionary ---")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(CLASS_DICT_PATH):
        print(f"ERROR: Class dictionary file not found at {CLASS_DICT_PATH}")
        raise RuntimeError(f"Class dictionary file not found at {CLASS_DICT_PATH}")

    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        with open(CLASS_DICT_PATH, "rb") as f:
            class_dict = pickle.load(f)
        class_names = [k for k, v in sorted(class_dict.items(), key=lambda item: item[1])]
        print(f"Class dictionary loaded successfully from {CLASS_DICT_PATH}")
        print(f"Classes: {class_names}")
        print("--- Resources Loaded ---")
    except Exception as e:
        print(f"Error loading resources: {e}")
        raise RuntimeError(f"Failed to load ML model or class dictionary: {e}")

# --- Image Preprocessing Function ---
def preprocess_image_with_cv2(image_bytes: bytes, target_size: tuple) -> np.ndarray:
    """Preprocesses the input image bytes using OpenCV for model prediction."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None: raise ValueError("Could not decode image using OpenCV.")
        img_resized = cv2.resize(img_cv, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")

# --- Prediction Endpoint ---
@app.post("/predict/", summary="Classify an uploaded image", response_description="The predicted class and confidence score")
async def predict_image(file: UploadFile = File(..., description="Image file to classify")):
    """Receives an image file, preprocesses it, predicts the class, and returns the result."""
    global model, class_names
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model or class dictionary not loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}.")

    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    try:
        image_bytes = await file.read()
        input_array = preprocess_image_with_cv2(image_bytes, target_size=IMAGE_SIZE)
        print("Making prediction...")
        predictions = model.predict(input_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_class_name = class_names[predicted_index]
        print(f"Prediction successful: Class='{predicted_class_name}', Confidence={confidence:.4f}")
        return JSONResponse(content={
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "filename": file.filename
        })
    except ValueError as ve:
         print(f"Preprocessing error for {file.filename}: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error processing image: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/", summary="API Root", description="Basic information about the API")
async def read_root():
    return {"message": "Welcome to the Image Classification API using OpenCV!",
            "model_status": "Loaded" if model is not None else "Not Loaded",
            "classes": class_names if class_names is not None else "Not Loaded"}

# --- Main Block to Run Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)