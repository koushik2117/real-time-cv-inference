from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import time
from typing import Dict, List
import logging
import os
import sys

# Add the parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.models.inference import ModelInference
from backend.api.core.preprocessor import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(title="Real-Time CV Inference API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = ImagePreprocessor()
model_inference = ModelInference()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    try:
        model_inference.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Real-Time CV Inference API", "status": "operational"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "models_loaded": model_inference.models_loaded
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about loaded models"""
    return model_inference.get_model_info()

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "quantized"):
    """
    Single image prediction endpoint
    
    Args:
        file: Image file
        model_type: 'original' or 'quantized'
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        start_time = time.time()
        input_tensor = preprocessor.preprocess(image)
        preprocessing_time = (time.time() - start_time) * 1000
        
        # Run inference
        inference_start = time.time()
        predictions = model_inference.predict(input_tensor, model_type)
        inference_time = (time.time() - inference_start) * 1000
        
        # Log metrics
        logger.info(f"Prediction completed - Model: {model_type}, "
                   f"Preprocessing: {preprocessing_time:.2f}ms, "
                   f"Inference: {inference_time:.2f}ms")
        
        return {
            "predictions": predictions,
            "model_used": model_type,
            "inference_time_ms": inference_time,
            "preprocessing_time_ms": preprocessing_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(images: List[str], model_type: str = "quantized"):
    """
    Batch prediction endpoint
    """
    try:
        results = []
        total_start = time.time()
        
        for image_data in images:
            # Decode base64 image
            image = preprocessor.decode_base64(image_data)
            input_tensor = preprocessor.preprocess(image)
            
            # Run inference
            predictions = model_inference.predict(input_tensor, model_type)
            results.append(predictions)
        
        total_time = (time.time() - total_start) * 1000
        
        return {
            "predictions": results,
            "total_time_ms": total_time,
            "avg_time_per_image_ms": total_time / len(images)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# This is needed for running directly with python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)