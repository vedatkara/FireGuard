from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import joblib
import os
from model_loader import load_model, load_feature_names
import logging
import datetime
import openpyxl
from openpyxl import Workbook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and feature names
model = load_model()
feature_names = load_feature_names()

# Load scaler and label encoder
scaler = joblib.load('Joblib/scaler.joblib')
label_encoder = joblib.load('Joblib/label_encoder.joblib')

# Print scaler information
print("\nScaler Information:")
print(f"Scaler type: {type(scaler)}")
print(f"Scaler features: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'No feature names'}")
print(f"Number of scaler features: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'Unknown'}")

class PredictionRequest(BaseModel):
    longitude: float
    latitude: float
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: str
    longitude: float
    latitude: float

@app.get("/")
async def root():
    return {"message": "FireGuard API is running"}

@app.get("/features")
async def get_features():
    try:
        return {"features": feature_names}
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request with data: {request}")
        
        # Convert input features to array in the correct order
        input_features = []
        for feature in feature_names:
            if feature in request.features:
                input_features.append(request.features[feature])
            else:
                logger.error(f"Missing feature: {feature}")
                raise HTTPException(status_code=400, detail=f"Missing feature: {feature}")
        
        input_features = np.array(input_features).reshape(1, -1)
        logger.info(f"Input features shape: {input_features.shape}")
        
        # Scale the features
        try:
            scaled_features = scaler.transform(input_features)
            logger.info(f"Scaled features: {scaled_features}")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scaling features: {str(e)}")
        
        # Make prediction
        try:
            raw_prediction = model.predict(scaled_features)[0]
            logger.info(f"Raw prediction: {raw_prediction}")
            
            # Transform prediction using label encoder
            transformed_prediction = label_encoder.inverse_transform([raw_prediction])[0]
            logger.info(f"Transformed prediction: {transformed_prediction}")

            # --- Excel logging ---
            excel_path = "predictions.xlsx"
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Prepare row data
            row = [now, request.latitude, request.longitude]
            for feature in feature_names:
                row.append(request.features.get(feature, ''))
            row.append(transformed_prediction)
            # Create or append to Excel file
            try:
                try:
                    wb = openpyxl.load_workbook(excel_path)
                    ws = wb.active
                except FileNotFoundError:
                    wb = Workbook()
                    ws = wb.active
                    headers = ["timestamp", "latitude", "longitude"] + feature_names + ["prediction"]
                    ws.append(headers)
                ws.append(row)
                wb.save(excel_path)
            except Exception as e:
                logger.error(f"Error writing to Excel: {str(e)}")
            # --- End Excel logging ---

            return {
                "prediction": transformed_prediction,
                "longitude": request.longitude,
                "latitude": request.latitude
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 