from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import os

app = FastAPI(title="Forest Cover Prediction API", description="Predict forest cover type using machine learning")

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Load the trained model
try:
    model = joblib.load('forest_cover_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    # Serve the frontend.html file
    frontend_path = os.path.join(os.getcwd(), "frontend.html")
    return FileResponse(frontend_path)

@app.post("/predict")
def predict_cover_type(input_data: PredictionInput):
    print(f"Received features length: {len(input_data.features)}")
    print(f"Received features: {input_data.features[:10]}...")  # print first 10 for brevity

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(input_data.features) != 54:
        raise HTTPException(status_code=400, detail="Input must have exactly 54 features")

    try:
        # Convert input to numpy array and reshape for prediction
        features = np.array(input_data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        # Get the probability of the predicted class
        confidence = float(prediction_proba[prediction - 1])  # prediction is 1-7, array is 0-6

        return {
            "prediction": int(prediction),
            "confidence": round(confidence * 100, 2),
            "cover_type_name": get_cover_type_name(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def get_cover_type_name(cover_type: int) -> str:
    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    return cover_types.get(cover_type, "Unknown")

@app.get("/model-info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": model.n_estimators,
        "features_required": 54,
        "cover_types": {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }
    }
