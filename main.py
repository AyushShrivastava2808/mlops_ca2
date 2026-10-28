
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load model from the project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival using a trained ML model.",
    version="1.0.0",
)


class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(data)[0])

    return {
        "prediction": prediction,
        "survival": (
            "Survived" if prediction == 1
            else "Did not survive"
        )
    }