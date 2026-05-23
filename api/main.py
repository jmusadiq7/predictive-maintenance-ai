from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.alerts import AlertManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "Models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
ALERT_LOG_PATH = PROJECT_ROOT / "logs" / "alerts.log"

ROLLING_WINDOW = 10
TOP_SENSORS = ["Ps30", "T50", "Phi", "W32", "P30"]


class PredictionRequest(BaseModel):
    engine_id: int = Field(..., ge=1)
    sensor_readings: dict[str, float]


class AlertResponse(BaseModel):
    triggered: bool
    message: str | None = None


class PredictionResponse(BaseModel):
    engine_id: int
    prediction: str
    confidence: float
    alert: AlertResponse


class HealthResponse(BaseModel):
    status: str


class ModelState:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_columns: list[str] = []
        self.engine_history: dict[int, list[dict[str, float]]] = {}
        self.lock = Lock()


state = ModelState()
alert_manager = AlertManager(ALERT_LOG_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.model = joblib.load(MODEL_PATH)
        state.feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required model artifact not found: {exc.filename}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc

    yield

    state.model = None
    state.feature_columns = []
    state.engine_history.clear()


app = FastAPI(title="Jet Engine Predictive Maintenance API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if state.model is None or not state.feature_columns:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    missing_features = [
        feature
        for feature in state.feature_columns
        if not feature.endswith("_rolling_mean")
        and feature not in payload.sensor_readings
    ]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required sensor readings: {missing_features}",
        )

    missing_rolling_sensors = [
        sensor for sensor in TOP_SENSORS if sensor not in payload.sensor_readings
    ]
    if missing_rolling_sensors:
        raise HTTPException(
            status_code=400,
            detail=f"Missing sensors required for rolling means: {missing_rolling_sensors}",
        )

    current_reading = {
        key: float(value) for key, value in payload.sensor_readings.items()
    }

    with state.lock:
        history = state.engine_history.setdefault(payload.engine_id, [])
        history.append(current_reading)
        state.engine_history[payload.engine_id] = history[-ROLLING_WINDOW:]

        if len(state.engine_history[payload.engine_id]) < ROLLING_WINDOW:
            readings_needed = ROLLING_WINDOW - len(state.engine_history[payload.engine_id])
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Need {ROLLING_WINDOW} readings for engine_id "
                    f"{payload.engine_id} before prediction. "
                    f"Send {readings_needed} more reading(s)."
                ),
            )

        prediction_row = dict(current_reading)
        for sensor in TOP_SENSORS:
            rolling_values = [
                reading[sensor] for reading in state.engine_history[payload.engine_id]
            ]
            prediction_row[f"{sensor}_rolling_mean"] = sum(rolling_values) / ROLLING_WINDOW

    try:
        input_df = pd.DataFrame([prediction_row], columns=state.feature_columns)
        predicted_class = int(state.model.predict(input_df)[0])

        if hasattr(state.model, "predict_proba"):
            probabilities = state.model.predict_proba(input_df)[0]
            class_index = list(state.model.classes_).index(predicted_class)
            confidence = float(probabilities[class_index])
        else:
            confidence = 1.0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    prediction_label = "DANGER" if predicted_class == 1 else "SAFE"
    confidence = round(confidence, 4)

    if prediction_label == "DANGER":
        try:
            alert_manager.log_alert(payload.engine_id, confidence)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Alert logging failed: {exc}") from exc

        alert = AlertResponse(
            triggered=True,
            message=(
                f"Engine {payload.engine_id} at risk. "
                f"Confidence: {confidence * 100:.2f}%"
            ),
        )
    else:
        alert = AlertResponse(triggered=False)

    return PredictionResponse(
        engine_id=payload.engine_id,
        prediction=prediction_label,
        confidence=confidence,
        alert=alert,
    )
