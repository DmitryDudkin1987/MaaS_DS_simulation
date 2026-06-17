import os
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from src.inference import load_model, predict


# ====== FastAPI приложение ======

app: FastAPI = FastAPI()


# ====== Pydantic модели ======

class OrderFeatures(BaseModel):
    """Модель входных данных для предсказания."""
    
    items_count: float
    distance_km: float
    is_express_delivery: float
    base_speed_kmh: float
    is_fast_food: float
    prep_time_avg: float
    precip_mm: float
    traffic_level: float
    order_hour: float
    vehicle_type_bike: float


class OrderPrediction(BaseModel):
    """Модель выходных данных с предсказанием."""
    
    predicted_delivery_time: float


# ====== Загрузка модели ======

logger.info("Loading model")

MODEL_PATH: str = os.path.join("models", "model.joblib")
MODEL = load_model(MODEL_PATH)

logger.info("Model loaded successfully")


# ====== Endpoints ======

@app.get("/")
def health_check() -> Dict[str, str]:
    """
    Проверка работоспособности сервиса.
    
    Returns:
        Dict[str, str]: Статус сервиса.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=OrderPrediction)
def get_prediction(features: OrderFeatures) -> OrderPrediction:
    """
    Эндпоинт для получения предсказания ML-модели.
    
    Args:
        features: Признаки заказа в формате OrderFeatures.
        
    Returns:
        OrderPrediction: Предсказанное время доставки.
        
    Raises:
        HTTPException: Ошибка при выполнении предсказания.
    """
    try:
        # Преобразуем входные данные в DataFrame
        data: pd.DataFrame = pd.DataFrame([features.model_dump()])

        # Получаем предсказание модели
        prediction: Union[List[float], Any] = predict(MODEL, data)
        predicted_value: float = float(prediction[0])

        logger.info(f"Raw model prediction: {prediction[0]}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    return OrderPrediction(predicted_delivery_time=predicted_value)
