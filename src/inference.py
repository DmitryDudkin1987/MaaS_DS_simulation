import joblib
from typing import List, Any, Union

import pandas as pd
from sklearn.base import BaseEstimator


def load_model(model_path: str) -> BaseEstimator:
    """
    Загрузка модели из файла.
    
    Args:
        model_path: Путь к файлу модели.
        
    Returns:
        BaseEstimator: Загруженная модель.
        
    Raises:
        FileNotFoundError: Если файл модели не найден.
    """
    try:
        model: BaseEstimator = joblib.load(model_path)
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model not found at {model_path}") from e


def predict(model: BaseEstimator, df: pd.DataFrame) -> List[int]:
    """
    Предсказание с использованием модели.
    
    Args:
        model: Обученная модель.
        df: DataFrame с признаками.
        
    Returns:
        List[int]: Список предсказаний.
    """
    predictions: Union[List[int], Any] = model.predict(df)
    
    # Преобразуем в список int для гарантии типа
    if hasattr(predictions, 'tolist'):
        return predictions.tolist()
    return list(predictions)