
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool


def load_data() -> pd.DataFrame:
    # URL к базе
    local_path = "/home/yc-user/mlops/airflow_mlflow/dags/MaaS/db/delivery_final_homework.db"
            
    # Подключение через SQLAlchemy
    engine = create_engine(f"sqlite:///{local_path}")


    df = pd.read_sql(
    """ SELECT *, CAST(STRFTIME('%H', orders.order_datetime) AS INTEGER) AS order_hour, CAST (STRFTIME('%w', orders.order_datetime) AS INTEGER) AS weekday_number  FROM orders
left join customers on orders.customer_id=customers.customer_id
left join couriers on orders.courier_id = couriers.courier_id
left join restaurants on orders.restaurant_id = restaurants.restaurant_id
left join weather on restaurants.restaurant_city=weather.city and DATE(orders.order_datetime)=weather.date
left join traffic on restaurants.restaurant_city=traffic.city and DATE(orders.order_datetime)=traffic.date
""",
    engine,
)
    # удаляем столбцы с дубликатами
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # привести к datetime
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["open_date"] = pd.to_datetime(df["open_date"])
    df["date"] = pd.to_datetime(df["date"])

        # заполним пропуски temperature
    df = df.sort_values(["restaurant_city", "date"])
    df["temperature"] = df.groupby("restaurant_city")["temperature"].transform(
        lambda x: x.interpolate(method="linear")
    )
    # заполним пропуски precip_mm
    df["precip_mm"] = df.groupby("restaurant_city")["precip_mm"].transform(
        lambda x: x.interpolate(method="linear")
    )
    # Затем заполняем начало группы следующим значением (bfill)
    df["precip_mm"] = df.groupby("restaurant_city")["precip_mm"].transform(
    lambda x: x.bfill()
)

    
    df["wind_speed"] = df.groupby("restaurant_city")["wind_speed"].transform(
        lambda x: x.interpolate(method="linear")
    )
    df["traffic_level"] = df.groupby("restaurant_city")["traffic_level"].transform(
        lambda x: x.interpolate(method="linear")
    )
    

    cat_cols = df.select_dtypes(include=["object"]).columns
    
    for catigory in cat_cols:
        # One-Hot Encoding (создаёт отдельные столбцы)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[[catigory]])
        # encoder.categories_
        # Создаём DataFrame с результатами
        columns = [f"{catigory}_{item}" for item in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded, columns=columns, index=df.index)
        # Добавляем к исходному DataFrame
        df = pd.concat([df, encoded_df], axis=1)


    df = df[['items_count', 'distance_km', 'is_express_delivery', 'base_speed_kmh', 'is_fast_food', 'prep_time_avg', 'precip_mm', 'traffic_level', 'order_hour', 'vehicle_type_bike', 'delivery_time_minutes','order_datetime']]
    return df


def split_data(df: pd.DataFrame) -> Tuple[Pool, Pool, Pool]:


    target_col = "delivery_time_minutes"
    cols_to_drop = ['delivery_time_minutes','order_datetime']

    # Разделяем по квантилям времени (60% / 20% / 20%)
    q60 = df["order_datetime"].quantile(0.60)
    q80 = df["order_datetime"].quantile(0.80)
    
    train_df = df[df["order_datetime"] <= q60].copy()
    val_df = df[(df["order_datetime"] > q60) & (df["order_datetime"] <= q80)].copy()
    oot_df = df[df["order_datetime"] > q80].copy()
    
    # Формируем матрицы признаков и целевые переменные
    X_train = train_df.drop(cols_to_drop, axis=1)
    y_train = train_df[target_col]
    
    X_val = val_df.drop(cols_to_drop, axis=1)
    y_val = val_df[target_col]
    
    X_oot = oot_df.drop(cols_to_drop, axis=1)
    y_oot = oot_df[target_col]
    
    # Оборачиваем в Pool (CatBoost-формат)
    
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    oot_pool = Pool(X_oot, label=y_oot)

    return train_pool, val_pool, oot_pool 
   