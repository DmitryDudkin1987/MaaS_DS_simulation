
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

def train_model (train_pool: Pool, val_pool: Pool, oot_pool: Pool):
    cb_params = {
        "iterations": 1000,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 100,
        "random_state": 42,
        "verbose": False,
    }

    cb_baseline = CatBoostRegressor(**cb_params)

    cb_baseline.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    return cb_baseline

def evaluate_model(model: CatBoostRegressor, oot_pool: Pool):
    
        oot_pred = model.predict(oot_pool)
        oot_true = oot_pool.get_label()
        oot_rmse = np.sqrt(mean_squared_error(oot_true, oot_pred))
        return oot_rmse
