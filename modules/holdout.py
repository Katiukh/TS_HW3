import numpy as np
import pandas as pd

from .metrics import evaluate_forecasts
from .models import CatBoostDirectMIMO


def make_holdout_split(
    data: pd.DataFrame,
    id_col: str = "unique_id",
    timestamp_col: str = "ds",
    value_col: str = "y",
    history: int = 24,
    horizon: int = 18,
):
    data = data.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    min_len = history + horizon
    lengths = data.groupby(id_col).size()
    valid_ids = lengths[lengths >= min_len].index

    data = data[data[id_col].isin(valid_ids)].copy()
    data = data.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    train_parts = []
    holdout_fit_parts = []
    holdout_pred_parts = []
    truth_parts = []

    for uid, g in data.groupby(id_col, sort=False):
        g = g.sort_values(timestamp_col).reset_index(drop=True)

        train_part = g.iloc[:-horizon].copy()

        # для fit: holdout с истинными значениями
        holdout_fit_part = g.iloc[-(history + horizon):].copy()

        # для predict: те же строки, но будущее замаскировано
        holdout_pred_part = holdout_fit_part.copy()
        holdout_pred_part.loc[holdout_pred_part.index[-horizon:], value_col] = np.nan

        truth_part = g.iloc[-horizon:][[id_col, timestamp_col, value_col]].copy()

        train_parts.append(train_part)
        holdout_fit_parts.append(holdout_fit_part)
        holdout_pred_parts.append(holdout_pred_part)
        truth_parts.append(truth_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    holdout_fit_df = pd.concat(holdout_fit_parts, ignore_index=True)
    holdout_pred_df = pd.concat(holdout_pred_parts, ignore_index=True)
    truth_df = pd.concat(truth_parts, ignore_index=True)

    return train_df, holdout_fit_df, holdout_pred_df, truth_df


def run_holdout_experiment(
    data: pd.DataFrame,
    feature_config: dict,
    model_horizon: int = 6,
    history: int = 24,
    horizon: int = 18,
    freq: str = "M",
    id_col: str = "unique_id",
    timestamp_col: str = "ds",
    value_col: str = "y",
    seasonality: int = 12,
):
    train_df, holdout_fit_df, holdout_pred_df, truth_df = make_holdout_split(
        data=data,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
        history=history,
        horizon=horizon,
    )

    model = CatBoostDirectMIMO(
        model_horizon=model_horizon,
        history=history,
        horizon=horizon,
        freq=freq,
        feature_config=feature_config,
    )

    model.fit(
        train_df,
        holdout_fit_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
    )

    pred_df = model.predict(
        holdout_pred_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
    )

    metrics = evaluate_forecasts(
        train_df=train_df,
        truth_df=truth_df,
        pred_df=pred_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        value_col=value_col,
        seasonality=seasonality,
    )

    return {
        "train_df": train_df,
        "holdout_fit_df": holdout_fit_df,
        "holdout_pred_df": holdout_pred_df,
        "truth_df": truth_df,
        "pred_df": pred_df,
        "merged_predictions": metrics["merged_predictions"],
        "sMAPE": metrics["sMAPE"],
        "MASE": metrics["MASE"],
    }
