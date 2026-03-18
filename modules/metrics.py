import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0

    result = np.zeros_like(y_true, dtype=float)
    result[mask] = 200 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]

    return np.mean(result)


def mase_per_series(y_train, y_true, y_pred, seasonality=1):
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_train) <= seasonality:
        return np.nan

    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / scale


def evaluate_forecasts(
    train_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str = "unique_id",
    timestamp_col: str = "ds",
    value_col: str = "y",
    seasonality: int = 12,
):
    merged = truth_df.merge(
        pred_df,
        on=[id_col, timestamp_col],
        how="inner",
        validate="one_to_one",
    )

    overall_smape = smape(
        merged[value_col].values,
        merged["predicted_value"].values,
    )

    mase_values = []
    for uid, g_true in truth_df.groupby(id_col):
        y_train = train_df.loc[train_df[id_col] == uid, value_col].values
        g_pred = pred_df.loc[pred_df[id_col] == uid].sort_values(timestamp_col)
        g_true = g_true.sort_values(timestamp_col)

        if len(g_true) != len(g_pred):
            continue

        mase_val = mase_per_series(
            y_train=y_train,
            y_true=g_true[value_col].values,
            y_pred=g_pred["predicted_value"].values,
            seasonality=seasonality,
        )
        mase_values.append(mase_val)

    overall_mase = np.nanmean(mase_values)

    return {
        "sMAPE": overall_smape,
        "MASE": overall_mase,
        "merged_predictions": merged,
    }
