from typing import Sequence, Union

import numpy as np
import pandas as pd

from .index_slicing import get_cols_idx, get_slice


def _make_calendar_features(df: pd.DataFrame, date_column):
    df = df.copy()

    if np.issubdtype(df[date_column].dtype, np.datetime64):
        df["month"] = df[date_column].dt.month
        df["quarter"] = df[date_column].dt.quarter
        df["year"] = df[date_column].dt.year
    else:
        # для M4 monthly, если ds = 1,2,3,...
        ds_int = df[date_column].astype(int)
        df["month"] = ((ds_int - 1) % 12) + 1
        df["quarter"] = ((df["month"] - 1) // 3) + 1
        df["year"] = ((ds_int - 1) // 12) + 1

    return df


def _select_lag_positions(history_size: int, selected_lags):
    """
    history window устроено так:
    [lag_history_size, ..., lag_2, lag_1]

    значит:
    lag_1 -> позиция history_size - 1
    lag_12 -> позиция history_size - 12
    """
    positions = []
    for lag in selected_lags:
        if lag < 1 or lag > history_size:
            raise ValueError(f"Лаг {lag} не помещается в окно history_size={history_size}")
        positions.append(history_size - lag)
    return positions


def _make_fourier_for_targets(df: pd.DataFrame, targets_ids, date_column, period=12, order=2):
    target_times = get_slice(df, (targets_ids, get_cols_idx(df, date_column)))

    if np.issubdtype(df[date_column].dtype, np.datetime64):
        month_num = pd.to_datetime(target_times.reshape(-1)).month.values.reshape(target_times.shape)
        t = month_num
    else:
        t = target_times.astype(float)

    fourier_blocks = []
    for k in range(1, order + 1):
        fourier_blocks.append(np.sin(2 * np.pi * k * t / period))
        fourier_blocks.append(np.cos(2 * np.pi * k * t / period))

    return np.concatenate(fourier_blocks, axis=1)


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = "datetime",
    target_column: str = "target",
    use_id: bool = True,
    use_calendar: bool = True,
    selected_lags=None,          # None => все лаги
    use_fourier: bool = False,
    fourier_period: int = 12,
    fourier_order: int = 2,
):
    df = df.copy()
    df = _make_calendar_features(df, date_column)

    feature_blocks = []
    categorical_features_idx = []
    current_col = 0

    # 1. id
    if use_id:
        features_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
        feature_blocks.append(features_id)

        n_id_cols = features_id.shape[1] if features_id.ndim == 2 else 1
        categorical_features_idx.extend(range(current_col, current_col + n_id_cols))
        current_col += n_id_cols

    # 2. calendar
    if use_calendar:
        calendar_cols = ["month", "quarter", "year"]
        features_time = get_slice(df, (targets_ids, get_cols_idx(df, calendar_cols)))
        feature_blocks.append(features_time)

        n_time_cols = features_time.shape[1]
        categorical_features_idx.extend(range(current_col, current_col + n_time_cols))
        current_col += n_time_cols

    # 3. lags
    features_lags = get_slice(df, (features_ids, get_cols_idx(df, target_column)))

    if selected_lags is not None:
        history_size = features_lags.shape[1]
        lag_positions = _select_lag_positions(history_size, selected_lags)
        features_lags = features_lags[:, lag_positions]

    feature_blocks.append(features_lags)
    current_col += features_lags.shape[1]

    # 4. fourier
    if use_fourier:
        fourier_features = _make_fourier_for_targets(
            df=df,
            targets_ids=targets_ids,
            date_column=date_column,
            period=fourier_period,
            order=fourier_order,
        )
        feature_blocks.append(fourier_features)
        current_col += fourier_features.shape[1]

    features = np.hstack(feature_blocks)

    # catboost: категориальные в object/string
    features_obj = features.astype(object)
    for j in categorical_features_idx:
        features_obj[:, j] = features_obj[:, j].astype(str)

    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))

    return features_obj, targets, np.array(categorical_features_idx)
