from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

def get_slice(data: pd.DataFrame, k: Tuple[np.ndarray]) -> np.ndarray:
    """Получение среза из DataFrame по индексам строк и колонок.

    Args:
        data: Исходный DataFrame.
        k: Кортеж из двух элементов:
            - Массив индексов строк.
            - Массив индексов колонок или None (если нужны все колонки).

    Returns:
        Массив значений из DataFrame по заданным индексам.

    """
    rows, cols = k
    if cols is None:
        new_data = data.values[rows]
    else:
        new_data = data.iloc[:, cols].values[rows]

    if new_data.ndim == 3:
        new_data = new_data.reshape(new_data.shape[0], -1)

    return new_data


def get_cols_idx(data: pd.DataFrame, columns: Union[str, Sequence[str]]) -> Union[int, np.ndarray]:
    """Получение индексов колонок по их названиям.

    Args:
        data: DataFrame с колонками.
        columns: Название колонки или список названий колонок.

    Returns:
        Индекс колонки или массив индексов колонок.

    """
    if type(columns) is str:
        idx = data.columns.get_loc(columns)
    else:
        idx = data.columns.get_indexer(columns)
    return idx

def direct_mimo_features_targets__train_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
    offset: int,
):
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []

    window_size = history_size + offset + model_horizon

    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start_indices[i + 1]

        if series_end - series_start < window_size:
            continue

        sliding_window = np.lib.stride_tricks.sliding_window_view(
            np.arange(series_start, series_end),
            window_size
        )

        # признаки = первые history_size значений
        features_indices.append(sliding_window[:, :history_size])

        # таргет = блок длины model_horizon после offset
        targets_indices.append(sliding_window[:, history_size + offset: history_size + offset + model_horizon])

    return np.vstack(features_indices), np.vstack(targets_indices)


def direct_mimo_features__test_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
    offset: int,
):
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []

    total_size = history_size + offset + model_horizon

    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start_indices[i + 1]

        series_len = series_end - series_start
        if series_len < total_size:
            raise ValueError(
                f"Ряд {i} слишком короткий: нужен минимум {total_size}, есть {series_len}"
            )

        window = np.arange(series_start, series_start + total_size)

        features_indices.append(window[:history_size])
        targets_indices.append(
            window[history_size + offset : history_size + offset + model_horizon]
        )

    return np.vstack(features_indices), np.vstack(targets_indices)
    
def direct_mimo_features_targets__holdout_idx(
    id_column: pd.Series,
    series_length: int,
    model_horizon: int,
    history_size: int,
    offset: int,
):
    series_start_indices = np.append(
        np.unique(id_column.values, return_index=True)[1], series_length
    )

    features_indices = []
    targets_indices = []

    total_size = history_size + offset + model_horizon

    for i in range(len(series_start_indices) - 1):
        series_start = series_start_indices[i]
        series_end = series_start_indices[i + 1]

        series_len = series_end - series_start
        if series_len < total_size:
            raise ValueError(
                f"Ряд {i} слишком короткий: нужен минимум {total_size}, есть {series_len}"
            )

        # holdout уже имеет структуру [history | future]
        window = np.arange(series_start, series_start + total_size)

        features_indices.append(window[:history_size])
        targets_indices.append(
            window[history_size + offset : history_size + offset + model_horizon]
        )

    return np.vstack(features_indices), np.vstack(targets_indices)
