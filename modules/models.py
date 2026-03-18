import catboost as cb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

from .feature_generation import get_features_df_and_targets
from .index_slicing import get_slice, get_cols_idx, direct_mimo_features_targets__train_idx, direct_mimo_features__test_idx, direct_mimo_features_targets__holdout_idx


class BaseModel:
    """Базовый класс модели."""

    def __init__(self):
        raise NotImplementedError

    def fit(
        self, train_data, val_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.
        - value_col: название столбца с значением ряда.

        Returns: None
        """
        raise NotImplementedError

    def predict(self, test_data, id_col="ts_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
        - test_data: DataFrame с тестовыми данными.
        - id_col: название столбца с идентификатором ряда.
        - timestamp_col: название столбца с временной меткой.

        Returns:
        - predictions: DataFrame с предсказанными значениями со столбцами
            [id_col, timestamp_col, 'predicted_value'].
        """
        raise NotImplementedError


class StatsforecastModel(BaseModel):
    """Модель, использующая библиотеку statsforecast для прогнозирования."""

    def __init__(self, model, freq: str, horizon: int):
        """Инициализация модели.

        Args:
            - model: экземпляр модели из библиотеки statsforecast.
            - freq: частота временного ряда (например, 'H' для почасовых данных).
            - horizon: общий горизонт прогнозирования.

        """
        self.model = model
        self.freq = freq
        self.horizon = horizon

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        """Обучение модели на тренировочных и валидационных данных.

        Args:
        - train_data: DataFrame с тренировочными данными.
        - val_data: DataFrame с валидационными данными.

        """
        # Объединяем тренировочные и валидационные данные
        combined_data = pd.concat([train_data, val_data])
        # Удаялем дубликаты, которые образовались после объединения
        # так как val_data начинается с history точек из конца train_data
        combined_data = combined_data.drop_duplicates(subset=[id_col, timestamp_col], keep="last")

        # Преобразуем данные в формат, необходимый для StatsForecast
        sf = StatsForecast(models=[self.model], freq=self.freq)
        self.sf = sf.fit(
            combined_data.rename(
                columns={id_col: "unique_id", timestamp_col: "ds", value_col: "y"}
            )
        )

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        """Прогнозирование на тестовых данных.

        Args:
            - test_data: DataFrame с тестовыми данными.

        Returns:
            - predictions: DataFrame с предсказанными значениями со столбцами
                [id_col, timestamp_col, 'predicted_value'].

        """
        forecasts = self.sf.predict(h=self.horizon)

        # Преобразуем прогнозы обратно в исходный формат
        pred_column = [col for col in forecasts.columns if col not in ["unique_id", "ds"]][0]
        predictions = forecasts[["unique_id", "ds", pred_column]].rename(
            columns={"unique_id": id_col, "ds": timestamp_col, pred_column: "predicted_value"}
        )

        return predictions


class CatBoostDirectMIMO(BaseModel):
    def __init__(
        self,
        model_horizon: int,
        history: int,
        horizon: int,
        freq: str,
        feature_config: dict | None = None,
    ):
        self.model_horizon = model_horizon
        self.history = history
        self.horizon = horizon
        self.freq = freq
        self.models = []
        self.feature_config = feature_config or {}

    def fit(
        self,
        train_data,
        val_data,
        id_col="sensor_id",
        timestamp_col="timestamp",
        value_col="value",
    ):
        steps = self.horizon // self.model_horizon
        self.models = []

        for h in range(steps):
            offset = self.model_horizon * h

            train_features_idx, train_targets_idx = direct_mimo_features_targets__train_idx(
                id_column=train_data[id_col],
                series_length=len(train_data),
                model_horizon=self.model_horizon,
                history_size=self.history,
                offset=offset,
            )
            val_features_idx, val_targets_idx = direct_mimo_features_targets__holdout_idx(
                id_column=val_data[id_col],
                series_length=len(val_data),
                model_horizon=self.model_horizon,
                history_size=self.history,
                offset=offset,
            )

            train_features, train_targets, categorical_features_idx = get_features_df_and_targets(
                train_data,
                train_features_idx,
                train_targets_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
                **self.feature_config,
            )
            val_features, val_targets, _ = get_features_df_and_targets(
                val_data,
                val_features_idx,
                val_targets_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
                **self.feature_config,
            )

            train_dataset = cb.Pool(
                data=train_features,
                label=train_targets,
                cat_features=categorical_features_idx,
            )
            eval_dataset = cb.Pool(
                data=val_features,
                label=val_targets,
                cat_features=categorical_features_idx,
            )

            cb_model = cb.CatBoostRegressor(
                loss_function="MultiRMSE",
                random_seed=42,
                verbose=100,
                iterations=200,
                learning_rate=0.1,
                depth=6,
                early_stopping_rounds=50,
                cat_features=categorical_features_idx,
            )

            cb_model.fit(
                train_dataset,
                eval_set=eval_dataset,
                use_best_model=True,
                plot=False,
            )

            self.models.append(cb_model)

    def predict(self, test_data, id_col="sensor_id", timestamp_col="timestamp", value_col="value"):
        steps = self.horizon // self.model_horizon
        preds_blocks = []

        for h in range(steps):
            offset = self.model_horizon * h

            test_features_idx, target_features_idx = direct_mimo_features__test_idx(
                id_column=test_data[id_col],
                series_length=len(test_data),
                model_horizon=self.model_horizon,
                history_size=self.history,
                offset=offset,
            )

            test_features, _, _ = get_features_df_and_targets(
                test_data,
                test_features_idx,
                target_features_idx,
                id_column=id_col,
                date_column=timestamp_col,
                target_column=value_col,
                **self.feature_config,
            )

            preds = self.models[h].predict(test_features)
            preds_blocks.append((target_features_idx, preds))

        result_parts = []
        for target_idx, preds in preds_blocks:
            pred_df = pd.DataFrame({
                id_col: test_data.iloc[target_idx.flatten()][id_col].values,
                timestamp_col: test_data.iloc[target_idx.flatten()][timestamp_col].values,
                "predicted_value": preds.reshape(-1),
            })
            result_parts.append(pred_df)

        predictions = pd.concat(result_parts, ignore_index=True)
        predictions = predictions.sort_values([id_col, timestamp_col]).reset_index(drop=True)

        return predictions
