import numpy as np
import numpy.typing as npt
import pandas as pd

from .methods import MCBoost


class CASMCBoostAlgorithm:
    """
    Our own MCBoost implementation.
    """

    def __init__(self, params) -> None:
        """
        Initialize Multicalibration Predictor.
        MCBoost variants:
        - jinetal uses only subgroups as categorical features;
        - nogroups uses only the features and not the groups;
        - alltogether uses both subgroups and features.
        """
        self.mcboost_variant = params["mcboost_variant"] or "jinetal"
        self.weight_column_name = params["weight_column_name"] or None
        self.categorical_feature_column_names = (
            params["categorical_feature_column_names"] or None
        )
        self.numerical_feature_column_names = (
            params["numerical_feature_column_names"] or None
        )
        self.auto_infer_column_types = params.get("auto_infer_column_types", True)
        self.categorical_threshold = params.get("categorical_threshold", 10)

        self.mcboost = MCBoost(
            encode_categorical_variables=params["encode_categorical_variables"] or True,
            monotone_t=params["monotone_t"] or None,
            num_rounds=params["num_rounds"] or 100,
            lightgbm_params=params["lightgbm_params"] or None,
            early_stopping=params["early_stopping"] or None,
            patience=params["patience"] or None,
            early_stopping_score_func=params["early_stopping_score_func"] or None,
            early_stopping_minimize_score=params["early_stopping_minimize_score"]
            or None,
            early_stopping_timeout=params["early_stopping_timeout"] or 8 * 60 * 60,
            save_training_performance=params["save_training_performance"] or False,
            monitored_metrics_during_training=params[
                "monitored_metrics_during_training"
            ]
            or None,
        )

    def fit(self, confs, labels, subgroups, df=None, categorical_features=None, numerical_features=None) -> None:
        if self.mcboost_variant in ["nogroups", "alltogether"] and df is None:
            raise ValueError(
                "df has to be passed if variant is nogroups or alltogether"
            )

        df['precali_scores'] = confs[:, 1]

        self.mcboost.fit(
            df_train=df,
            prediction_column_name="precali_scores",
            label_column_name="label",
            weight_column_name=self.weight_column_name,
            categorical_feature_column_names=categorical_features,
            numerical_feature_column_names=numerical_features,
        )

    def batch_predict(self, f_xs, groups, df=None, categorical_features=None, numerical_features=None) -> npt.NDArray:
        if self.mcboost_variant in ["nogroups", "alltogether"] and df is None:
            raise ValueError(
                "df has to be passed if variant is nogroups or alltogether"
            )

        return self.mcboost.predict(
            df=df,
            prediction_column_name="precali_scores",
            categorical_feature_column_names=categorical_features,
            numerical_feature_column_names=numerical_features,
            return_all_rounds=False,
        )
