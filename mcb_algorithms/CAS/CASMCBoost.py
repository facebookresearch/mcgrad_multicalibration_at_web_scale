import numpy as np
import numpy.typing as npt
import pandas as pd
import lightgbm as lgb
from configs.constants import FEATURE_TYPE_GROUPS, FEATURE_TYPE_FEATURES

from .methods import MCBoost


class MCBoostNoUnshrink(MCBoost):
    def _fit_single_round(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        logits: npt.NDArray,
        w: npt.NDArray | None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> npt.NDArray:
        """
        This is a patch of the original fit function that omits the unshrink step
        """
        x = np.c_[x, logits]

        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        self.mr.append(
            lgb.train(
                params=self.get_lgbm_params(x),
                train_set=lgb.Dataset(
                    x,
                    label=y,
                    init_score=logits,
                    weight=w,
                    categorical_feature=categorical_feature_column_names,
                    feature_name=categorical_feature_column_names
                    + numerical_feature_column_names
                    + ["logits"],
                ),
            )
        )

        new_pred = self.mr[-1].predict(x, raw_score=True)
        self.unshrink_factors.append(1)
        return logits + new_pred



class CASMCBoostAlgorithm:

    def __init__(self, params) -> None:
        if params["feature_type"] not in [FEATURE_TYPE_GROUPS, FEATURE_TYPE_FEATURES]:
            raise ValueError("Invalid feature type")
        self.feature_type = params["feature_type"]
        self.unshrink = params["unshrink"]

        mcb_cls = MCBoost if self.unshrink else MCBoostNoUnshrink

        self.mcboost = mcb_cls(
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

    def _groups_to_dataframe(self, subgroups, n) -> pd.DataFrame:
        df_out = pd.DataFrame()
        if subgroups:
            for i, segment in enumerate(subgroups):
                segment_col = f"segment_{i + 1}"
                df_out[segment_col] = [
                    1 if idx in segment else 0 for idx in range(n)
                ]

        return df_out

    def fit(self, confs, labels, subgroups, df=None, categorical_features=None, numerical_features=None) -> None:

        if self.feature_type == FEATURE_TYPE_GROUPS:
            fit_df = self._groups_to_dataframe(subgroups, len(labels))
            categorical_features = list(fit_df.columns)
            numerical_features = []
        else:
            fit_df = df.copy()

        fit_df['precali_scores'] = confs[:, 1]
        fit_df['label'] = labels

        self.mcboost.fit(
            df_train=fit_df,
            prediction_column_name="precali_scores",
            label_column_name="label",
            categorical_feature_column_names=categorical_features,
            numerical_feature_column_names=numerical_features,
        )

    def batch_predict(self, f_xs, groups, df=None, categorical_features=None, numerical_features=None) -> npt.NDArray:
        if self.feature_type == FEATURE_TYPE_GROUPS:
            predict_df = self._groups_to_dataframe(groups, len(f_xs))
            categorical_features = list(predict_df.columns)
            numerical_features = []
        else:
            predict_df = df.copy()

        predict_df['precali_scores'] = f_xs[:, 1]

        return self.mcboost.predict(
            df=predict_df,
            prediction_column_name="precali_scores",
            categorical_feature_column_names=categorical_features,
            numerical_feature_column_names=numerical_features,
            return_all_rounds=False,
        )
