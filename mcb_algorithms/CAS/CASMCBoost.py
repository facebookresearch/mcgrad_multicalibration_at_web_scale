import numpy as np
import numpy.typing as npt
import pandas as pd
import lightgbm as lgb
from configs.constants import FEATURE_TYPE_GROUPS, FEATURE_TYPE_FEATURES
import logging
from .methods import MCBoost
from .tuning import tune_mcboost_params, default_parameter_configurations


logger = logging.getLogger(__name__)

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
        self.tune_hyperparameters = params.get("tune_hyperparameters", True)
        # Fix some hyperparameters to be excluded from tuning
        self.fixed_parameters = params.get("fixed_parameters", [])
        self.tuning_parameter_space = [
            config for config in default_parameter_configurations if config.name not in self.fixed_parameters
        ]

        logger.info(f"Tuning hyperparameters {[c.name for c in self.tuning_parameter_space]}")

        mcb_cls = MCBoost if self.unshrink else MCBoostNoUnshrink

        self.mcboost = mcb_cls(
            encode_categorical_variables=params["encode_categorical_variables"] or True,
            monotone_t=params.get("monotone_t", None),
            num_rounds=params.get("num_rounds", 10),
            lightgbm_params=params.get('lightgbm_params', None),
            early_stopping=params.get("early_stopping", None),
            patience=params.get("patience", None),
            early_stopping_score_func=params.get("early_stopping_score_func", None),
            early_stopping_minimize_score=params.get("early_stopping_minimize_score", None),
            early_stopping_timeout=params.get("early_stopping_timeout", 8 * 60 * 60),
            save_training_performance=params.get("save_training_performance", False),
            monitored_metrics_during_training=params.get("monitored_metrics_during_training", None),
        )


    def _groups_to_dataframe(self, subgroups, n) -> pd.DataFrame:
        print(f"Groups to dataframe: {n}")
        if not subgroups:
            return pd.DataFrame()

        # Initialize an empty (n x len(subgroups)) array of zeros
        data = np.zeros((n, len(subgroups)), dtype=np.uint8)

        # Fill in 1s for each subgroup
        for i, segment in enumerate(subgroups):
            data[segment, i] = 1

        # Create column names
        col_names = [f"segment_{i + 1}" for i in range(len(subgroups))]

        # Return as DataFrame
        return pd.DataFrame(data, columns=col_names)

    def fit(self, confs, labels, subgroups, confs_val=None, labels_val=None, subgroups_val=None, df=None, df_val=None,
            categorical_features=None, numerical_features=None) -> None:

        if not self.tune_hyperparameters:
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
        else:
            if self.feature_type == FEATURE_TYPE_GROUPS:
                print("fitting with groups features")
                fit_df = self._groups_to_dataframe(subgroups, len(labels))
                val_df = self._groups_to_dataframe(subgroups_val, len(labels_val))
                categorical_features = list(fit_df.columns)
                numerical_features = []
            else:
                print("fitting with features features")
                fit_df = df.copy()
                val_df = df_val.copy()

            fit_df['precali_scores'] = confs[:, 1]
            fit_df['label'] = labels
            val_df['precali_scores'] = confs_val
            val_df['label'] = labels_val

            best_model, _ = tune_mcboost_params(
                model=self.mcboost,
                df_train=fit_df,
                df_val=val_df,
                prediction_column_name="precali_scores",
                label_column_name="label",
                categorical_feature_column_names=categorical_features,
                numerical_feature_column_names=numerical_features,
                parameter_configurations=self.tuning_parameter_space,
            )
            self.mcboost = best_model

    def batch_predict(self, f_xs, groups, df=None, categorical_features=None, numerical_features=None) -> npt.NDArray:
        if self.feature_type == FEATURE_TYPE_GROUPS:
            logger.info("predicting with groups features")
            predict_df = self._groups_to_dataframe(groups, len(f_xs))
            categorical_features = list(predict_df.columns)
            numerical_features = []
        else:
            logger.info("predicting with features features")
            predict_df = df.copy()

        predict_df['precali_scores'] = f_xs[:, 1]

        return self.mcboost.predict(
            df=predict_df,
            prediction_column_name="precali_scores",
            categorical_feature_column_names=categorical_features,
            numerical_feature_column_names=numerical_features,
            return_all_rounds=False,
        )
