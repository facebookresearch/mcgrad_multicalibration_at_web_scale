# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import json
import logging
import time
import warnings

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial

from typing import Any, cast, Dict, Generic, TypeVar

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import isotonic, metrics as skmetrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from typing_extensions import Self

from . import utils
from .metrics import ScoreFunctionInterface, wrap_sklearn_metric_func

logger: logging.Logger = logging.getLogger(__name__)


class BaseCalibrator(ABC):
    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        This method fits the calibration method on the provided training data.

        :param df_train: the dataframe containing the training data
        :param prediction_column_name: name of the column in dataframe df that contains the predictions
        :param label_column_name: name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: list of column names in the df that contain the categorical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param numerical_feature_column_names: list of column names in the df that contain the numerical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        """
        pass

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """
        Applies a calibration model to a DataFrame. This requires the `fit` method to have been previously called on this calibrator object.

        :param df: the dataframe containing the data to calibrate
        :param prediction_column_name: name of the column in dataframe df that contains the predictions
        :param categorical_feature_column_names: list of column names in the df that contain the categorical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param numerical_feature_column_names: list of column names in the df that contain the numerical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        """
        pass

    def fit_transform(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        is_train_set_col_name: str | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """
        Fits the model using the training data and then applies the calibration transformation to all data.

        :param df: the dataframe containing the data to calibrate
        :param prediction_column_name: name of the column in dataframe df that contains the predictions
        :param label_column_name: name of the column in dataframe df that contains the ground truth labels
        :param weight_column_name: name of the column in dataframe df that contains the instance weights
        :param categorical_feature_column_names: list of column names in the df that contain the categorical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param numerical_feature_column_names: list of column names in the df that contain the numerical
            dimensions that are part of the segment space. This argument is ignored by methods that merely
            calibrate and do not multicalibrate (e.g., Isotonic regression and Platt scaling).
        :param is_train_set_col_name: name of the column in the dataframe that contains a boolean indicating
            whether the row is part of the training set (0) or test set (1). If no is_train_set_col_name is
            provided, then all rows are considered part of the training set.
        """
        df_train = (
            df if is_train_set_col_name is None else df[df[is_train_set_col_name]]
        )
        self.fit(
            df_train=df_train,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            **kwargs,
        )
        result = self.predict(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            **kwargs,
        )
        return result


class EstimationMethod(Enum):
    CROSS_VALIDATION = 1
    HOLDOUT = 2
    AUTO = 3


class MCBoost(BaseCalibrator):
    """
    Multicalibration boosting (MCBoost) as introduced in [1].

    References:
    [1] Hébert-Johnson, U., Kim, M., Reingold, O., & Rothblum, G. (2018). Multicalibration: Calibration for the
        (computationally-identifiable) masses. In International Conference on Machine Learning (pp. 1939-1948). PMLR.
    """

    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        # Resulting from a meta-analysis on 30 benchmark datasets in N6254284, specific to early stopped runs
        "monotone_t": False,
        "early_stopping": True,
        "patience": 0,
        "n_folds": 5,
        "lightgbm_params": {
            "learning_rate": 0.028729759162731475,
            "max_depth": 5,
            "min_child_samples": 160,
            "n_estimators": 94,
            "num_leaves": 5,
            "lambda_l2": 0.009131373863997217,
            "min_gain_to_split": 0.15007305226251808,
        },
    }
    VALID_SIZE = 0.4  # Relevant only when early_stopping_use_crossvalidation = False
    MCE_STAT_SIGN_THRESHOLD = 2.49767216  # p-value = 0.05
    MCE_STRONG_EVIDENCE_THRESHOLD = 4.70812972  # p-value = 0.00001
    DEFAULT_ALLOW_MISSING_SEGMENT_FEATURE_VALUES = True
    ESS_THRESHOLD_FOR_CROSS_VALIDATION = (
        2500000  # See N6787810 and self._determine_estimation_method for more details
    )

    def __init__(
        self,
        encode_categorical_variables: bool = True,
        monotone_t: bool | None = None,
        num_rounds: int = 100,
        lightgbm_params: dict[str, Any] | None = None,
        early_stopping: bool | None = None,
        patience: int | None = None,
        early_stopping_use_crossvalidation: bool | None = None,
        n_folds: int | None = None,
        early_stopping_score_func: ScoreFunctionInterface | None = None,
        early_stopping_minimize_score: bool | None = None,
        early_stopping_timeout: int | None = 8 * 60 * 60,  # 8 hours
        save_training_performance: bool = False,
        monitored_metrics_during_training: list[ScoreFunctionInterface] | None = None,
        allow_missing_segment_feature_values: bool = DEFAULT_ALLOW_MISSING_SEGMENT_FEATURE_VALUES,
    ) -> None:
        """
        :param encode_categorical_variables: whether to encode categorical variables using a modified label encoding (when True),
            or whether to assume that categorical variables are already manipulated into the right format prior to calling MCBoost
            (when False).
        :param monotone_t: whether to use a monotonicity constraint on the logit feature (i.e., t): value
            True implies that the decision tree is blocked from creating splits where a lower value of t
            results in a higher predicted probability.
        :param num_rounds: number of rounds boosting that is used in MCBoost. When early stopping is used, then num_rounds specifies the maximum
            number of rounds.
        :param lightgbm_params: the training parameters of lightgbm model. See: https://lightgbm.readthedocs.io/en/stable/Parameters.html
            if None, we will use a set of default parameters.
        :param early_stopping: whether to use early stopping based on cross-validation. When early stopping is used, then num_rounds specifies
            the maximum number of rounds that are fit, and the effective number of rounds is determined based on cross-validation.
        :param patience: the maximum number of consecutive rounds without improvement in `early_stopping_score_func`.
        :param early_stopping_use_crossvalidation: whether to use cross-validation (k-fold) for early stopping (otherwise use holdout). If set to None, then the evaluation method is determined automatically.
        :param early_stopping_score_func: the metric (default = log_loss if set to None) used to select the optimal number of rounds, when early stopping is used. It can be the Multicalibration Error (MulticalibrationError) or any SkLearn metric (SkLearnWrapper).
        :param early_stopping_minimize_score: whether the score function used for early stopping should be minimized. If set to False score is maximized.
        :param early_stopping_timeout: number of seconds after which early stopping is forced to stop and the number of rounds is determined. If set to None, then early stopping will not time out. Ignored when early stopping is disabled.
        :param n_folds: number of folds for k-fold cross-validation (used only when `early_stopping_use_crossvalidation` is `True`; or when that argument is `None` and k-fold is chosen automatically).
        :param save_training_performance: whether to save the training performance values for each round, in addition to the performance on the held-out validation set.
            This parameter is only relevant when early stopping is used. If set to False, then only the performance on the held-out validation set is saved.
        :param monitored_metrics_during_training: a list of metrics to monitor during training. This parameter is only relevant when early stopping is used.
            It includes which metrics to monitor during training, in addition to the metric used for early stopping (score_func).
        :param allow_missing_segment_feature_values: whether to allow missing values in the segment feature data. If set to True, missing values are used for training and prediction. If set to False, training with missing values will raise an Exception and prediction
            with missing values will return None.
        """

        if early_stopping_score_func is not None:
            assert (
                early_stopping_minimize_score is not None
            ), "If using a custom score function the attribute `early_stopping_minimize_score` has to be set."
            self.early_stopping_score_func: ScoreFunctionInterface = (
                early_stopping_score_func
            )
            self.early_stopping_minimize_score: bool = early_stopping_minimize_score
        else:
            # Note: When changing the default score function, make sure to update the default value of `early_stopping_minimize_score` in the next line accordingly.
            self.early_stopping_score_func = wrap_sklearn_metric_func(
                skmetrics.log_loss
            )
            self.early_stopping_minimize_score: bool = True
            assert (
                early_stopping_minimize_score is None
            ), f"`early_stopping_minimize_score` is only relevant when using a custom score function. The default score function is {self.early_stopping_score_func.name} for which `early_stopping_minimize_score` is set to {self.early_stopping_minimize_score} automatically."

        self._set_lightgbm_params(lightgbm_params)

        self.encode_categorical_variables = encode_categorical_variables
        self.MONOTONE_T: bool = (
            self.DEFAULT_HYPERPARAMS["monotone_t"] if monotone_t is None else monotone_t
        )
        self.NUM_ROUNDS: int = num_rounds

        self.EARLY_STOPPING: bool = (
            self.DEFAULT_HYPERPARAMS["early_stopping"]
            if early_stopping is None
            else early_stopping
        )

        if not self.EARLY_STOPPING:
            if patience is not None:
                raise ValueError(
                    "`patience` must be None when argument `early_stopping` is disabled."
                )
            if early_stopping_use_crossvalidation is not None:
                raise ValueError(
                    "`early_stopping_use_crossvalidation` must be None when `early_stopping` is disabled."
                )
            if early_stopping_score_func is not None:
                raise ValueError(
                    "`score_func` must be None when `early_stopping` is disabled."
                )
            if early_stopping_minimize_score is not None:
                raise ValueError(
                    "`minimize` must be None when `early_stopping` is disabled"
                )
            # Override the timeout when early stopping is disabled
            early_stopping_timeout = None

        self.EARLY_STOPPING_ESTIMATION_METHOD: EstimationMethod = (
            EstimationMethod.CROSS_VALIDATION
            if early_stopping_use_crossvalidation
            else (
                EstimationMethod.AUTO
                if early_stopping_use_crossvalidation is None
                else EstimationMethod.HOLDOUT
            )
        )

        if self.EARLY_STOPPING_ESTIMATION_METHOD == EstimationMethod.HOLDOUT:
            if n_folds is not None:
                raise ValueError(
                    "`n_folds` must be None when `early_stopping_use_crossvalidation` is disabled."
                )

        self.PATIENCE: int = (
            self.DEFAULT_HYPERPARAMS["patience"] if patience is None else patience
        )

        self.EARLY_STOPPING_TIMEOUT: int | None = early_stopping_timeout

        self.N_FOLDS: int = (
            1  # Because we make a single train/test split when using holdout
            if (self.EARLY_STOPPING_ESTIMATION_METHOD == EstimationMethod.HOLDOUT)
            else self.DEFAULT_HYPERPARAMS["n_folds"] if n_folds is None else n_folds
        )

        self.mr: list[lgb.Booster] = []
        self.unshrink_factors: list[float] = []
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None

        self.save_training_performance = save_training_performance
        self._performance_metrics: Dict[str, list[float]] = defaultdict(list)
        self.monitored_metrics_during_training: list[ScoreFunctionInterface] = (
            []
            if monitored_metrics_during_training is None
            else monitored_metrics_during_training
        )
        # Include the score function in the monitored metrics, if not there already
        if self.early_stopping_score_func.name not in [
            monitored_metric.name
            for monitored_metric in self.monitored_metrics_during_training
        ]:
            self.monitored_metrics_during_training.append(
                self.early_stopping_score_func
            )

        self.monitored_metrics_during_training = self._remove_duplicate_metrics(
            self.monitored_metrics_during_training
        )

        self.mce_below_initial: bool | None = None
        self.mce_below_strong_evidence_threshold: bool | None = None
        self.allow_missing_segment_feature_values = allow_missing_segment_feature_values
        self.categorical_feature_names: list[str] | None = None
        self.numerical_feature_names: list[str] | None = None

    def _set_lightgbm_params(self, lightgbm_params: dict[str, Any] | None) -> None:
        try:
            if self.mr:
                logger.warning(
                    "Model has already been fit. To avoid inconsistent state all training state will be reset after setting lightgbm_params."
                )
                self.reset_training_state()
        except AttributeError:
            pass

        lightgbm_params = (
            self.DEFAULT_HYPERPARAMS["lightgbm_params"] | dict(lightgbm_params)
            if lightgbm_params is not None
            else self.DEFAULT_HYPERPARAMS["lightgbm_params"]
        )
        assert (
            "num_rounds" not in lightgbm_params
        ), "avoid using `num_rounds` in `lightgbm_params` due to a naming conflict with `num_rounds` in MCBoost. Use any of the other aliases instead (https://lightgbm.readthedocs.io/en/latest/Parameters.html)"

        self.lightgbm_params: dict[str, Any] = {
            **lightgbm_params,
            "objective": "binary",
            "seed": 42,
            "deterministic": True,
            "verbosity": -1,
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns a dataframe with the feature importance of the final model.

        Importance is defined as the total gain from splits on a feature from the first round of MCBoost.

        :return: a dataframe with the feature importance.
        """
        if (
            not self.mr
            or self.categorical_feature_names is None
            or self.numerical_feature_names is None
        ):
            raise ValueError("Model has not been fit yet.")

        feature_importance = self.mr[0].feature_importance(importance_type="gain")

        return pd.DataFrame(
            {
                # Ordering of features here relies on two things 1) that MCBoost.extract_features returns first categoricals then
                # numericals and 2) that .fit method concatenates logits to the end of the feature matrix
                # pyre-ignore[58] if either feature_names attribute is None an error is raised above
                "feature": self.categorical_feature_names
                + self.numerical_feature_names
                + ["logits"],
                "importance": feature_importance,
            }
        ).sort_values("importance", ascending=False)

    def reset_training_state(self) -> None:
        self.mr = []
        self.unshrink_factors = []
        self.mce_below_initial = None
        self.mce_below_strong_evidence_threshold = None
        self._performance_metrics = defaultdict(list)
        self.enc: utils.OrdinalEncoderWithUnknownSupport | None = None
        self.categorical_feature_names = None
        self.numerical_feature_names = None

    @property
    def mce_is_satisfactory(self) -> bool | None:
        return self.mce_below_initial and self.mce_below_strong_evidence_threshold

    @property
    def performance_metrics(self) -> dict[str, list[float]]:
        if not self._performance_metrics:  # empty
            raise ValueError(
                "Performance metrics are only available after the model has been fit with `early_stopping=True`"
            )
        return self._performance_metrics

    def _check_segment_features(
        self,
        df: pd.DataFrame,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> None:
        segment_df = df[
            categorical_feature_column_names + numerical_feature_column_names
        ]
        if segment_df.isnull().any().any():
            if self.allow_missing_segment_feature_values:
                logger.info(
                    "Missing values found in segment feature data. MCBoost supports handling of missing data in segment features. If you want to disable native missing value support and predict None for examples with missing values in segment features, set `allow_missing_segment_feature_values=False` in the constructor of MCBoost. "
                )
            else:
                raise ValueError(
                    "Missing values found in segment feature data and `allow_missing_segment_feature_values` is set to False. If you want to enable native missing value support, set `allow_missing_segment_feature_values=True` in the constructor of MCBoost."
                )

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        MCBoost._check_scores(df_train, prediction_column_name)
        MCBoost._check_labels(df_train, label_column_name)
        self._check_segment_features(
            df_train,
            categorical_feature_column_names or [],
            numerical_feature_column_names or [],
        )

        self.reset_training_state()
        self.categorical_feature_names = categorical_feature_column_names or []
        self.numerical_feature_names = numerical_feature_column_names or []

        x = self.extract_features(
            df=df_train,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
            is_fit_phase=True,
        )
        logits = utils.logit(df_train[prediction_column_name].values)
        y = df_train[label_column_name].values.astype(float)
        w = (
            df_train[weight_column_name].values.astype(float)
            if weight_column_name
            else None
        )

        num_rounds = self.NUM_ROUNDS
        if self.EARLY_STOPPING:
            logger.info(
                "Starting early stopping to determine the number of MCBoost rounds"
                + (
                    " (with a {:,}-second timeout)".format(self.EARLY_STOPPING_TIMEOUT)
                    if self.EARLY_STOPPING_TIMEOUT
                    else ""
                )
            )

            num_rounds = self.determine_best_num_rounds(
                df_train,
                label_column_name,
                prediction_column_name,
                categorical_feature_column_names,
                numerical_feature_column_names,
                weight_column_name,
            )
        logger.info(
            f"Fitting {num_rounds} boosting rounds. The num_rounds argument was {self.NUM_ROUNDS}"
        )
        for _ in range(num_rounds):
            logits = self._fit_single_round(
                x=x,
                y=y,
                logits=logits,
                w=w,
                categorical_feature_column_names=categorical_feature_column_names,
                numerical_feature_column_names=numerical_feature_column_names,
            )

        return self

    def _fit_single_round(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        logits: npt.NDArray,
        w: npt.NDArray | None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> npt.NDArray:
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
        logits = logits + new_pred
        self.unshrink_factors.append(utils.unshrink(y, logits, w))
        logits *= self.unshrink_factors[-1]

        return logits

    def _get_output_presence_mask(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> npt.NDArray:
        """
        Returns a boolean mask indicating for which examples predictions are valid (i.e., not NaN).

        For examples with missing or otherwise invalid uncalibrated score as well as for examples with missing segment features (if self.allow_missing_segment_feature_values is False), predictions are not valid.
        """
        predictions = df[prediction_column_name].to_numpy()
        nan_mask = np.isnan(predictions)
        outofbounds_mask = (predictions < 0) | (predictions > 1)
        if nan_mask.any():
            logger.warning(
                f"MCBoost does not support missing values in the prediction column. Found {nan_mask.sum()} missing values. MCBoost.predict will return np.nan for these predictions."
            )
        if outofbounds_mask.any():
            min_score = np.min(df[prediction_column_name].values)
            max_score = np.max(df[prediction_column_name].values)
            logger.warning(
                f"MCBoost calibrates probabilistic binary classifiers, hence predictions must be in (0,1). Found min {min_score} and max {max_score}. MCBoost.predict will return np.nan for these predictions."
            )
        invalid_mask = nan_mask | outofbounds_mask
        if not self.allow_missing_segment_feature_values:
            segment_feature_missing_mask = (
                df[categorical_feature_column_names + numerical_feature_column_names]
                .isnull()
                .any(axis=1)
            )
            if segment_feature_missing_mask.any():
                logger.warning(
                    f"Found {segment_feature_missing_mask.sum()} missing values in segment features. MCBoost.predict will return np.nan for these predictions. MCBoost supports handling of missing data in segment features. If you want to enable native missing value support set `allow_missing_segment_feature_values=True` in the constructor of MCBoost. "
                )
            invalid_mask = invalid_mask | segment_feature_missing_mask
        return np.logical_not(invalid_mask)

    @staticmethod
    def _check_scores(
        df_train: pd.DataFrame,
        prediction_column_name: str,
    ) -> None:
        predictions = df_train[prediction_column_name].to_numpy()
        if (predictions < 0).any() or (predictions > 1).any():
            min_score = np.min(df_train[prediction_column_name].values)
            max_score = np.max(df_train[prediction_column_name].values)
            raise ValueError(
                f"MCBoost calibrates probabilistic binary classifiers, hence predictions must be in (0,1). Found min {min_score} and max {max_score}"
            )
        if df_train[prediction_column_name].isnull().any():
            raise ValueError(
                f"MCBoost does not support missing values in the prediction column, but {df_train[prediction_column_name].isnull().sum()}"
                f" of {len(df_train[prediction_column_name])} are null."
            )

    @staticmethod
    def _check_labels(df_train: pd.DataFrame, label_column_name: str) -> None:
        if df_train[label_column_name].isnull().any():
            raise ValueError(
                f"MCBoost does not support missing values in the label column, but {df_train[label_column_name].isnull().sum()}"
                f" of {len(df_train[label_column_name])} are null."
            )
        unique_labels = list(df_train[label_column_name].unique())
        labels_are_valid_int = df_train[label_column_name].isin([0, 1]).all()
        labels_are_valid_bool = df_train[label_column_name].isin([True, False]).all()
        if not (labels_are_valid_bool or labels_are_valid_int):
            raise ValueError(
                f"Labels in column `{label_column_name}` must be binary, either 0/1 or True/False. Got {unique_labels=}"
            )
        if not len(unique_labels) == 2:
            raise ValueError(
                f"Labels in column `{label_column_name}` must have at least 2 values but the data contains only 1: {unique_labels=}"
            )

    @staticmethod
    def _remove_duplicate_metrics(
        monitored_metrics_during_training: list[ScoreFunctionInterface],
    ) -> list[ScoreFunctionInterface]:
        """
        Removes duplicate metrics from the list of monitored metrics during training.
        """
        unique_metrics = []
        for metric in monitored_metrics_during_training:
            if metric.name not in [m.name for m in unique_metrics]:
                unique_metrics.append(metric)
        return unique_metrics

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        return_all_rounds: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray:
        presence_mask = self._get_output_presence_mask(
            df,
            prediction_column_name,
            categorical_feature_column_names or [],
            numerical_feature_column_names or [],
        )
        x = self.extract_features(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        predictions = self._predict(
            x, df[prediction_column_name].values, return_all_rounds
        )

        return np.where(presence_mask, predictions, np.nan)

    def _predict(
        self,
        x: npt.NDArray,
        preds: npt.NDArray,
        return_all_rounds: bool = False,
    ) -> npt.NDArray:
        """
        Predicts the calibrated probabilities using the trained model.

        :param x: the segment features.
        :param preds: the uncalibrated predictions that we are looking to calibrate.
        """
        assert len(self.mr) == len(self.unshrink_factors)
        if len(self.mr) < 1:
            logger.warning(
                "MCBoost has not been fit. Returning the uncalibrated predictions."
            )
            return preds.reshape(1, -1) if return_all_rounds else preds

        logits = utils.logit(preds)
        x = np.c_[x, logits]
        predictions_per_round = np.zeros((len(self.mr), len(logits)))
        for i in range(len(self.mr)):
            new_pred = self.mr[i].predict(x, raw_score=True)
            logits += new_pred
            logits *= self.unshrink_factors[i]
            x[:, -1] = logits
            predictions_per_round[i] = utils.logistic_vectorized(logits)

        return predictions_per_round if return_all_rounds else predictions_per_round[-1]

    def get_lgbm_params(self, x: npt.NDArray) -> dict[str, Any]:
        lgb_params = self.lightgbm_params.copy()
        if self.MONOTONE_T:
            score_constraint = [1]
            segment_feature_constraints = [0] * (x.shape[1] - 1)
            lgb_params["monotone_constraints"] = (
                segment_feature_constraints + score_constraint
            )
        return lgb_params

    def extract_features(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
        is_fit_phase: bool = False,
    ) -> npt.NDArray:
        if categorical_feature_column_names:
            cat_features = df[categorical_feature_column_names].values
            if self.encode_categorical_variables:
                if is_fit_phase:
                    self.enc = utils.OrdinalEncoderWithUnknownSupport()
                    self.enc.fit(cat_features)

                if self.enc is not None:
                    cat_features = self.enc.transform(cat_features)
                else:
                    raise ValueError(
                        "Fit has to be called before encoder can be applied."
                    )
            if np.nanmax(cat_features) >= np.iinfo(np.int32).max:
                raise ValueError(
                    "All categorical feature values must be smaller than 2^32 to prevent integer overflow internal to LightGBM."
                )
            if not self.encode_categorical_variables and np.nanmin(cat_features) < 0:
                raise ValueError(
                    "All categorical feature values must be non-negative, because LightGBM treats negative categorical values as missing."
                )
        else:
            cat_features = np.empty((df.shape[0], 0))

        if numerical_feature_column_names:
            num_features = df[numerical_feature_column_names].values
        else:
            num_features = np.empty((df.shape[0], 0))

        x = np.concatenate((cat_features, num_features), axis=1)
        return x

    def determine_best_num_rounds(
        self,
        df_train: pd.DataFrame,
        label_column_name: str,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
        weight_column_name: str | None,
    ) -> int:
        MCBoost._check_scores(df_train, prediction_column_name)
        MCBoost._check_labels(df_train, label_column_name)

        patience_counter = 0

        num_rounds = 0
        best_num_rounds = 0

        mcboost_per_fold: Dict[int, MCBoost] = {}
        logits_per_fold: Dict[int, npt.NDArray] = {}

        final_n_folds: int = 0

        # The train/test splitter (trainTestSplitter) must have a random_state set to ensure that the folds are the same across runs.
        if (
            self._determine_estimation_method(
                df_train[weight_column_name]
                if weight_column_name
                else np.ones(len(df_train))  # Use uniform weights
            )
            == EstimationMethod.CROSS_VALIDATION
        ):
            trainTestSplitter = StratifiedKFold(
                n_splits=self.N_FOLDS,
                shuffle=True,
                random_state=42,
            )
            final_n_folds = self.N_FOLDS
        else:
            trainTestSplitter = utils.TrainTestSplitWrapper(
                test_size=self.VALID_SIZE,
                shuffle=True,
                random_state=42,
            )
            final_n_folds = 1  # Because we make a single train/test split

        best_score = -np.inf

        start_time = time.time()

        while num_rounds <= self.NUM_ROUNDS and patience_counter <= self.PATIENCE:
            if self.EARLY_STOPPING_TIMEOUT is not None and self._get_elapsed_time(
                start_time
            ) > cast(
                int, self.EARLY_STOPPING_TIMEOUT
            ):  # Cast is needed to prevent code-verification issues
                logger.warn(
                    f"Stopping cross-validation upon exceeding the {self.EARLY_STOPPING_TIMEOUT:,}-second timeout; "
                    + "MCBoost results will likely improve by increasing `early_stopping_timeout` or setting it to None"
                )
                break

            valid_monitored_metrics_per_round = np.zeros(
                (len(self.monitored_metrics_during_training), final_n_folds),
                dtype=float,
            )
            train_monitored_metrics_per_round = np.zeros(
                (len(self.monitored_metrics_during_training), final_n_folds),
                dtype=float,
            )

            fold_num = 0
            for train_index, valid_index in trainTestSplitter.split(
                df_train, df_train[label_column_name].values
            ):
                df_train_cv, df_valid_cv = (
                    df_train.iloc[train_index],
                    df_train.iloc[valid_index],
                )
                if num_rounds == 0:
                    # Get predictions from the uncalibrated model for step 0
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df_train["mcboost_preds"] = df_train[
                            prediction_column_name
                        ].values
                else:
                    if fold_num not in mcboost_per_fold:
                        mcboost = MCBoost(
                            encode_categorical_variables=self.encode_categorical_variables,
                            monotone_t=self.MONOTONE_T,
                            lightgbm_params=self.lightgbm_params,
                            early_stopping=False,
                            num_rounds=0,
                        )
                        mcboost_per_fold[fold_num] = mcboost

                        logits_per_fold[fold_num] = utils.logit(
                            df_train_cv[prediction_column_name].values
                        )

                    x = mcboost.extract_features(
                        df=df_train_cv,
                        prediction_column_name=prediction_column_name,
                        categorical_feature_column_names=categorical_feature_column_names,
                        numerical_feature_column_names=numerical_feature_column_names,
                        is_fit_phase=True,
                    )
                    y = df_train_cv[label_column_name].values.astype(float)
                    w = (
                        df_train_cv[weight_column_name].values.astype(float)
                        if weight_column_name
                        else None
                    )
                    new_logits = mcboost_per_fold[fold_num]._fit_single_round(
                        x=x,
                        y=y,
                        logits=logits_per_fold[fold_num],
                        w=w,
                        categorical_feature_column_names=categorical_feature_column_names,
                        numerical_feature_column_names=numerical_feature_column_names,
                    )
                    logits_per_fold[fold_num] = new_logits

                    logger.info(f"CV fit with {num_rounds} rounds")
                    if self.save_training_performance:
                        full_preds = mcboost_per_fold[fold_num].predict(
                            df=df_train,
                            prediction_column_name=prediction_column_name,
                            categorical_feature_column_names=categorical_feature_column_names,
                            numerical_feature_column_names=numerical_feature_column_names,
                        )
                    else:
                        mcboost_valid_preds = mcboost_per_fold[fold_num].predict(
                            df=df_valid_cv,
                            prediction_column_name=prediction_column_name,
                            categorical_feature_column_names=categorical_feature_column_names,
                            numerical_feature_column_names=numerical_feature_column_names,
                        )
                        full_preds = np.zeros((df_train.shape[0],))
                        full_preds[valid_index] = mcboost_valid_preds
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df_train["mcboost_preds"] = full_preds

                for metric_idx, monitored_metric in enumerate(
                    self.monitored_metrics_during_training
                ):
                    valid_monitored_metrics_per_round[metric_idx, fold_num] = (
                        monitored_metric(
                            df=df_train.iloc[valid_index],
                            label_column=label_column_name,
                            score_column="mcboost_preds",
                            weight_column=weight_column_name,
                        )
                    )
                    if self.save_training_performance:
                        train_monitored_metrics_per_round[metric_idx, fold_num] = (
                            monitored_metric(
                                df=df_train.iloc[train_index],
                                label_column=label_column_name,
                                score_column="mcboost_preds",
                                weight_column=weight_column_name,
                            )
                        )

                fold_num += 1

            valid_mean_scores = np.mean(valid_monitored_metrics_per_round, axis=1)
            train_mean_scores = np.mean(train_monitored_metrics_per_round, axis=1)

            for metric_idx, monitored_metric in enumerate(
                self.monitored_metrics_during_training
            ):
                self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ].append(valid_mean_scores[metric_idx])
                logger.info(
                    f"avg_valid_performance_{monitored_metric.name} on the validation set for step #{num_rounds} = {valid_mean_scores[metric_idx]}"
                )
                if self.save_training_performance:
                    self._performance_metrics[
                        f"avg_train_performance_{monitored_metric.name}"
                    ].append(train_mean_scores[metric_idx])

            current_score = (
                -self._performance_metrics[
                    f"avg_valid_performance_{self.early_stopping_score_func.name}"
                ][-1]
                if self.early_stopping_minimize_score
                else self._performance_metrics[
                    f"avg_valid_performance_{self.early_stopping_score_func.name}"
                ][-1]
            )
            logger.info(
                f"CV score used for early stopping: {current_score}, best = {best_score}"
            )
            if current_score > best_score:
                best_score = current_score
                best_num_rounds = num_rounds
                patience_counter = 0
                logger.info(
                    f"New best score: {best_score} with {num_rounds} rounds, patience reset"
                )
            else:
                patience_counter += 1
                logger.info(f"Score did not improve for {patience_counter} rounds")
            num_rounds += 1

        if best_num_rounds == 0:
            logger.warning(
                "Selected 0 to be the best number of rounds for MCBoost for this dataset, meaning that uncalibrated predictions will be returned. This is because the optimization metric did not improve during the first round of boosting."
            )
        else:
            logger.info(
                f"Selected {best_num_rounds} to be the best number of rounds for MCBoost for this dataset."
            )

        # Throw warnings if
        # (1) the final validation MCE > MCE_STRONG_EVIDENCE_THRESHOLD (mce_below_strong_evidence_threshold is False), or
        # (2) the final validation MCE >= initial validation MCE (self.mce_below_initial is False).
        for monitored_metric in self.monitored_metrics_during_training:
            if monitored_metric.name == "Multicalibration Error<br>(mce_sigma_scale)":
                mce_at_best_num_rounds = self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ][best_num_rounds]
                mce_at_initial_round = self._performance_metrics[
                    f"avg_valid_performance_{monitored_metric.name}"
                ][0]

                self.mce_below_initial = (
                    True if mce_at_best_num_rounds < mce_at_initial_round else False
                )
                self.mce_below_strong_evidence_threshold = (
                    True
                    if mce_at_best_num_rounds < self.MCE_STRONG_EVIDENCE_THRESHOLD
                    else False
                )

                if not self.mce_below_strong_evidence_threshold:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using MCBoost is {mce_at_best_num_rounds}. This is higher than 4.0, which still indicates strong evidence for miscalibration."
                    )
                if not self.mce_below_initial:
                    logger.warning(
                        f"The final Multicalibration Error on the validation set after using MCBoost is {mce_at_best_num_rounds}, which is not lower than the initial Multicalibration Error of {mce_at_initial_round}. This indicates that MCBoost did not improve the multi-calibration of the model."
                    )

        return best_num_rounds

    def _get_elapsed_time(self, start_time: float) -> int:
        """
        Returns the elapsed time since the given start time in seconds.
        """
        return int(time.time() - start_time)

    def serialize(self) -> str:
        """
        Serializes the model into a JSON string.

        :return: serialised model.
        """
        serialized_boosters = [booster.model_to_string() for booster in self.mr]
        json_obj: dict[str, Any] = {
            "mcboost": [
                {
                    "booster": serialized_booster,
                    "unshrink_factor": unshrink_factor,
                }
                for serialized_booster, unshrink_factor in zip(
                    serialized_boosters, self.unshrink_factors
                )
            ],
            "params": {
                "allow_missing_segment_feature_values": self.allow_missing_segment_feature_values,
            },
        }
        json_obj["has_encoder"] = self.encode_categorical_variables
        if hasattr(self, "enc") and self.enc is not None:
            json_obj["encoder"] = self.enc.serialize()
        return json.dumps(json_obj)

    @classmethod
    def deserialize(cls, model_str: str) -> "MCBoost":
        """
        Deserializes a JSON string into a MCBoost model.

        :param model_str: serialized model (i.e., an output string of MCBoost serialize method).
        :return: deserialized model.
        """
        json_obj = json.loads(model_str)
        model = cls()

        # Initialise models
        model.mr = []
        model.unshrink_factors = []

        # Restore the trained models and unshrink factors
        for model_info in json_obj["mcboost"]:
            booster = lgb.Booster(model_str=model_info["booster"])
            model.mr.append(booster)
            model.unshrink_factors.append(model_info["unshrink_factor"])

        # Used in the predict method. We need to set correctly to use correct num rounds at inference time.
        model.NUM_ROUNDS = len(model.mr)

        # Restore the encoder. The encoder is made optional since at production serving time we can't
        # run sklearn encoders and instead must encode categorical features before model call.
        model.encode_categorical_variables = json_obj["has_encoder"]
        if json_obj["has_encoder"] and "encoder" in json_obj:
            model.enc = utils.OrdinalEncoderWithUnknownSupport.deserialize(
                json_obj["encoder"]
            )

        return model

    def _compute_effective_sample_size(self, weights: npt.NDArray) -> int:
        """
        Computes the effective sample size for the given weights.
        The effective sample size is defined as square of the sum of weights over the sum of the squared weights,
        as common in the importance sampling literature (e.g., see https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02412-1).

        :param weights: weights for each sample.
        :return: effective sample size.
        """
        # Compute the effective sample size using the weights
        return (weights.sum() ** 2) / np.power(weights, 2).sum()

    def _determine_estimation_method(self, weights: npt.NDArray) -> EstimationMethod:
        """
        Returns the estimation method to use for early stopping given the arguments and the weights (when relevant).
        This is especially useful for the AUTO option, where we infer the proper estimation method to use based on the effective sample size.

        :return: the estimation method to use.
        """
        if self.EARLY_STOPPING_ESTIMATION_METHOD != EstimationMethod.AUTO:
            return self.EARLY_STOPPING_ESTIMATION_METHOD

        if self.early_stopping_score_func.name != "log_loss":
            # Automatically infer the estimation method only when using the logistic loss, otherwise use k-fold.
            # This is because we analyzed the effective sample size specifically with log_loss.
            return EstimationMethod.CROSS_VALIDATION

        # We use a rule-of-thumb to determine whether to use cross-validation or holdout for early stopping.
        # Namely, if the effective sample size is less than 2.5M, we use cross-validation, otherwise we use holdout.
        # See N6787810 for more details.
        ess = self._compute_effective_sample_size(weights)

        if ess < self.ESS_THRESHOLD_FOR_CROSS_VALIDATION:
            logger.info(
                f"Found a relatively small effective sample size ({ess:,}), choosing k-fold for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `False`."
            )
            return EstimationMethod.CROSS_VALIDATION
        else:
            logger.info(
                f"Found a large enough effective sample size ({ess:,}), choosing holdout for early stopping. "
                + "You can override this by explicitly setting `early_stopping_use_crossvalidation` to `True`."
            )
            return EstimationMethod.HOLDOUT


class PlattScaling(BaseCalibrator):
    """
    Provides an implementation of Platt scaling, which is just a Logistic Regression applied to the logits.

    References:
    - Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
        likelihood methods. Advances in large margin classifiers, 10(3), 61-74.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
        International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
        self.log_reg = LogisticRegression()

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        y = df_train[label_column_name].values.astype(float)
        y_hat = df_train[prediction_column_name].values.astype(float)
        w = df_train[weight_column_name] if weight_column_name else np.ones_like(y)

        logits = utils.logit(y_hat).reshape(-1, 1)
        self.log_reg = LogisticRegression(penalty=None)
        self.log_reg.fit(logits, y, sample_weight=w)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        y_hat = df[prediction_column_name].values.astype(float)

        logits = utils.logit(y_hat).reshape(-1, 1)
        return self.log_reg.predict_proba(logits)[:, 1]


class IsotonicRegression(BaseCalibrator):
    """
    Provides an implementation of Isotonic regression. For input values outside of the training
    domain, predictions are set to the value corresponding to the nearest training interval endpoint.

    References:
    - Zadrozny, B., & Elkan, C. (2001). Obtaining calibrated probability estimates from decision trees and
        naive bayesian classifiers. International Conference on Machine Learning (ICML). pp. 609-616.
    - Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
        International Conference on Machine Learning (ICML). pp. 625-632.
    """

    def __init__(self) -> None:
        self.isoreg = isotonic.IsotonicRegression()

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        y = df_train[label_column_name].values.astype(float)
        y_hat = df_train[prediction_column_name].values.astype(float)
        w = df_train[weight_column_name] if weight_column_name else np.ones_like(y)

        # out_of_bounds=clip ensures predictions outside training domain range are clipped to nearest valid value instead of NaN
        # These are set to nearest train interval endpoints
        self.isoreg = isotonic.IsotonicRegression(out_of_bounds="clip").fit(
            y_hat, y, sample_weight=w
        )
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        y_hat = df[prediction_column_name].values.astype(float)
        return self.isoreg.transform(y_hat)


class MultiplicativeAdjustment(BaseCalibrator):
    """
    Calibrates predictions by multiplying scores with a correction factor derived from the ratio of total positive
    labels to sum of predicted scores. This helps align the overall prediction distribution with the true label distribution.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
        self.multiplier: float | None = None
        self.clip_to_zero_one = clip_to_zero_one

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        w = (
            df_train[weight_column_name]
            if weight_column_name
            else np.ones(df_train.shape[0])
        )
        total_score = (w * df_train[prediction_column_name]).sum()
        total_positive = (w * df_train[label_column_name]).sum()
        self.multiplier = total_positive / total_score if total_score != 0 else 1.0
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        preds = df[prediction_column_name].values * self.multiplier
        if self.clip_to_zero_one:
            preds = np.clip(preds, 0, 1)
        return preds


class AdditiveAdjustment(BaseCalibrator):
    """
    Calibrates predictions by adding a correction term derived from the difference between total positive labels
    and sum of predicted scores. This helps align the overall prediction distribution with the true label distribution.
    """

    def __init__(self, clip_to_zero_one: bool = True) -> None:
        self.offset: float | None = None
        self.clip_to_zero_one = clip_to_zero_one

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        w = (
            df_train[weight_column_name]
            if weight_column_name
            else np.ones(df_train.shape[0])
        )
        total_score = (w * df_train[prediction_column_name]).sum()
        total_positive = (w * df_train[label_column_name]).sum()
        self.offset = (total_positive - total_score) / w.sum()
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        preds = df[prediction_column_name].values + self.offset
        if self.clip_to_zero_one:
            preds = np.clip(preds, 0, 1)
        return preds


class IdentityCalibrator(BaseCalibrator):
    """
    A pass-through calibrator that returns predictions unchanged. Useful as a baseline or fallback option.
    """

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        return df[prediction_column_name].values


class SwissCheesePlattScaling(BaseCalibrator):
    """
    A variant of Platt scaling that incorporates additional categorical and numerical features alongside logits.
    Numerical features are discretized into bins.
    """

    def __init__(self) -> None:
        self.log_reg = LogisticRegression()
        self.logits_column_name = "__logits"
        self.ohe: OneHotEncoder | None = None
        self.kbd: KBinsDiscretizer | None = None
        self.ohe_columns: list[str] | None = None
        self.kbd_columns: list[str] | None = None
        self.features: list[str] | None = None

    def fit_feature_encoders(
        self,
        df: pd.DataFrame,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
    ) -> None:
        if categorical_feature_column_names:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(df[categorical_feature_column_names])
        else:
            self.ohe = None

        if numerical_feature_column_names:
            self.kbd = KBinsDiscretizer(encode="onehot-dense", n_bins=3, subsample=None)
            self.kbd.fit(df[numerical_feature_column_names])
        else:
            self.kbd = None

    def convert_df(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None,
        numerical_feature_column_names: list[str] | None,
    ) -> pd.DataFrame:
        y_hat = df[prediction_column_name].values.astype(float)
        df[self.logits_column_name] = utils.logit(y_hat)
        if categorical_feature_column_names and self.ohe is not None:
            ohe_df = pd.DataFrame(
                self.ohe.transform(df[categorical_feature_column_names])
            )
            if hasattr(self.ohe, "get_feature_names"):
                ohe_df.columns = self.ohe.get_feature_names(  # pyre-ignore: Maintain compatibility with sklearn <1.0
                    categorical_feature_column_names
                )
            elif hasattr(self.ohe, "get_feature_names_out"):
                ohe_df.columns = self.ohe.get_feature_names_out(  # pyre-ignore
                    categorical_feature_column_names
                )
            else:
                raise ValueError(
                    "Could not obtain feature names from OneHotEncoder. Expected get_feature_names_out for sklearn >1.0 or get_feature_names for sklearn <1.0."
                )
            df = pd.concat([df, ohe_df], axis=1)
            self.ohe_columns = list(ohe_df.columns)
        else:
            self.ohe_columns = []

        if numerical_feature_column_names and self.kbd is not None:
            kbd_df = pd.DataFrame(
                self.kbd.transform(df[numerical_feature_column_names])
            )
            kbd_df.columns = [str(col) for col in kbd_df.columns]
            df = pd.concat([df, kbd_df], axis=1)
            self.kbd_columns = list(kbd_df.columns)
        else:
            self.kbd_columns = []

        return df

    def train_model(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> LogisticRegression:
        categorical_feature_column_names = self.ohe_columns or []
        numerical_feature_column_names = self.kbd_columns or []

        features = (
            [self.logits_column_name]
            + categorical_feature_column_names
            + numerical_feature_column_names
        )

        y = df[label_column_name].values.astype(float)

        w = (
            df[weight_column_name].values
            if weight_column_name
            else np.ones(df.shape[0])
        )
        w = w.astype(float)

        log_reg = LogisticRegression(C=0.1).fit(df[features], y, sample_weight=w)
        self.features = features
        return log_reg

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        df_train = df_train.copy().reset_index().fillna(0)
        self.fit_feature_encoders(
            df_train, categorical_feature_column_names, numerical_feature_column_names
        )

        df_train = self.convert_df(
            df_train,
            prediction_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )

        log_reg = self.train_model(
            df_train,
            prediction_column_name,
            label_column_name,
            weight_column_name,
            categorical_feature_column_names,
            numerical_feature_column_names,
        )
        self.log_reg = log_reg
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        df = df.copy().reset_index().fillna(0)

        df = self.convert_df(
            df=df,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        return self.log_reg.predict_proba(df[self.features])[:, 1]


TCalibrator = TypeVar("TCalibrator", bound=BaseCalibrator)


class SegmentwiseCalibrator(Generic[TCalibrator], BaseCalibrator):
    """
    A meta-calibrator that partitions data into segments based on categorical features and applies a separate calibration
    method to each segment. This enables more precise calibration when different segments require different calibration
    adjustments.
    """

    calibrator_per_segment: dict[str, BaseCalibrator]
    calibrator_class: type[TCalibrator]
    calibrator_kwargs: dict[str, Any]

    def __init__(
        self,
        calibrator_class: type[TCalibrator],
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.calibrator_class = calibrator_class
        self.calibrator_kwargs = calibrator_kwargs or {}

        # Check if calibrator_class can be instantiated with calibrator_kwargs
        try:
            self.calibrator_class(**self.calibrator_kwargs)
        except TypeError:
            raise ValueError(
                f"Unable to instantiate calibrator class {self.calibrator_class.__name__} with the provided keyword arguments: {str(calibrator_kwargs)}"
            )

        self.calibrator_per_segment = {}

    def fit(
        self,
        df_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        # Create a unique identifier for each segment
        df_train = df_train.copy()
        df_train["segment"] = df_train[categorical_feature_column_names].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )

        fit_segment_func = partial(
            self._fit_segment,
            prediction_column_name=prediction_column_name,
            label_column_name=label_column_name,
            weight_column_name=weight_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        df_train.groupby("segment").apply(fit_segment_func)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        if categorical_feature_column_names is None:
            categorical_feature_column_names = []
        if numerical_feature_column_names is None:
            numerical_feature_column_names = []

        # Create a unique identifier for each segment
        df = df.copy()
        df["segment"] = df[categorical_feature_column_names].apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )

        predict_segment_func = partial(
            self._predict_segment,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        calibrated_scores_df = df.groupby("segment").apply(predict_segment_func)
        return calibrated_scores_df["calibrated_scores"].sort_index(level=-1).values

    def _fit_segment(
        self,
        df_segment_train: pd.DataFrame,
        prediction_column_name: str,
        label_column_name: str,
        weight_column_name: str | None = None,
        categorical_feature_column_names: list[str] | None = None,
        numerical_feature_column_names: list[str] | None = None,
    ) -> pd.DataFrame:
        # If the current segment contains only one class, we cannot fit a calibrator,
        # we fall back to the IdentityCalibrator, which we don't need to fit.
        if len(df_segment_train[label_column_name].unique()) > 1:
            calibrator = self.calibrator_class(**self.calibrator_kwargs)
            calibrator.fit(
                df_train=df_segment_train,
                prediction_column_name=prediction_column_name,
                label_column_name=label_column_name,
                weight_column_name=weight_column_name,
                categorical_feature_column_names=categorical_feature_column_names,
                numerical_feature_column_names=numerical_feature_column_names,
            )
            self.calibrator_per_segment[df_segment_train.name] = calibrator
        else:
            self.calibrator_per_segment[df_segment_train.name] = IdentityCalibrator()
        return df_segment_train  # return DataFrame to satisfy pandas apply, even though we don't use it

    def _predict_segment(
        self,
        df_segment: pd.DataFrame,
        prediction_column_name: str,
        categorical_feature_column_names: list[str],
        numerical_feature_column_names: list[str],
    ) -> pd.DataFrame:
        # Handle edge case of unseen segment
        if df_segment.name not in self.calibrator_per_segment:
            self.calibrator_per_segment[df_segment.name] = IdentityCalibrator()
        df_segment["calibrated_scores"] = self.calibrator_per_segment[
            df_segment.name
        ].predict(
            df=df_segment,
            prediction_column_name=prediction_column_name,
            categorical_feature_column_names=categorical_feature_column_names,
            numerical_feature_column_names=numerical_feature_column_names,
        )
        return df_segment
