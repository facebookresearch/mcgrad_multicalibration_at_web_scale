# pyre-unsafe
import ast
import hashlib
import logging
import math
import warnings
from collections.abc import Collection, Iterable
from typing import Any, Protocol

import numpy as np

import pandas as pd
from scipy import stats
from scipy.optimize._linesearch import LineSearchWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


def unshrink(y: np.ndarray, logits: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        w = np.ones_like(y)
    logits = logits.reshape(-1, 1)

    primary_solver = LogisticRegression(
        fit_intercept=False, solver="newton-cg", penalty=None
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        primary_solver.fit(logits, y, sample_weight=w)
    for rec_warn in recorded_warnings:
        if isinstance(rec_warn.message, LineSearchWarning):
            logger.info(
                f"Line search warning (unshrink): {str(rec_warn.message)}. Solution is approximately optimal - no ideal step size for the gradient descent update can be found. These warnings are generally harmless, see D69983734 for details."
            )
        else:
            logger.debug(rec_warn)
            warnings.warn_explicit(
                message=str(rec_warn.message),
                category=rec_warn.category,
                filename=rec_warn.filename,
                lineno=rec_warn.lineno,
                source=rec_warn.source,
            )

    # Return result if logistic regression with Newton-CG converged to a solution, if no try LBFGS.
    # pyre-ignore, coef_ is available after `fit()` has been called
    if not np.isnan(primary_solver.coef_).any():
        return primary_solver.coef_[0][0]

    fallback_solver = LogisticRegression(
        fit_intercept=False, solver="lbfgs", penalty=None
    )
    fallback_solver.fit(logits, y, sample_weight=w)
    if not np.isnan(fallback_solver.coef_).any():
        return fallback_solver.coef_[0][0]

    # If both solvers fail, return default value. Not disastrous, but requires GBDT to do more heavy-lifting.
    return 1


def logistic(logits: float) -> float:
    # Numerically stable sigmoid - Computational trick to avoid overflow/underflow
    if logits >= 0:
        return 1.0 / (1.0 + math.exp(-logits))
    else:
        return math.exp(logits) / (1.0 + math.exp(logits))


logistic_vectorized = np.vectorize(logistic)


def logit(probs: np.ndarray, epsilon=1e-20) -> np.ndarray:
    return np.log((probs + epsilon) / (1 - probs + epsilon))


def absolute_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimate - reference)


def proportional_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimate - reference) / reference


class BinningMethodInterface(Protocol):
    def __call__(
        self,
        predicted_scores: np.ndarray,
        num_bins: int,
        epsilon: float = 1e-8,
    ) -> np.ndarray: ...


def make_equispaced_bins(
    predicted_scores: np.ndarray,
    num_bins: int,
    epsilon: float = 1e-8,
    set_range_to_zero_one: bool = True,
) -> np.ndarray:
    lower_bound = min(0, predicted_scores.min())
    upper_bound = max(1, predicted_scores.max())

    bins = (
        np.linspace(0, 1, num_bins + 1)
        if set_range_to_zero_one
        else np.linspace(predicted_scores.min(), predicted_scores.max(), num_bins + 1)
    )
    bins[0] = (
        lower_bound - epsilon
        if set_range_to_zero_one
        else predicted_scores.min() - epsilon
    )
    bins[-1] = (
        upper_bound + epsilon
        if set_range_to_zero_one
        else predicted_scores.max() + epsilon
    )
    return bins


def make_equisized_bins(
    predicted_scores: np.ndarray,
    num_bins: int,
    epsilon: float = 1e-8,
    **kwargs: Any,
) -> np.ndarray:
    upper_bound = max(1, predicted_scores.max())
    bins = np.array(
        sorted(
            pd.qcut(
                predicted_scores, q=num_bins, duplicates="drop"
            ).categories.left.tolist()
        )
        + [upper_bound + epsilon]
    )
    return bins


def _compute_prop_positive_ci(
    label_weighted: Iterable,
    sample_weight: Iterable,
    assigned_bin: Iterable,
    bins: Collection[Any],
    alpha,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the confidence interval for the proportion in each bin using the Clopper-Pearson method
    (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval)
    """

    label_binned_preds = pd.DataFrame(
        {
            "label_weighted": label_weighted,
            "sample_weight": sample_weight,
            "assigned_bin": assigned_bin,
        }
    )

    bin_stats = (
        label_binned_preds[["assigned_bin", "label_weighted", "sample_weight"]]
        .groupby("assigned_bin")
        .agg(["sum"])
    )

    def _row_ci(row):
        n_positive = row["label_weighted"]["sum"]
        n = row["sample_weight"]["sum"]
        lower = stats.beta.ppf(alpha / 2, n_positive, n - n_positive + 1)
        upper = stats.beta.ppf(1 - alpha / 2, n_positive + 1, n - n_positive)
        return pd.Series({"lower": lower, "upper": upper})

    cis = bin_stats.apply(_row_ci, axis=1)
    # pyre-ignore, index takes some complicated Union of types. Collection[Any] is fine
    all_cis = pd.DataFrame(index=bins, columns=["lower", "upper"], data=np.nan)
    all_cis.update(cis)

    return all_cis.lower.values, all_cis.upper.values


def positive_label_proportion(
    labels: np.ndarray,
    predictions: np.ndarray,
    bins: np.ndarray,
    sample_weight: np.ndarray | None = None,
    alpha: float = 0.05,
    use_weights_in_sample_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the proportion of positive labels in each bin.
    :param labels: array of labels
    :param predictions: array of predictions
    :param bins: array of bin boundaries
    :param sample_weight: array of weights for each instance. If None, then all instances are considered to have weight 1
    :param alpha: 1-alpha is the confidence level of the CI
    :param use_weights_in_sample_size: the effective sample size of this dataset depends on how the weights in this dataset were generated. This
        should be set to True in the case of Option 1 below and set to False in the case of Option 2 below.
        Option 1. it could be the case that there once existed a dataset that for example had 10 rows with score 0.6 and label 1 and 100 rows
        with score 0.1 and label 0 that has been turned into an aggregated dataset with one row with weight 10 and score 0.6 and label 1 with
        weight 10 and a row with score 0.1 and label 0 with weight 100.
        Option 2. it could also be the case that weights merely reflects the inverse of the sampling probability of the instance.

    :return: array of proportions
    """
    assert not np.any(np.isnan(predictions)), "predictions must not contain NaNs"
    sample_weight = sample_weight if sample_weight is not None else np.ones_like(labels)

    label_binned_preds = pd.DataFrame(
        {
            "label_weighted": labels * sample_weight,
            "score_weighted": predictions * sample_weight,
            "n_sample_weighted": sample_weight,
            "n_sample_unweighted": np.ones_like(labels),
            "assigned_bin": bins[np.digitize(predictions, bins)],
        }
    )

    bin_means = (
        label_binned_preds[
            [
                "assigned_bin",
                "label_weighted",
                "score_weighted",
                "n_sample_weighted",
                "n_sample_unweighted",
            ]
        ]
        .groupby("assigned_bin")
        .sum()
    )

    # Compute average label
    bin_means["label_proportion"] = (
        bin_means["label_weighted"] / bin_means["n_sample_weighted"]
    )

    # Compute average score
    bin_means["score_average"] = (
        bin_means["score_weighted"] / bin_means["n_sample_weighted"]
    )

    # Compute confidence intervals
    def _row_ci(row):
        if use_weights_in_sample_size:
            n_positive = row["label_weighted"]
            n = row["n_sample_weighted"]
        else:
            n = row["n_sample_unweighted"]
            n_positive = int(row["label_proportion"] * n)

        lower = stats.beta.ppf(alpha / 2, n_positive, n - n_positive + 1)
        upper = stats.beta.ppf(1 - alpha / 2, n_positive + 1, n - n_positive)
        return pd.Series({"lower": lower, "upper": upper})

    cis = bin_means.apply(_row_ci, axis=1)

    # Rather than using bin_means directly, we create a new DataFrame and update, to
    # ensure consistent shape of the output array when there exists bins without predictions.
    prop_pos_label = pd.DataFrame(
        index=bins,
        columns=["label_proportion", "score_average", "lower", "upper"],
        data=np.nan,
    )
    prop_pos_label.update(bin_means["label_proportion"])
    prop_pos_label.update(bin_means["score_average"])
    prop_pos_label.update(cis["lower"])
    prop_pos_label.update(cis["upper"])

    return (
        prop_pos_label.label_proportion.values,
        prop_pos_label.lower.values,
        prop_pos_label.upper.values,
        prop_pos_label.score_average.values,
    )


def geometric_mean(x: np.ndarray) -> float:
    """
    Computes the geometric mean of an array of numbers. If any of the numbers are 0, then the geometric mean is 0.
    The exp-log trick is used to avoid underflow/overflow problems when computing the product of many numbers.

    :param x: array of numbers
    :return: geometric mean of the array
    """
    return np.exp(np.log(x).mean())


def make_unjoined(x: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
    """
    In the Ads organization, it is common to work with a data format that is commonly referred to as
    'unjoined'. This means that there is always a row with a negative label and there will be a
    second row with positive label if there is a conversion. This means that we will have 2 rows
    for the same impression in case that that impression resulted in a conversion.

    This method takes a regular dataset (one row per impression) and returns an unjoined
    version of that dataset.

    :param x: array of features
    :param y: array of labels
    :return: tuple of arrays (x_unjoined, y_unjoined)
    """
    assert x.shape[0] == y.shape[0], "x and y must have the same number of instances"
    # Find the indices where y is positive, create duplicates for those instances
    positive_indices = np.where(y == 1)[0]
    unjoined_x = np.concatenate([x, x[positive_indices]])
    # Create an array of artificial negatives
    artificial_negatives = np.zeros(len(positive_indices), dtype=y.dtype)
    unjoined_y = np.concatenate([y, artificial_negatives])
    return unjoined_x, unjoined_y


class OrdinalEncoderWithUnknownSupport(OrdinalEncoder):
    """
    Extends the scikit-learn OrdinalEncoder by addressing the issue that the transform method
    of the OrdinalEncoder raises an error if any of the categorical features contains categories
    that were never observed when fitting the encoder. This encoder assigns value -1 to all
    unknown categories.

    Note: this is only needed in scikit-learn version 0.22. In later versions, scikit-learn's
    OrdinalEncoder supports unknown categories using the handle_unknown and unknown_value arguments.
    """

    def __init__(self, categories="auto", dtype=np.float64):
        super().__init__(categories=categories, dtype=dtype)
        self._category_map = {}

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else X
        super().fit(X, y)
        for i, category in enumerate(self.categories_):
            self._category_map[i] = {
                value: index for index, value in enumerate(category)
            }
        return self

    def transform(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        if not self._category_map:
            raise ValueError("The fit method should be called before transform.")
        X_transformed = np.zeros_like(X, dtype=int)
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                X_transformed[j, i] = self._category_map[i].get(X[j, i], -1)
        return X_transformed

    def serialize(self) -> str:
        return str(self._category_map)

    @classmethod
    def deserialize(cls, encoder_str) -> "OrdinalEncoderWithUnknownSupport":
        enc = cls()
        enc._category_map = ast.literal_eval(encoder_str)
        return enc


def hash_categorical_feature(categorical_feature: str) -> int:
    """
    This implements the categorical feature encoding scheme that @Jiayuanm implemented in Hack: D56290586
    It uses the last two bytes of SHA256 for categorical features.

    Sometimes we need to perform the equivalent encoding in Presto, which can be done with:
        FROM_BASE(SUBSTR(TO_HEX(SHA256(CAST(categorical_feature AS VARBINARY))), -4), 16)

    This Daiquery link shows that that generates equivalent output for all test cases of test_hash_categorical_feature in test_base.py:
        https://fburl.com/daiquery/neiirvol
    """
    signature = hashlib.sha256(categorical_feature.encode("utf-8")).digest().hex()
    last_four_hex_chars = signature[-4:]
    return int(last_four_hex_chars, 16)


def rank_log_discount(n_samples: int, log_base: int = 2) -> np.ndarray:
    """
    Rank log discount function used for the rank metrics DCG and NDCG.
    More information about the function here: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain.

    :param n_samples: number of samples
    :param log_base: base of the logarithm
    :return: array of size n_samples with the discount factor for each sample
    """
    return 1 / (np.log(np.arange(n_samples) + 2) / np.log(log_base))


def rank_no_discount(num_samples: int) -> np.ndarray:
    """
    Rank discount function used for the rank metrics DCG and NDCG.
    Returns uniform discount factor of 1 for all samples.

    :param num_samples: number of samples
    :return: array of size num_samples with the value of 1 as the discount factor for each sample
    """
    return np.ones(num_samples)


class TrainTestSplitWrapper:
    def __init__(
        self,
        test_size: float = 0.4,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """
        Customized train-test split class that allows to specify the test size (fraction).
        This is useful for the case where we want to have a single split with given test size, rather than doing k-fold crossvalidation.
        :param test_size: size of the test set as a fraction of the total size of the dataset.
        :param shuffle: whether to shuffle the data before splitting;
        :param random_state: random state;
        """
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=self.test_size,
            shuffle=self.shuffle,
            stratify=y,
            random_state=self.random_state,
        )
        yield train_idx, val_idx


def convert_arrow_columns_to_numpy(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if isinstance(df[col].values, pd.core.arrays.ArrowExtensionArray):
            df[col] = df[col].to_numpy()
    return df


# Check if the values in the columns are within the valid range
def check_range(series, precision_type):
    precision_limits = {
        "float16": (np.finfo(np.float16).min, np.finfo(np.float16).max),
        "float32": (np.finfo(np.float32).min, np.finfo(np.float32).max),
        "float64": (np.finfo(np.float64).min, np.finfo(np.float64).max),
    }

    min_val, max_val = precision_limits[precision_type]
    return not (
        (series.min() < min_val)
        or (series.max() > max_val)
        or (series.sum() > math.sqrt(max_val))
    )
