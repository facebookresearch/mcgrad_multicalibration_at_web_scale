import pandas as pd
import numpy as np
import logging
import sys
import math
from numpy import typing as npt
import functools
from typing import Protocol
from segmentation import get_segment_masks

logger = logging.getLogger(__name__)


DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD: str = "kuiper_standard_deviation"
DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE: int = 3
DEFAULT_MULTI_KUIPER_MIN_DEPTH: int = 0
DEFAULT_MULTI_KUIPER_MAX_DEPTH: int = 3
DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT: int = 10
DEFAULT_MULTI_KUIPER_GLOBAL_NORMALIZATION: str = "prevalence_adjusted"
DEFAULT_MULTI_KUIPER_N_SEGMENTS: int | None = 1000
DEFAULT_PRECISION_DTYPE = np.float64


class KuiperNormalizationInterface(Protocol):
    def __call__(
        self,
        predicted_scores: npt.NDArray,
        labels: npt.NDArray | None,
        sample_weight: npt.NDArray | None,
        segments: npt.NDArray | None,
        precision_dtype: np.float16 | np.float32 | np.float64,
    ) -> npt.NDArray: ...


def _normalization_method_assignment(
    method: str | None,
) -> KuiperNormalizationInterface:
    methods = {
        "kuiper_standard_deviation": kuiper_standard_deviation_per_segment,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown normalization method {method}. Available methods are {methods.keys()}"
        )
    return methods[method]
def kuiper_calibration_per_segment(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    normalization_method: str | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    """
    Calculates Kuiper calibration distance between responses and scores.

    For details, see:
    Mark Tygert. (2024, January 10). Conditioning on and controlling for
    variates via cumulative differences: measuring calibration, reliability,
    biases, and other treatment effects. Zenodo.
    https://doi.org/10.5281/zenodo.10481097

    :param labels: Array of binary labels (0 or 1)
    :param predicted_scores: Array of predicted probability scores. (floats between 0 and 1)
    :param sample_weight: Optional array of sample weights (non-negative floats)
    :param normalization_method: Optional function to calculate a normalization constant.
            See for example kuiper_sd or inverse_sqrt_sample_size, methods need to follow the same interface.
    :param segments: Optional array of segments to parallelize the computation of the kuiper calibration distance.
    :param precision_dtype: Optional dtype for the precision of the output. Defaults to np.float64.
    :return: Kuiper calibration distance
    """

    normalization_func = _normalization_method_assignment(normalization_method)

    denominator = normalization_func(
        predicted_scores=predicted_scores,
        labels=labels,
        sample_weight=sample_weight,
        segments=segments,
        precision_dtype=precision_dtype,
    )

    differences = _calculate_cumulative_differences(
        labels, predicted_scores, sample_weight, segments, precision_dtype
    )
    if segments is None:
        differences = differences.reshape(1, -1)

    c_range = np.ptp(differences, axis=1)
    return np.where(
        (denominator == 0) & (c_range != 0),
        np.inf,
        np.where(denominator == 0, 0, c_range / denominator),
    )

def _calculate_cumulative_differences(
    labels: npt.NDArray,
    predicted_scores: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)
        sorted_indices = np.argsort(predicted_scores)
        predicted_scores = predicted_scores[sorted_indices]
        labels = labels[sorted_indices]
        sample_weight = (
            sample_weight[sorted_indices] if sample_weight is not None else None
        )

    if not segments.shape[1] == labels.shape[0] == predicted_scores.shape[0]:
        raise ValueError("Segments must be the same length as labels/predictions.")

    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores) / predicted_scores.shape[0]

    differences = np.empty(
        shape=(np.shape(segments)[0], np.shape(segments)[1] + 1),
        dtype=precision_dtype,
    )
    differences[:, 0] = 0
    weighted_diff = np.multiply((segments * sample_weight), (labels - predicted_scores))
    normalization = (segments * sample_weight).sum(axis=1)[:, np.newaxis]
    normalized_diff = np.divide(weighted_diff, normalization)
    np.cumsum(
        normalized_diff,
        axis=1,
        out=differences[:, 1:],
    )
    differences[np.isnan(differences)] = 0
    return differences

def kuiper_distribution(x: float) -> float:
    """
    Source: https://github.com/facebookresearch/ecevecce/blob/main/codes/dists.py

    Evaluates the cumulative distribution function for the range
    (maximum minus minimum) of the standard Brownian motion on [0, 1].

    :param float x: argument at which to evaluate the cumulative distribution function
                    (must be positive)
    :return: cumulative distribution function evaluated at x
    :rtype: float
    """
    assert (
        x > 0
    ), f"Can only evaluate cumulative Kuiper distribution at positive x, not at {x}"
    # If x goes to infinity, c tends to 1.0
    if x >= 8.26732673:
        return 1.0 - sys.float_info.epsilon

    # Compute the machine precision assuming binary numerical representations.
    eps = sys.float_info.epsilon
    # Determine how many terms to use to attain accuracy eps.
    fact = 4.0 / math.sqrt(2.0 * math.pi) * (1.0 / x + x / math.pi**2)
    kmax = math.ceil(
        1.0 / 2.0 + x / math.pi / math.sqrt(2) * math.sqrt(math.log(fact / eps))
    )

    # Sum the series.
    c = 0.0
    for k in range(kmax):
        kplus = k + 1.0 / 2.0
        c += (8.0 / x**2.0 + 2.0 / kplus**2.0 / math.pi**2.0) * math.exp(
            -2.0 * kplus**2.0 * math.pi**2.0 / x**2.0
        )
    return c


def kuiper_standard_deviation_per_segment(
    predicted_scores: npt.NDArray,
    labels: npt.NDArray | None = None,
    sample_weight: npt.NDArray | None = None,
    segments: npt.NDArray | None = None,
    precision_dtype: np.float16 | np.float32 | np.float64 = DEFAULT_PRECISION_DTYPE,
) -> npt.NDArray:
    if sample_weight is None:
        sample_weight = np.ones_like(predicted_scores)
    if segments is None:
        segments = np.ones(shape=(1, len(predicted_scores)), dtype=np.bool_)

    if segments.shape[1] != predicted_scores.shape[0]:
        raise ValueError("Segments must be the same length as labels/predictions.")

    kuip_std_dev = np.zeros(
        shape=(np.shape(segments)[0],),
        dtype=precision_dtype,
    )
    weighted_segments = segments * np.square(sample_weight)
    variance_preds = predicted_scores * (1 - predicted_scores)
    variance_weighted_segments = np.multiply(weighted_segments, variance_preds).sum(
        axis=1
    )
    normalization_variance = np.square((segments * sample_weight).sum(axis=1))
    np.sqrt(
        np.divide(
            variance_weighted_segments,
            normalization_variance,
        ),
        out=kuip_std_dev,
    )
    kuip_std_dev[np.isnan(kuip_std_dev)] = 0
    return kuip_std_dev


class MulticalibrationError:
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        score_column: str,
        weight_column: str | None = None,
        categorical_segment_columns: list[str] | None = None,
        numerical_segment_columns: list[str] | None = None,
        max_depth: int | None = DEFAULT_MULTI_KUIPER_MAX_DEPTH,
        max_values_per_segment_feature: int = DEFAULT_MULTI_KUIPER_MAX_VALUES_PER_SEGMENT_FEATURE,
        min_samples_per_segment: int = DEFAULT_MULTI_KUIPER_MIN_SAMPLES_PER_SEGMENT,
        sigma_estimation_method: str | None = DEFAULT_MULTI_KUIPER_NORMALIZATION_METHOD,
        max_n_segments: int | None = DEFAULT_MULTI_KUIPER_N_SEGMENTS,
        chunk_size: int = 50,
        precision_dtype: str = "float32",
    ) -> None:
        """
        Calculates the multicalibration error with respect to a set of segments for a given dataset.

        See this wiki for a detailed description of the metric: https://www.internalfb.com/wiki/MCBoost/Measuring_Multicalibration/Methodology_Deep_Dive

        :param df: A pandas DataFrame containing the data.
        :param label_column: The name of the column in `df` that contains the true labels.
        :param score_column: The name of the column in `df` that contains the predicted scores.
        :param weight_column: An optional column in `df` that contains sample weights.
        :param categorical_segment_columns: An optional list of column names in `df` to be used for categorical segmentation.
        :param numerical_segment_columns: An optional list of column names in `df` to be used for numerical segmentation.
        :param max_depth: The maximum depth for segment generation.
        :param max_values_per_segment_feature: The maximum number of unique values per segment feature.
        :param min_samples_per_segment: The minimum number of samples required per segment.
        :param sigma_estimation_method: The method used for sigma estimation.
        :param max_n_segments: The maximum number of segments to generate.
        :param chunk_size: Size of chunks of segments to process per iteration of the algorithm. Larger values improve runtime but increase memory usage (OOM errors are possible).
        :param precision_dtype: The precision type for the metric. Can be 'float16', 'float32', or 'float64'.
        """
        self.label_column = label_column
        self.score_column = score_column
        self.weight_column = weight_column
        self.categorical_segment_columns = categorical_segment_columns
        self.numerical_segment_columns = numerical_segment_columns
        self.max_depth = max_depth
        self.max_values_per_segment_feature = max_values_per_segment_feature
        self.min_samples_per_segment = min_samples_per_segment
        self.estimate_sigma: KuiperNormalizationInterface = (
            _normalization_method_assignment(sigma_estimation_method)
        )
        self.df: pd.DataFrame = df.copy(deep=False)
        self.df.sort_values(by=score_column, inplace=True)
        self.df.reset_index(inplace=True)

        if max_n_segments and chunk_size > max_n_segments:
            logger.warning(
                f"The chunk size {chunk_size} cannot be greater than max number of segments {max_n_segments}. "
                f"Setting speedup chunk size to {max_n_segments}."
            )
            chunk_size = max_n_segments

        self.chunk_size = chunk_size
        self.max_n_segments = max_n_segments

        if precision_dtype not in ["float16", "float32", "float64"]:
            raise ValueError(
                f"Invalid precision type: {precision_dtype}. Must be one of ['float16', 'float32', 'float64']."
            )
        self.precision_dtype: np.float16 | np.float32 | np.float64 = getattr(
            np, precision_dtype
        )

        self.df[self.score_column] = self.df[self.score_column].astype(
            self.precision_dtype
        )
        # if self.weight_column is not None:
        #     if utils.check_range(self.df[self.weight_column], precision_dtype):
        #         self.df[self.weight_column] = self.df[self.weight_column].astype(
        #             self.precision_dtype
        #         )
        #     else:
        #         logger.info(
        #             f"Sample weights are not in range for {precision_dtype}. Keeping their initial type {self.df[self.weight_column].dtype}."
        #         )

        # Motivation for total_number_segments: chunks of segments with less than chunk_size elements are topped up with zeros
        # Such zeros are not needed for the computation of the metric and must be removed (lines: 1548, 1663)
        self.total_number_segments: int = -1  # initialized as -1

    def __str__(self) -> str:
        return f"""{self.mce}% (sigmas={self.mce_sigma_scale}, p={self.p_value}, mde={self.mde})"""

    def __format__(self, format_spec: str) -> str:
        # Use the format specifier to format each attribute
        formatted_mce_relative = format(self.mce, format_spec)
        formatted_p_value = format(self.p_value, format_spec)
        formatted_mde = format(self.mde, format_spec)
        formatted_mce_sigma_scale = format(self.mce_sigma_scale, format_spec)
        return f"""{formatted_mce_relative}% (sigmas={formatted_mce_sigma_scale}, p={formatted_p_value}, mde={formatted_mde})"""

    @functools.cached_property
    def segments(self) -> tuple[npt.NDArray[np.bool_], pd.DataFrame]:
        segments_masks = []
        segments_feature_values = pd.DataFrame(
            columns=["segment_column", "value", "idx_segment"]
        )
        tot_segments: int = 0
        segments_generator = get_segment_masks(
            df=self.df,
            categorical_segment_columns=self.categorical_segment_columns,
            numerical_segment_columns=self.numerical_segment_columns,
            max_depth=self.max_depth,
            max_values_per_segment_feature=self.max_values_per_segment_feature,
            min_samples_per_segment=self.min_samples_per_segment,
            chunk_size=self.chunk_size,
        )
        for (
            segments_chunk_mask,
            size_chunk_mask,
            segment_chunk_feature_values,
        ) in segments_generator:
            if self.max_n_segments is not None and tot_segments >= self.max_n_segments:
                logger.warning(f"Reached max number of segments {self.max_n_segments}")
                break

            segments_masks.append(segments_chunk_mask)
            segments_feature_values = pd.concat(
                [
                    segments_feature_values,
                    segment_chunk_feature_values,
                ],
                ignore_index=True,
            )
            tot_segments += size_chunk_mask

        segments = np.stack(segments_masks, axis=0)
        self.total_number_segments = tot_segments
        return segments, segments_feature_values

    @functools.cached_property
    def segment_indices(self) -> pd.Series:
        segments_2d = self.segments[0].reshape(-1, self.segments[0].shape[2])
        indices = np.argwhere(segments_2d)
        index_series = pd.Series(indices[:, 1], index=indices[:, 0])

        return index_series

    @functools.cached_property
    def segment_ecces_absolute(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        segments = self.segments[0]
        statistics = np.zeros(
            self.total_number_segments,
            dtype=self.precision_dtype,
        )

        for i, segment in enumerate(segments):
            statistics[
                self.chunk_size * i : min(
                    self.chunk_size * (i + 1),
                    self.total_number_segments,
                )
            ] = kuiper_calibration_per_segment(
                labels=self.df[self.label_column].values,
                predicted_scores=self.df[self.score_column].values,
                sample_weight=(
                    None
                    if self.weight_column is None
                    else self.df[self.weight_column].values
                ),
                normalization_method=None,
                segments=segment[: self.total_number_segments - self.chunk_size * i,],
            )
        return statistics

    @functools.cached_property
    def segment_ecces(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        return self.segment_ecces_sigma_scale * self.sigma_0 / self.prevalence * 100

    @functools.cached_property
    def global_ecce(self) -> float:
        return self.segment_ecces[0]

    @functools.cached_property
    def global_ecce_sigma_scale(self) -> float:
        return self.segment_ecces_sigma_scale[0]

    @functools.cached_property
    def global_ecce_p_value(self) -> float:
        return self.segment_p_values[0]

    @functools.cached_property
    def segment_ecces_sigma_scale(
        self,
    ) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        statistics = np.where(
            (self.segment_ecces_absolute != 0) & (self.segment_sigmas == 0),
            np.inf,
            np.where(
                self.segment_sigmas == 0,
                0,
                self.segment_ecces_absolute / self.segment_sigmas,
            ),
        )
        return statistics

    @functools.cached_property
    def segment_p_values(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        kuiper_distribution_vec = np.vectorize(kuiper_distribution)
        p_values = np.ones_like(
            self.segment_ecces_sigma_scale
        ) - kuiper_distribution_vec(self.segment_ecces_sigma_scale)
        return p_values

    @functools.cached_property
    def segment_sigmas(self) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
        segments = self.segments[0]
        sigmas = np.zeros(self.total_number_segments, dtype=self.precision_dtype)
        for i, segment in enumerate(segments):
            sigmas[
                self.chunk_size * i : min(
                    self.chunk_size * (i + 1),
                    self.total_number_segments,
                )
            ] = self.estimate_sigma(
                predicted_scores=self.df[self.score_column].values,
                labels=self.df[self.label_column].values,
                sample_weight=(
                    None
                    if self.weight_column is None
                    else self.df[self.weight_column].values
                ),
                segments=segment[: self.total_number_segments - self.chunk_size * i,],
                precision_dtype=self.precision_dtype,
            )
        return sigmas

    @functools.cached_property
    def sigma_0(self) -> float:
        if "segment_sigmas" in self.__dict__:
            return self.segment_sigmas[0]
        sigma_0 = self.estimate_sigma(
            predicted_scores=self.df[self.score_column].values,
            labels=self.df[self.label_column].values,
            sample_weight=(
                None
                if self.weight_column is None
                else self.df[self.weight_column].values
            ),
            segments=np.ones(shape=(1, len(self.df)), dtype=np.bool_),
            precision_dtype=self.precision_dtype,
        )
        return sigma_0.item()

    @functools.cached_property
    def mce_sigma_scale(self) -> float:
        return np.max(self.segment_ecces_sigma_scale)

    @functools.cached_property
    def mce_absolute(self) -> float:
        return self.mce_sigma_scale * self.sigma_0

    @functools.cached_property
    def prevalence(self) -> float:
        p = (
            (self.df[self.label_column] * self.df[self.weight_column]).sum()
            / (self.df[self.weight_column].sum())
            if self.weight_column is not None
            else self.df[self.label_column].mean()
        )
        return min(p, 1 - p)

    @functools.cached_property
    def mce(self) -> float:
        return self.mce_absolute / self.prevalence * 100

    @functools.cached_property
    def p_value(self) -> float:
        if "segment_p_values" in self.__dict__:
            return np.min(self.segment_p_values)
        return 1 - kuiper_distribution(self.mce_sigma_scale)

    @functools.cached_property
    def mde(self) -> float:
        # This is a rough, conservative approximation of the MDE. We divide by the prevalence
        # and multiply by 100 to get the MDE in the same unit as the MCE metric
        return 5 * self.sigma_0 / self.prevalence * 100
