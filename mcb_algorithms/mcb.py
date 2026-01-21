# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from configs.constants import MCGRAD_NAME
from mcb_algorithms.Calibration.Isotonic import IsotonicRegression
from mcb_algorithms.Calibration.Platt import PlattScaling
from mcb_algorithms.Calibration.Temperature import TemperatureScaling
from mcb_algorithms.HKRR.hkrr import HKRRAlgorithm
from .mcgrad_wrapper import MCGradWrapper


class MulticalibrationPredictor:
    """
    General Multicalibration Predictor class.
    """

    def __init__(self, algorithm, params):
        """
        Initialize Multicalibration Predictor.
        """
        self.algorithm = algorithm
        self.params = params
        if algorithm == "HKRR":
            self.mcbp = HKRRAlgorithm(params)
        elif algorithm == "Platt":
            self.mcbp = PlattScaling(params)
        elif algorithm == "Temp":
            self.mcbp = TemperatureScaling(params)
        elif algorithm == "Isotonic":
            self.mcbp = IsotonicRegression(params)
        elif algorithm == MCGRAD_NAME:
            self.mcbp = MCGradWrapper(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def fit(
        self,
        confs,
        labels,
        subgroups,
        confs_val=None,
        labels_val=None,
        subgroups_val=None,
        df_val=None,
        df=None,
        categorical_features=None,
        numerical_features=None,
    ):
        """
        Returns vector of confidences on calibration set.

        HKRR: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')

        Parameters:
            confs: Array of prediction scores
            labels: Array of true labels
            subgroups: List of subgroups
            df: Optional dataframe with additional features (required for MCGradWrapper with alltogether variant)
        """
        # Only pass df parameter to algorithms that support it (currently only MCGradWrapper)
        if self.algorithm == MCGRAD_NAME:
            self.mcbp.fit(
                confs=confs,
                labels=labels,
                subgroups=subgroups,
                df=df,
                confs_val=confs_val,
                labels_val=labels_val,
                subgroups_val=subgroups_val,
                df_val=df_val,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
        else:
            self.mcbp.fit(confs, labels, subgroups)

    def batch_predict(
        self, f_xs, groups, df=None, categorical_features=None, numerical_features=None
    ):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None

        Parameters:
            f_xs: Array of prediction scores
            groups: List of groups
            df: Optional dataframe with additional features (required for MCGradWrapper with alltogether variant)
        """
        # Only pass df parameter to algorithms that support it (currently only MCGradWrapper)
        if self.algorithm == MCGRAD_NAME:
            return self.mcbp.batch_predict(
                f_xs,
                groups,
                df,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
            )
        else:
            return self.mcbp.batch_predict(f_xs, groups)
