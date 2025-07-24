import numpy as np
from mcb_algorithms.Calibration.Isotonic import IsotonicRegression
from mcb_algorithms.Calibration.Platt import PlattScaling
from mcb_algorithms.Calibration.Temperature import TemperatureScaling
from mcb_algorithms.CAS.CASMCBoost import CASMCBoostAlgorithm
from mcb_algorithms.HJZ.hjz import HJZAlgorithm
from mcb_algorithms.HKRR.hkrr import HKRRAlgorithm
from tqdm import trange
from configs.constants import MCBOOST_NAME

## MCB algorithm
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
        elif algorithm == "HJZ":
            self.mcbp = HJZAlgorithm(params)
        elif algorithm == "Platt":
            self.mcbp = PlattScaling(params)
        elif algorithm == "Temp":
            self.mcbp = TemperatureScaling(params)
        elif algorithm == "Isotonic":
            self.mcbp = IsotonicRegression(params)
        elif algorithm == MCBOOST_NAME:
            self.mcbp = CASMCBoostAlgorithm(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def fit(self, confs, labels, subgroups, confs_val=None, labels_val=None, subgroups_val=None, df_val=None, df=None,
            categorical_features=None, numerical_features=None):
        """
        Returns vector of confidences on calibration set.

        HKRR: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')

        Parameters:
            confs: Array of prediction scores
            labels: Array of true labels
            subgroups: List of subgroups
            df: Optional dataframe with additional features (required for CASMCBoost with alltogether variant)
        """
        # Only pass df parameter to algorithms that support it (currently only CASMCBoost)
        if self.algorithm == MCBOOST_NAME:
            self.mcbp.fit(confs=confs, labels=labels, subgroups=subgroups, df=df, confs_val=confs_val,
                          labels_val=labels_val, subgroups_val=subgroups_val, df_val=df_val,
                          categorical_features=categorical_features, numerical_features=numerical_features)
        else:
            self.mcbp.fit(confs, labels, subgroups)

    def batch_predict(self, f_xs, groups, df=None, categorical_features=None, numerical_features=None):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None

        Parameters:
            f_xs: Array of prediction scores
            groups: List of groups
            df: Optional dataframe with additional features (required for CASMCBoost with alltogether variant)
        """
        # Only pass df parameter to algorithms that support it (currently only CASMCBoost)
        if self.algorithm == MCBOOST_NAME:
            return self.mcbp.batch_predict(f_xs, groups, df, categorical_features=categorical_features, numerical_features=numerical_features)
        else:
            return self.mcbp.batch_predict(f_xs, groups)
