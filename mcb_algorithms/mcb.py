import numpy as np
from mcb_algorithms.Calibration.Isotonic import IsotonicRegression
from mcb_algorithms.Calibration.Platt import PlattScaling
from mcb_algorithms.Calibration.Temperature import TemperatureScaling
from mcb_algorithms.CAS.CASMCBoost import CASMCBoostAlgorithm
from mcb_algorithms.HJZ.hjz import HJZAlgorithm
from mcb_algorithms.HKRR.hkrr import HKRRAlgorithm
from tqdm import trange


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
        elif algorithm == "CASMCBoost":
            self.mcbp = CASMCBoostAlgorithm(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def fit(self, confs, labels, subgroups, df=None):
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
        if self.algorithm == "CASMCBoost":
            self.mcbp.fit(confs, labels, subgroups, df)
        else:
            self.mcbp.fit(confs, labels, subgroups)

    def batch_predict(self, f_xs, groups, df=None):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None

        Parameters:
            f_xs: Array of prediction scores
            groups: List of groups
            df: Optional dataframe with additional features (required for CASMCBoost with alltogether variant)
        """
        # Only pass df parameter to algorithms that support it (currently only CASMCBoost)
        if self.algorithm == "CASMCBoost":
            return self.mcbp.batch_predict(f_xs, groups, df)
        else:
            return self.mcbp.batch_predict(f_xs, groups)
