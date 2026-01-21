# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import time
import warnings

import numpy as np
import pandas as pd

from configs.constants import MCGRAD_NAME
from mcb_algorithms.mcb import MulticalibrationPredictor
from metrics import subgroup_metrics


class Experiment:
    MCB_ALGO_KEY = "mcb_algorithm"
    MCB_ALGO_PARAMS_KEY = "mcb_algorithm_params"
    SET_NAME_KEY = "set_name"
    SEED_KEY = "seed"
    FIT_TIME_KEY = "fit_time"
    NUM_ROUNDS_KEY = "num_rounds"
    LIGHTGBM_PARAMS_KEY = "lightgbm_params"

    def __init__(
        self,
        dataset,
        model,
        calib_frac,
        calib_train_overlap=0,
        calib_seed=50,
        results_storage_path="",
    ):
        """
        Parameters
            :dataset: Dataset object
            :model: Model object
            :calib_frac: float, fraction of calibration split
            :calib_train_overlap: float, fraction of train set to include in calibration set
            :calib_seed: int, seed for splitting calibration set
        """
        self.dataset = dataset
        self.model = model
        self.calib_frac = calib_frac
        self.calib_train_overlap = calib_train_overlap
        self.calib_seed = calib_seed
        self.mcb_models = []
        self.logger = None
        self.wandb = False
        self.results_storage_path = results_storage_path

        if self.calib_frac > 0 or self.calib_train_overlap > 0:
            (
                self.X_train,
                self.y_train,
                self.groups_train,
                self.X_calib,
                self.y_calib,
                self.groups_calib,
                self.df_calib,
            ) = self.dataset.train_calibration_split(
                self.calib_frac, train_overlap=calib_train_overlap, seed=calib_seed
            )

        else:
            self.X_train, self.y_train, self.groups_train = (
                self.dataset.X_train,
                self.dataset.y_train,
                self.dataset.groups_train,
            )

        self.X_test, self.y_test, self.groups_test = (
            self.dataset.X_test,
            self.dataset.y_test,
            self.dataset.groups_test,
        )
        self.X_val, self.y_val, self.groups_val, self.df_val = (
            self.dataset.X_val,
            self.dataset.y_val,
            self.dataset.groups_val,
            self.dataset.df_val,
        )

        # Print a nicely formatted table with
        # train, calibration, validation, and test split sizes
        print(f"\n{'Split':<15}{'Size':<10}{'Fraction of 1s':<15}")
        print(
            f"{'Train':<15}{len(self.y_train):<10}{np.mean(self.y_train) if len(self.y_train) > 0 else 0:<15.2%}"
        )
        if self.calib_frac > 0:
            print(
                f"{'Calibration':<15}{len(self.y_calib):<10}{np.mean(self.y_calib):<15.2%}"
            )
        print(f"{'Validation':<15}{len(self.y_val):<10}{np.mean(self.y_val):<15.2%}")
        print(f"{'Test':<15}{len(self.y_test):<10}{np.mean(self.y_test):<15.2%}")
        # include the total length
        print(
            f"{'Total':<15}{len(self.dataset.y):<10}{np.mean(self.dataset.y):<15.2%}\n"
        )

    def train_model(self):
        print(f"Training {self.model.name} on train split")
        # train model on train split, calibrate on calib split with mcb
        # if calib_frac == 1.0, we cannot train
        if self.calib_frac >= 1.0:
            return
        self.model.train(
            self.X_train,
            self.y_train,
            self.groups_train,
            self.X_val,
            self.y_val,
            self.groups_val,
        )

    def multicalibrate_multiple(self, config_list):
        """
        Multicalibrate predictor using multiple algorithms and parameters.

        Params:
            config_list: list of dicts, each containing 'type' and 'params' keys
                        (see configs/constants.py for examples)
        """
        for alg in config_list:
            alg_type = alg["type"]
            params_list = alg["params"]
            for params in params_list:
                self.multicalibrate(alg_type=alg_type, params=params)

    def multicalibrate(self, alg_type, params):
        """
        Multicalibrate predictor using the specified algorithm and parameters.

        Params:
            alg_type: str, the type of algorithm to use for multicalibration
            params: dict, the parameters to use for multicalibration
        """
        if len(self.X_calib) == 0:
            raise ValueError("No calibration set available for postprocessing.")

        print("Multicalibrating model on calib split")
        print(f"Algorithm-type: {alg_type}, Params: {params}")
        # calibrate model on calib ssplit with mcb
        mcbp = MulticalibrationPredictor(alg_type, params)

        # Get probability of positive class
        confs_calib, logits_calib = self.model.predict_proba(
            self.X_calib, with_logits=True
        )

        # pass in confidence corresponding to correct class
        # mcb algorithms which require logits will use logits_calib
        model_properties_to_track = {}
        fit_start = time.time()
        if alg_type in ["Temp"]:
            mcbp.fit(
                confs=logits_calib, labels=self.y_calib, subgroups=self.groups_calib
            )
        elif alg_type == MCGRAD_NAME:
            confs_val, logits_val = self.model.predict_proba(
                self.X_val, with_logits=True
            )
            mcbp.fit(
                confs=logits_calib,
                confs_val=confs_val,
                labels=self.y_calib,
                labels_val=self.y_val,
                subgroups=self.groups_calib,
                subgroups_val=self.groups_val,
                df=self.df_calib,
                df_val=self.df_val,
                categorical_features=self.dataset.categorical_features,
                numerical_features=self.dataset.numerical_features,
            )
            model_properties_to_track[self.NUM_ROUNDS_KEY] = len(mcbp.mcbp.mcgrad.mr)
            model_properties_to_track[self.LIGHTGBM_PARAMS_KEY] = (
                mcbp.mcbp.mcgrad.lightgbm_params
            )
        else:
            mcbp.fit(
                confs=confs_calib, labels=self.y_calib, subgroups=self.groups_calib
            )
        fit_time = time.time() - fit_start
        model_properties_to_track[self.FIT_TIME_KEY] = fit_time
        self.mcb_models.append([mcbp, alg_type, params, model_properties_to_track])

    def evaluate_val(self, with_rel_diagram=False):
        self.evaluate_model(
            self.X_val,
            self.y_val,
            self.groups_val,
            "validation",
            with_rel_diagram,
            self.dataset.df_val,
            self.dataset.categorical_features,
            self.dataset.numerical_features,
        )

    def evaluate_test(self, with_rel_diagram=False):
        self.evaluate_model(
            self.X_test,
            self.y_test,
            self.groups_test,
            "test",
            with_rel_diagram,
            self.dataset.df_test,
            self.dataset.categorical_features,
            self.dataset.numerical_features,
        )

    def evaluate_train(self, with_rel_diagram=False):
        self.evaluate_model(
            self.X_train,
            self.y_train,
            self.groups_train,
            "train",
            with_rel_diagram,
            self.dataset.df_train,
            self.dataset.categorical_features,
            self.dataset.numerical_features,
        )

    def evaluate_calib(self, with_rel_diagram=False):
        if len(self.X_calib) == 0:
            raise ValueError("No calibration set available for evaluation.")

        # warn if calib_train_overlap > 0
        if self.calib_train_overlap > 0:
            print(
                f"Calibration split includes {self.calib_train_overlap:.2%} of train set"
            )

        self.evaluate_model(
            self.X_calib,
            self.y_calib,
            self.groups_calib,
            "calibration",
            with_rel_diagram,
            self.dataset.df_calib,
            self.dataset.categorical_features,
            self.dataset.numerical_features,
        )

    def _metrics_dict_to_df(self, data):

        # Extract common fields
        algorithm = data[self.MCB_ALGO_KEY]
        algorithm_params = data[self.MCB_ALGO_PARAMS_KEY]
        set_name = data[self.SET_NAME_KEY]
        seed = data[self.SEED_KEY]
        fit_time = data[self.FIT_TIME_KEY]
        num_rounds = data[self.NUM_ROUNDS_KEY]
        lightgbm_params = data[self.LIGHTGBM_PARAMS_KEY]

        # Extract group entries (filter out non-integer keys)
        group_rows = [
            {
                "group": k,
                self.MCB_ALGO_KEY: algorithm,
                self.MCB_ALGO_PARAMS_KEY: algorithm_params,
                self.SET_NAME_KEY: set_name,
                self.SEED_KEY: seed,
                self.FIT_TIME_KEY: fit_time,
                self.NUM_ROUNDS_KEY: num_rounds,
                self.LIGHTGBM_PARAMS_KEY: lightgbm_params,
                **v,
            }
            for k, v in data.items()
            if isinstance(k, int) or k in ["max", "min", "mean", "agg"]
        ]

        return pd.DataFrame(group_rows)

    def save_metrics(self, dicts, dataset_split_name):
        df = (
            pd.concat([self._metrics_dict_to_df(d) for d in dicts])
            .reset_index(drop=True)
            .assign(
                dataset=self.dataset.name,
                model=self.model.name,
            )
        )
        fname = f"dataset={self.dataset.name}_model={self.model.name}_seed={self.dataset.val_split_seed}_split={dataset_split_name}.pkl"
        outpath = pathlib.Path(self.results_storage_path) / pathlib.Path(fname)
        pathlib.Path(self.results_storage_path).mkdir(parents=True, exist_ok=True)
        df.to_pickle(outpath)

    def evaluate_model(
        self,
        X,
        y,
        groups,
        dataset_split_name,
        with_rel_diagram=False,
        df=None,
        categorical_columns=None,
        numerical_columns=None,
    ):
        # evaluate orig model and mcb model on the given dataset split
        preds = self.model.predict(X)
        (confs, logits) = self.model.predict_proba(X, with_logits=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original_model_metrics_val = subgroup_metrics(
                groups, y, confs, preds, df, categorical_columns, numerical_columns
            )

        # print_metrics(original_model_metrics_val, algorithm=self.model.name, split=dataset_split_name)
        original_model_metrics_val[self.MCB_ALGO_KEY] = None
        original_model_metrics_val[self.MCB_ALGO_PARAMS_KEY] = None
        original_model_metrics_val[self.SET_NAME_KEY] = dataset_split_name
        original_model_metrics_val[self.SEED_KEY] = self.dataset.val_split_seed
        original_model_metrics_val[self.FIT_TIME_KEY] = np.nan
        original_model_metrics_val[self.NUM_ROUNDS_KEY] = np.nan
        original_model_metrics_val[self.LIGHTGBM_PARAMS_KEY] = np.nan
        all_metrics = [original_model_metrics_val]

        for mcbp, alg_type, mcb_params, mcb_properties in self.mcb_models:
            # predict and evaluate for each mcb model we have trained
            # temp scaling needs logits, others need confs
            if alg_type == "Temp":
                mcb_confs = mcbp.batch_predict(logits, groups, df=df)
            elif alg_type == MCGRAD_NAME:
                mcb_confs = mcbp.batch_predict(
                    logits,
                    groups,
                    df=df,
                    categorical_features=categorical_columns,
                    numerical_features=numerical_columns,
                )
            else:
                mcb_confs = mcbp.batch_predict(confs, groups, df=df)
            mcb_preds = np.round(mcb_confs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mcb_metrics = subgroup_metrics(
                    groups,
                    y,
                    mcb_confs,
                    mcb_preds,
                    df,
                    categorical_columns,
                    numerical_columns,
                )

            mcb_metrics[self.MCB_ALGO_KEY] = alg_type
            mcb_metrics[self.MCB_ALGO_PARAMS_KEY] = mcb_params
            mcb_metrics[self.SET_NAME_KEY] = dataset_split_name
            mcb_metrics[self.SEED_KEY] = self.dataset.val_split_seed
            mcb_metrics[self.FIT_TIME_KEY] = mcb_properties.get(
                self.FIT_TIME_KEY, np.nan
            )
            mcb_metrics[self.NUM_ROUNDS_KEY] = mcb_properties.get(
                self.NUM_ROUNDS_KEY, np.nan
            )
            mcb_metrics[self.LIGHTGBM_PARAMS_KEY] = mcb_properties.get(
                self.LIGHTGBM_PARAMS_KEY, np.nan
            )
            all_metrics.append(mcb_metrics)

        # dump metric results to file
        self.save_metrics(all_metrics, dataset_split_name)
