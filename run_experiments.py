import itertools

import logging
import sys

from configs.constants import (
    CALIB_ALGS_DEFAULT,
    get_mcgrad_configs,
    HKRR_DEFAULT,
    SPLIT_DEFAULT,
)
from configs.hyperparameters import get_hyperparameters
from Dataset import Dataset
from Experiment import Experiment
from Model import Model


logger = logging.getLogger(__name__)

RESULTS_DIR = "results/"
BASE_MODELS = [
    # 'SVM',
    "LogisticRegression",
    # 'NaiveBayes',
    # 'RandomForest'
]

DATASETS = [
    "ACSIncome",
    "CreditDefault",
    "BankMarketing",
    "MEPS",
    "HMDA",
    "acs_employment_all_states",
    "acs_health_insurance_all_states",
    "acs_public_health_insurance_all_states",
    "acs_travel_time_all_states",
    "acs_mobility_all_states",
    "acs_income_all_states",
]


def data_reuse_experiment(
    model_name, dataset, seed, mcb_params, results_storage_path=""
):
    # set constants for the experiment
    calib_frac = 0
    calib_train_overlap = 1.0
    groups_collection = "default"

    hyp = get_hyperparameters(model_name, dataset, calib_frac)
    config = {
        "model": model_name,  # model name
        "dataset": dataset,  # dataset name
        "group_collection": groups_collection,  # group collection
        "calib_frac": calib_frac,  # calibration fraction
        "calib_train_overlap": calib_train_overlap,  # calibration train overlap
        "val_split_seed": seed,  # seed for validation split
        "split": SPLIT_DEFAULT,  # default split
        "mcb": mcb_params,  # just to keep track of mcb algorithm
        "val_save_epoch": 0,  # save model every epoch
        "val_eval_epoch": 1,  # evaluate model every epoch
        **hyp,
    }

    dataset_obj = Dataset(
        dataset, val_split_seed=config["val_split_seed"], groups=groups_collection
    )
    model = Model(model_name, config=config, SAVE_DIR=None)
    experiment = Experiment(
        dataset_obj,
        model,
        calib_frac=config["calib_frac"],
        # pyre-ignore
        calib_train_overlap=calib_train_overlap,
        results_storage_path=results_storage_path,
    )
    logger.info(f"Storing results at {experiment.results_storage_path}")

    # train and postprocess
    experiment.train_model()
    if config["calib_frac"] > 0 or config["calib_train_overlap"] > 0:
        experiment.multicalibrate_multiple(mcb_params)

    # evaluate splits
    experiment.evaluate_val()
    experiment.evaluate_test()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger.info(f"Running MC-Industry experiments")

    tune_hyperparams = False
    mcb_configuration = (
        get_mcgrad_configs(tune_hyperparams=tune_hyperparams)
        + HKRR_DEFAULT
        + CALIB_ALGS_DEFAULT
    )
    # mcb_configuration = MCB_TEST
    if tune_hyperparams:
        results_path = f"{RESULTS_DIR}/tuned/"
    else:
        results_path = f"{RESULTS_DIR}/oob/"

    seeds = [15]

    # create all combinations of datasets, models, and seeds
    combs = itertools.product(DATASETS, BASE_MODELS, seeds)
    for dataset, model, seed in combs:
        logger.info(f"********** {dataset} {model} seed={seed} **********")

        data_reuse_experiment(
            model_name=model,
            dataset=dataset,
            seed=seed,
            mcb_params=mcb_configuration,
            results_storage_path=results_path,
        )
