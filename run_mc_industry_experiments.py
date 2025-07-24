from run_experiment import data_reuse_experiment
from configs.constants import SEEDS_DEFAULT
import itertools
from mcb_algorithms.CAS.methods import logger as mcb_logger
from mcb_algorithms.CAS.CASMCBoost import logger as mcb_wrapper_logger

import logging
import sys

from configs.constants import HKRR_DEFAULT, CALIB_ALGS_DEFAULT, get_mcgrad_configs

logger = logging.getLogger(__name__)

def run_mc_industry_experiments():

    mcb_configuration = get_mcgrad_configs(tune_hyperparams=False)

    base_models = [
        # 'SVM',
        'LogisticRegression',
        # 'NaiveBayes',
        # 'RandomForest'
    ]

    datasets = [
        'acs_employment_all_states',
        'acs_health_insurance_all_states',
        'acs_public_health_insurance_all_states',
        'acs_travel_time_all_states',
        'acs_mobility_all_states',
        'acs_income_all_states',
        'ACSIncome',
        'CreditDefault',
        'BankMarketing',
        'HMDA',
        'MEPS',
    ]

    # seeds = SEEDS_DEFAULT
    seeds = [15]

    # create all combinations of datasets, models, and seeds
    combs = itertools.product(datasets, base_models, seeds)
    for dataset, model, seed in combs:
        logger.info(f'********** {dataset} {model} seed={seed} **********')
        try:
            data_reuse_experiment(model, dataset, seed, wandb=False, mcb_params=mcb_configuration)
        except Exception as e:
            logger.error(f'Experiment faild {model}, {dataset}, {seed}. {e}')
            continue


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logger.info(f'Running MC-Industry experiments')

    run_mc_industry_experiments()
