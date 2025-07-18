from dataloaders.ACS import load_ACSIncome
from dataloaders.CreditDefault import load_CreditDefault
from dataloaders.BankMarketing import load_BankMarketing
from dataloaders.HMDA import load_HMDA
from dataloaders.MEPS import load_MEPS
from run_experiment import data_reuse_experiment
from configs.constants import SEEDS_DEFAULT
import itertools

def run_mc_industry_experiments():

    models = [
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

    seeds = SEEDS_DEFAULT

    # create all combinations of datasets, models, and seeds
    combs = itertools.product(datasets, models, seeds)
    for dataset, model, seed in combs:
        print(f'********** {dataset} {model} seed={seed} **********')
        try:
            data_reuse_experiment(model, dataset, seed, wandb=False)
        except Exception as e:
            print(f'Experiment faild {model}, {dataset}, {seed}')
            print(e)
            continue


if __name__ == "__main__":
    run_mc_industry_experiments()
