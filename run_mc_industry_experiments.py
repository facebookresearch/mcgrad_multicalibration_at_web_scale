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
        'ACSIncome',
        'CreditDefault',
        'BankMarketing',
        'HMDA',
        'MEPS',
    ]

    # seeds = SEEDS_DEFAULT
    seeds = [15]

    # create all combinations of datasets, models, and seeds
    combs = itertools.product(datasets, models, seeds)
    for dataset, model, seed in combs:
        print(f'********** {dataset} {model} seed={seed} **********')
        data_reuse_experiment(model, dataset, seed, wandb=False)


if __name__ == "__main__":
    # load_ACSIncome()
    # load_CreditDefault()
    # load_BankMarketing()
    # load_HMDA()
    # load_MEPS()

    run_mc_industry_experiments()
