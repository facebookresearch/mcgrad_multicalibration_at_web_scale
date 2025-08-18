# default splits and seeds
SPLIT_DEFAULT = {"train": 0.6, "val": 0.2, "test": 0.2}

# default hyperparameters for HKRR
HKRR_DEFAULT = [
    {
        "type": "HKRR",
        "params": [
            {
                "lambda": 0.1,
                "alpha": 0.1,
            },
            {
                "lambda": 0.1,
                "alpha": 0.05,
            },
            {
                "lambda": 0.1,
                "alpha": 0.025,
            },
            {
                "lambda": 0.1,
                "alpha": 0.0125,
            },
        ],
    },
]
# standard calibration algorithms
CALIB_ALGS_DEFAULT = [{"type": "Isotonic", "params": [{}]}]

MCGRAD_NAME = "MCGrad"
FEATURE_TYPE_GROUPS = "group"
FEATURE_TYPE_FEATURES = "features"


def get_mcgrad_configs(tune_hyperparams: bool = False):
    return [
        {
            "type": MCGRAD_NAME,
            "params": [
                # Min_sum_hessian ablation
                # IF we tune hyperparams we set it explicitly to 0 if not the default is zero so we set it to 20. Hacky
                {
                    "name": MCGRAD_NAME
                    + "_msh_"
                    + (str(0) if tune_hyperparams else str(20)),
                    "feature_type": FEATURE_TYPE_FEATURES,
                    "unshrink": True,
                    "encode_categorical_variables": True,
                    "num_rounds": 10,
                    "lightgbm_params": {
                        "min_sum_hessian_in_leaf": 0 if tune_hyperparams else 20
                    },
                    "early_stopping": True,
                    "fixed_parameters": ["min_sum_hessian_in_leaf"],
                    "tune_hyperparameters": tune_hyperparams,
                },
                # Vanilla (our) variant
                {
                    "name": MCGRAD_NAME,
                    "feature_type": FEATURE_TYPE_FEATURES,
                    "unshrink": True,
                    "encode_categorical_variables": True,
                    "num_rounds": 10,
                    "lightgbm_params": None,
                    "early_stopping": True,
                    "tune_hyperparameters": tune_hyperparams,
                },
                # Our variant with group features
                {
                    "name": MCGRAD_NAME + "_group_features",
                    "feature_type": FEATURE_TYPE_GROUPS,
                    "unshrink": True,
                    "encode_categorical_variables": True,
                    "num_rounds": 30,
                    "lightgbm_params": None,
                    "early_stopping": True,
                    "tune_hyperparameters": tune_hyperparams,
                },
                # Unshrink ablation
                {
                    "name": MCGRAD_NAME + "_no_unshrink",
                    "feature_type": FEATURE_TYPE_FEATURES,
                    "unshrink": False,
                    "encode_categorical_variables": True,
                    "num_rounds": 10,
                    "lightgbm_params": None,
                    "early_stopping": True,
                    "tune_hyperparameters": tune_hyperparams,
                },
                # One round ablation
                {
                    "name": MCGRAD_NAME + "_one_round",
                    "feature_type": FEATURE_TYPE_FEATURES,
                    "unshrink": True,
                    "encode_categorical_variables": True,
                    "num_rounds": 1,
                    "lightgbm_params": None,
                    "early_stopping": False,
                    "tune_hyperparameters": tune_hyperparams,
                },
                # Jin et al. Variant
                {
                    "name": "DFMC",
                    "feature_type": FEATURE_TYPE_GROUPS,
                    "unshrink": False,
                    "encode_categorical_variables": True,
                    "num_rounds": 1,
                    "lightgbm_params": {
                        "learning_rate": 0.1,
                        "min_child_samples": 20,
                        "num_leaves": 31,
                        "n_estimators": 100,
                        "lambda_l2": 0.0,
                        "min_gain_to_split": 0.0,
                        "max_depth": 2,
                        "min_sum_hessian_in_leaf": 1e-3,
                    },
                    "early_stopping": False,
                    "tune_hyperparameters": tune_hyperparams,
                },
            ],
        }
    ]


US_STATES = [
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Federal_district.
    # "DC",
    # # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Inhabited_territories.
    # "AS", "GU", "MP", "PR", "VI",
]
