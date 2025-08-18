# pyre-ignore all
import numpy as np
from models.SimpleModel import (
    DecisionTree,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    RandomPredictor,
    SVM,
)


class Model:
    def __init__(
        self, model_name, SAVE_DIR=None, config=None, from_saved=False, **kwargs
    ):
        self.name = model_name
        self.SAVE_DIR = SAVE_DIR
        self.config = config

        # pyre-ignore
        self.calib_frac = config["calib_frac"]
        dataset_obj = kwargs["dataset_obj"] if "dataset_obj" in kwargs else None
        saved_epoch = kwargs["saved_epoch"] if "saved_epoch" in kwargs else 0
        save_scheme = (
            kwargs["save_scheme"] if "save_scheme" in kwargs else "best-val-acc"
        )

        # verify save_scheme
        assert save_scheme in ["best-val-acc", "all-epochs"]
        if self.name not in ["DistilBert", "ViT", "LanguageResNet"]:
            err_msg = f"all-epoch saving not supported for {self.name} model."
            assert save_scheme == "best-val-acc", err_msg

        # simple models
        if self.name == "LogisticRegression":
            self.model = LogisticRegression(config)

        elif self.name == "NaiveBayes":
            self.model = NaiveBayes(config)

        elif self.name == "RandomForest":
            self.model = RandomForest(config)

        elif self.name == "SVM":
            self.model = SVM(config)

        elif self.name == "RandomPredictor":
            self.model = RandomPredictor(config)

        elif self.name == "DecisionTree":
            self.model = DecisionTree(config)

        else:
            raise ValueError(f"Unknown model name: {self.name}")

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.model.train(X_train, y_train, groups_train, X_val, y_val, groups_val)

    def predict_proba(self, X, with_logits=False):
        """
        Returns positive class probabilities.
        If with_logits=True, returns both probabilities of the
        positive class, and logits for both classes.
        """
        # if no training set, return .5
        if self.calib_frac == 1.0:
            p = np.ones((X.shape[0], 2)) * 0.5
            if with_logits:
                return p[:, 1], p
            else:
                return p[:, 1]

        # nontrivial training set
        p = self.model.predict_proba(X, with_logits)
        if with_logits:
            return (p[0][:, 1], p[1])
        else:
            # pyre-ignore
            return p[:, 1]

    def predict(self, X):
        # if no training set, select random class
        if self.calib_frac == 1.0:
            return np.random.choice([0, 1], size=X.shape[0])

        # nontrivial training set
        return self.model.predict(X)
