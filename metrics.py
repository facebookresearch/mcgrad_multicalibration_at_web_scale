
import pandas as pd
from utils import ConsoleColor
from relplot import smECE as _smECE
from prettytable import PrettyTable as pt
from collections import defaultdict
from sklearn import metrics as skmetrics
import numpy as np
import wandb
from mcb_algorithms.CAS.metrics import MulticalibrationError, kuiper_calibration_per_segment



def binnedECE(f, y):
    '''Expected Calibration Error (binned)'''
    def _binning(f, y, bin_size=0.1, shift=0):
        bi = (f + shift) / bin_size
        bi = bi.astype(int)
        r = f - y
        bins_cnt = int((shift + 1) / bin_size) + 1
        bins = np.zeros(bins_cnt)
        np.add.at(bins, bi, r)
        return bins
    
    def _binnedECE(f, y, nbins=10):
        bins = _binning(f, y, bin_size=1.0 / nbins, shift=0)
        return np.abs(bins).sum() / len(f)
    
    return _binnedECE(f, y)


def smECE(f, y):
    '''Expected Calibration Error (smoothed)'''
    return _smECE(f, y)

def ecce_perc(f, y):
    '''Estimated Cumulative Calibration Error'''
    raw_kuiper = kuiper_calibration_per_segment(labels=y, predicted_scores=f, normalization_method=None)[0]
    return raw_kuiper / np.mean(y) * 100

def ecce_sigma(f, y):
    '''Estimated Cumulative Calibration Error'''
    return kuiper_calibration_per_segment(labels=y, predicted_scores=f, normalization_method='kuiper_standard_deviation')[0]


def subgroup_metrics(
        subgroups,
        targets,
        positive_class_confs,
        preds,
        df=None,
        categorical_features=None,
        numerical_features=None
):
    '''return dictionary of groupwise calibration metrics'''

    # Compute the MCE if features are provided
    df['score'] = positive_class_confs

    mce = MulticalibrationError(
        df=df,
        label_column='label',
        score_column='score',
        numerical_segment_columns=numerical_features,
        categorical_segment_columns=categorical_features,
    )

    subgroup_metrics = {}

    for i, group in enumerate(subgroups):
        subgroup_frac = len(group) / len(positive_class_confs)

        # check for empty subgroups
        if len(group) <= 1:
            subgroup_metrics[i] = {
                "size": round(subgroup_frac,4), 
                "acc": np.nan,
                "log-loss": np.nan,
                "ECE": np.nan,
                "smECE": np.nan,
                "ECCE_perc": np.nan,
                "ECCE_sigma": np.nan,
            }
            continue

        subgroup_confs = positive_class_confs[group]
        subgroup_preds = preds[group]
        subgroup_targets = targets[group]

        # deterministic metrics
        group_acc = len(np.where(subgroup_preds == subgroup_targets)[0]) / len(group)
        group_logloss = skmetrics.log_loss(subgroup_targets, subgroup_preds)
        group_rocauc = skmetrics.roc_auc_score(subgroup_targets, subgroup_preds)
        group_prauc = skmetrics.average_precision_score(subgroup_targets, subgroup_preds)
        ece = binnedECE(subgroup_confs, subgroup_targets)
        smece = smECE(subgroup_confs, subgroup_targets)
        ecce_perc_val = ecce_perc(subgroup_confs, subgroup_targets)
        ecce_sigma_val = ecce_sigma(subgroup_confs, subgroup_targets)

        subgroup_metrics[i] = {
            "size": round(subgroup_frac,4), 
            "acc": round(group_acc,4),
            "logloss": round(group_logloss,4),
            "rocauc": round(group_rocauc,4),
            "prauc": round(group_prauc,4),
            "ECE": round(ece,4),
            "smECE": round(smece, 4),
            "ECCE_perc": round(ecce_perc_val, 4),
            "ECCE_sigma": round(ecce_sigma_val, 4),
            "MCE_perc": np.nan,
            "MCE_sigma": np.nan,
        }

    # get aggregate metrics
    agg_metrics = {
        "size": 1.0,
        "acc": round(len(np.where(preds == targets)[0]) / len(targets), 4),
        "logloss": round(skmetrics.log_loss(targets, preds),4),
        "rocauc": round(skmetrics.roc_auc_score(targets, preds),4),
        "prauc": round(skmetrics.average_precision_score(targets, preds),4),
        "ECE": round(binnedECE(positive_class_confs, targets), 4),
        "smECE": round(smECE(positive_class_confs, targets), 4),
        "ECCE_perc": round(mce.global_ecce, 4),
        "ECCE_sigma": round(mce.global_ecce_sigma_scale, 4),
        "MCE_perc": round(mce.mce, 4),
        "MCE_sigma": round(mce.mce_sigma_scale, 4),
    }

    def _agg_metric(agg_func, metric_name, sg_metrics):
        return round(
            agg_func(
                [sg_metrics[i][metric_name] for i in sg_metrics if sg_metrics[i][metric_name] != np.nan]
            ), 4
        )
    # add mean subgroup metrics
    all_metrics = ["size", "acc", "logloss", "rocauc", "prauc", "ECE", "smECE", "ECCE_perc", "ECCE_sigma"]
    sg_mean = {metric_name: _agg_metric(np.mean, metric_name, subgroup_metrics) for metric_name in all_metrics}
    sg_mean |= {"MCE_perc": np.nan, "MCE_sigma": np.nan}

    # add subgroup max
    sg_max = {metric_name: _agg_metric(np.max, metric_name, subgroup_metrics) for metric_name in all_metrics}
    sg_max |= {"MCE_perc": np.nan, "MCE_sigma": np.nan}

    # add subgroup min
    sg_min = {metric_name: _agg_metric(np.min, metric_name, subgroup_metrics) for metric_name in all_metrics}
    sg_min |= {"MCE_perc": np.nan, "MCE_sigma": np.nan}

    # subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['max'] = sg_max
    subgroup_metrics['min'] = sg_min
    subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['agg'] = agg_metrics

    return subgroup_metrics

class Logger:
    def __init__(self, project, config, run_name=None):
        self.project = project
        self.run = run_name
        self.config = config
        wandb.init(project=project, name=run_name, config=config)

    def log(self, model, split, group_metrics):
        """
        Log group metrics to wandb.
        Parameters:
            model (str): The stage of the experiment, e.g. 'ERM', 'MCB'
            group_metrics (dict): The groupwise calibration metrics {group_index: {metric: value}}
        """
        log_dict = defaultdict(dict)
        for gp in group_metrics:
            for metric, value in group_metrics[gp].items():
                log_dict[f"{model}/{split}/{metric}/{gp}"] = value

        wandb.log(log_dict)

    def log_plot(self, name, plot):
        wandb.log({name: plot})

    def finish(self):
        wandb.finish()

