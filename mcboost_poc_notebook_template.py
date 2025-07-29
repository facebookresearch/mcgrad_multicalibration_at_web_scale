#!/usr/bin/env -S grimaldi --kernel bento_kernel_multicalibration
# fmt: off
# flake8: noqa
# FILE_UID: 7d0ddcb0-c5c5-4d5d-9908-3d35db3db20b
# NOTEBOOK_NUMBER: N6132191 (891766209726341)

""":md
# MCBoost Opportunity Sizing POC

This is a template for a POC of MCBoost for applications. Please clone the notebook, fill in the required inputs below, and execute all cells.

MCBoost is a lightweight algorithm to achieve approximate multicalibration - calibration not only globally but also in sub-segments of the data. You can find more information about MCBoost and the metrics used in this notebook in our [wiki](https://www.internalfb.com/wiki/MCBoost/). If you have any questions or run into issues with this notebook please reach out to us and the community in the [MCBoost Users](https://fb.workplace.com/groups/884914536988621) Workplace group.
"""

""":py"""
import functools
import logging
import os

import numpy as np
import pandas as pd
import pvc2
import matplotlib.pyplot as plt
import seaborn as sns
from multicalibration import methods, metrics, plotting, tuning, utils
from multicalibration.metrics import (
    wrap_multicalibration_error_metric,
    wrap_sklearn_metric_func,
)
from multicalibration.poc_notebook_template import notebook_helpers as helpers
from plotly import io as pio
from sklearn import metrics as skmetrics

""":py"""
# flake8: noqa W605 , the latex "sigma" in markdown cells below triggers these warnings.
pio.templates["custom"] = helpers.get_plotting_template() # pyre-ignore[16]
pio.templates.default = "custom"
logger: logging.Logger = logging.getLogger(__name__)
helpers.logger.setLevel(logging.INFO)
methods.logger.setLevel(logging.INFO)
rng = np.random.RandomState(42)

""":py"""
df = pd.read_csv('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/kdd_mcboost_results_oob_params.csv')
#names = ['dataset','model','mcb_algorithm','MCE_per_features', 'MCE_sigma_features','global_ECCE_perc','global_ECCE_sigma','global_ECE',
#        'logloss','prauc','fit_time','num_rounds_mcboost','MCE_perc_groups','MCE_sigma_groups','max_group_smECE'])

dslist = []
for ds in df[~df['dataset'].isna()].dataset.unique():
    dslist.extend([ds[:10]]*9)
df['dataset'] = dslist
#df.drop('model', axis = 1, inplace = True)
df['mcb_algorithm'] = df['mcb_algorithm'].replace(
    {
        'base_model': 'BasePred',
        'DFMCBoost': 'DFMC',
        'MCGrad_msh_20': 'MCGrad',
        'MCGrad_no_unshrink': 'No Rescaling',
        'MCGrad_one_round': 'Only One Round (T=1)',
        'MCGrad': 'No Min Hessian Leaf',
        'MCGrad_group_features': 'Prespecified Groups',
    }
)
symbols = ['p', 'o', 'D', 'v', 'X']
colors = ['chocolate', 'gray', 'green', 'pink', 'blue']

def compute_improvement(group):
    dataset_name = group.name
    # Find the baseline row for this dataset
    baseline_row = baseline_df[baseline_df['dataset'] == dataset_name]
    if not baseline_row.empty:
        baseline_row = baseline_row.iloc[0]  # get the first baseline row as Series
        cols = ['MCE_perc_features', 'MCE_sigma_features', 'MCE_sigma_groups', 'global_ECCE_perc', 'global_ECCE_sigma', 'global_ECE', 'logloss']
        for col in cols:
            # Compute relative improvement: (baseline - current) / baseline
            group[f'{col}_improvement'] = 100*(baseline_row[col] - group[col]) / baseline_row[col]
        group['prauc_improvement'] = -100*(baseline_row['prauc'] - group['prauc']) / baseline_row['prauc']
    else:
        # If no baseline found, fill improvements with NaN
        for col in cols:
            group[f'{col}_improvement'] = float('nan')
    return group

""":py '757865203381219'"""
df['mcb_algorithm'].unique()

""":md
# Q1: MCBoost vs Baselines
"""

""":py '771832648858915'"""
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
for i,method in enumerate(['BasePred $f_0$', 'HKRR', 'Isotonic', 'DFMC', 'MCGrad']):
    df_q1 = df[df['mcb_algorithm']==method].sort_values('dataset')
    plt.scatter(x = df_q1['dataset'].values, y = df_q1['MCE_sigma_features'], s = 150, marker = symbols[i], label = method,
                edgecolor = 'black', c = colors[i], lw = 1)

y_value = 4.70812972
x_start = -0.5
x_end = len(df.dataset.unique()) - 0.5
plt.hlines(y=y_value, xmin=x_start, xmax=x_end, lw=2, ls='--', color='darkred')
# Annotation position (right end of the line)
annot_x = x_end
annot_y = y_value
# Arrow start (on the line), arrow end (just before annotation)
arrow_start = (annot_x, y_value)
arrow_end = (annot_x, y_value+40)
# Draw the arrow
plt.annotate(
    '',  # No text for the arrow itself
    xy=arrow_end, xytext=arrow_start,
    arrowprops=dict(arrowstyle='->', color='darkred', lw=3)
)
# Add the annotation text
plt.annotate(
    "Stat. Sig. Miscalibration",
    xy=arrow_end,
    xytext=(8, -30),  # Offset text 10 points to the right
    textcoords='offset points',
    ha='left',
    va='center',
    fontsize=14,
    color='darkred',
    rotation=90
)

yticks = plt.yticks()[0].tolist()
# Append the new y_value if not already present
if y_value not in yticks:
    yticks.append(y_value)
    yticks = sorted(yticks)
plt.yticks(yticks)
plt.ylim(-1,90)
# After setting yticks, color the new tick label dark red
ax = plt.gca()  # Get current axes
for label in ax.get_yticklabels():
    if label.get_position()[1] == y_value:
        label.set_color('darkred')
        label.set_fontweight('bold')

plt.xticks(rotation = 90)
# (\u03C3 scale)
plt.ylabel('Multi-calibration Error', weight = 'bold', fontsize = 18)
plt.xlabel("Benchmark Datasets", weight = 'bold', fontsize = 18)
legend = plt.legend(fontsize = 14,handletextpad = 0, frameon = True, loc = 'upper left', bbox_to_anchor = (0, 1))
plt.xlim(-.5, len(df.dataset.unique())+.2)
#plt.ylim(1,37)
plt.grid(alpha = 0.4)
sns.despine()
#make white background
plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
#plt.savefig('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/MCE_sigmaScalePlot.pdf', dpi=300, bbox_inches='tight')
plt.show()

""":py '3982740008641870'"""
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
for i,method in enumerate(['BasePred', 'HKRR', 'Isotonic', 'DFMC', 'MCGrad']):
    df_q1 = df[df['mcb_algorithm']==method].sort_values('dataset')
    plt.scatter(x = df_q1['dataset'].values, y = df_q1['MCE_sigma_features'], s = 150, marker = symbols[i], label = method,
                edgecolor = 'black', c = colors[i], lw = 1)

plt.xticks(rotation = 90)
plt.ylabel('Multicalibration Error', weight = 'bold', fontsize = 18)
plt.xlabel("Benchmark Datasets", weight = 'bold', fontsize = 18)
legend = plt.legend(fontsize = 14,handletextpad = 0, frameon = True, loc = 'upper left', bbox_to_anchor = (0, 1))
plt.xlim(-.5, len(df.dataset.unique())+.2)
#plt.ylim(1,37)
plt.grid(alpha = 0.4)
sns.despine()
#make white background
plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
#plt.savefig('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/MCE_sigmaScalePlot_nostatsig.pdf', dpi=300, bbox_inches='tight')
plt.show()

""":py"""


""":py '1287446019512314'"""
dfmcboost = df[df['mcb_algorithm'].isin(['BasePred', 'HKRR', 'Isotonic', 'DFMC', 'MCGrad'])].sort_values(by = ['dataset', 'mcb_algorithm'])
baseline_df = dfmcboost[dfmcboost['mcb_algorithm'] == 'BasePred'].reset_index(drop=True)

df_with_improvements = dfmcboost.groupby('dataset').apply(compute_improvement).reset_index(drop=True)
#df_with_improvements = df_with_improvements[df_with_improvements['mcb_algorithm']!='BasePred $f_0$'].reset_index(drop = True)
df_with_improvements.head()

""":py '1084164876549312'"""
print("Average reduction of MCE sigma scale:")

for algo in df_with_improvements.mcb_algorithm.unique():
    print(f"{algo}: {df_with_improvements[df_with_improvements['mcb_algorithm']==algo]['MCE_sigma_features_improvement'].mean():.2f}%")


""":py '603877575834494'"""
df_with_improvements[(df_with_improvements['mcb_algorithm']=='MCGrad')][['dataset','MCE_sigma_groups_improvement']]

""":py '1805809310315280'"""
df_with_improvements['harm_f0'] = df_with_improvements['MCE_sigma_features_improvement']<0
df_with_improvements.groupby('mcb_algorithm')['harm_f0'].sum()

""":py '1771565800429595'"""
df_with_improvements['MCE_sigma_features_rank'] = df_with_improvements.groupby('dataset')['MCE_sigma_features'].rank(ascending = True, method = 'average').reset_index(drop = True)
df_with_improvements['prauc_rank'] = df_with_improvements.groupby('dataset')['prauc'].rank(ascending = False, method = 'average').reset_index(drop = True)
df_with_improvements['logloss_rank'] = df_with_improvements.groupby('dataset')['logloss'].rank(ascending = True, method = 'average').reset_index(drop = True)
df_with_improvements['global_ECCE_perc_rank'] = df_with_improvements.groupby('dataset')['global_ECCE_perc'].rank(ascending = True, method = 'average').reset_index(drop = True)

df_with_improvements.groupby('mcb_algorithm')[['MCE_sigma_features','MCE_sigma_features_rank']].mean().reset_index().sort_values('MCE_sigma_features_rank')

""":py '760842669634628'"""
df_with_improvements.groupby('mcb_algorithm')[['prauc','prauc_rank']].mean().reset_index().sort_values('prauc_rank')

""":py '1469156240756239'"""
df_with_improvements.groupby('mcb_algorithm')[['logloss','logloss_rank']].mean().reset_index().sort_values('logloss_rank')

""":py"""


""":py '683950594661653'"""
df_with_improvements[df_with_improvements['mcb_algorithm']=='MCGrad'][['dataset','MCE_sigma_features_rank']]

""":py '1611021622909937'"""
df_with_improvements[df_with_improvements['dataset']=='MEPS'][['mcb_algorithm','MCE_sigma_features_improvement']]

""":py '1418743322712924'"""


""":py '1108897174460981'"""
dfvalranks = df_with_improvements[['mcb_algorithm','MCE_sigma_features', 'logloss', 'prauc', 'global_ECCE_perc', 'MCE_sigma_features_rank', 'logloss_rank', 'prauc_rank', 'global_ECCE_perc_rank']].sort_values(by = 'logloss').groupby('mcb_algorithm').mean().round(3).reset_index()
for col in ['MCE_sigma_features','logloss', 'prauc', 'global_ECCE_perc']:
    dfvalranks[col] = dfvalranks.apply(lambda x: f"{x[col]:.2f} ± {x[col+'_rank']:.2f}", axis=1)

dfvalranks = dfvalranks[['mcb_algorithm', 'MCE_sigma_features', 'logloss', 'prauc', 'global_ECCE_perc']]
print(dfvalranks.to_latex())

""":md
# Q2: Prespecified groups
"""

""":py '1681745072343623'"""
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
for i,method in enumerate(['BasePred', 'HKRR', 'Isotonic', 'DFMC', 'MCGrad']):
    df_q1 = df[df['mcb_algorithm']==method].sort_values('dataset')
    plt.scatter(x = df_q1['dataset'].values, y = df_q1['MCE_sigma_groups'], s = 150, marker = symbols[i], label = method,
                edgecolor = 'black', c = colors[i], lw = 1)

plt.xticks(rotation = 90)
plt.ylabel('Multicalibration Error', weight = 'bold', fontsize = 18)
plt.xlabel("Benchmark Datasets", weight = 'bold', fontsize = 18)
legend = plt.legend(fontsize = 14,handletextpad = 0, frameon = True, loc = 'upper left', bbox_to_anchor = (0, 1))
plt.xlim(-.5, len(df.dataset.unique())+.2)
#plt.ylim(1,37)
plt.grid(alpha = 0.4)
sns.despine()
#make white background
plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
#plt.savefig('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/MCE_sigmaScalePlot_nostatsig_groups.pdf', dpi=300, bbox_inches='tight')
plt.show()

""":py '1452443279227912'"""
df_with_improvements['MCE_ranks'] = df_with_improvements.groupby('dataset')['MCE_sigma_groups'].rank(ascending = True, method = 'average').reset_index(drop = True)
df_with_improvements.groupby('mcb_algorithm')['MCE_ranks'].mean().reset_index(name = 'MCE_rank').sort_values('MCE_rank')

""":py '1116921543820844'"""
df_with_improvements[df_with_improvements['mcb_algorithm']=='MCGrad'][['dataset','MCE_ranks']]

""":py"""
df_with_improvements.groupby('mcb_algorithm')['MCE_sigma_groups_improvement'].mean().reset_index()

""":py"""
df_with_improvements.groupby('mcb_algorithm').mean().reset_index()

""":py"""
df_with_improvements[['mcb_algorithm','logloss_improvement', 'prauc_improvement', 'global_ECCE_sigma_improvement']].groupby('mcb_algorithm').mean().reset_index()

""":py"""
dfperf = df_with_improvements.sort_values(by = 'logloss_improvement', ascending = False).reset_index(drop = True)
dfmean = dfperf[['mcb_algorithm','logloss', 'prauc', 'global_ECCE_sigma']].groupby('mcb_algorithm').mean().reset_index()
dfste = dfperf[['mcb_algorithm','logloss', 'prauc', 'global_ECCE_sigma']].groupby('mcb_algorithm').sem().reset_index()

dfmeanpmste = dfmean.merge(dfste, on = 'mcb_algorithm', suffixes = ('', '_ste')).round(3)
for col in ['logloss', 'prauc', 'global_ECCE_sigma']:
    dfmeanpmste[col] = dfmeanpmste.apply(lambda x: f"{x[col]:.2f} ± {x[col+'_ste']:.2f}", axis=1)
dfmeanpmste = dfmeanpmste[['mcb_algorithm', 'logloss', 'prauc', 'global_ECCE_sigma']]
print(dfmeanpmste.to_latex())

""":py"""
df_with_improvements[['logloss_rank', 'prauc_rank', 'global_ECCE_sigma_rank']] = df_with_improvements[['dataset','logloss_improvement', 'prauc_improvement', 'global_ECCE_sigma_improvement']].groupby('dataset').rank(ascending = False, method = 'average').reset_index(drop = True)
df_with_improvements.groupby('mcb_algorithm')[['logloss_rank', 'prauc_rank', 'global_ECCE_sigma_rank']].mean()

""":py"""
df_with_improvements[df_with_improvements['mcb_algorithm']=='GBMCT'][['dataset','mcb_algorithm','logloss_rank', 'prauc_rank', 'global_ECCE_sigma_rank']]

""":py"""


""":py '680234005066217'"""
methods = ['BasePred', 'HKRR', 'Isotonic', 'DFMC', 'MCGrad']
fig, ax = plt.subplots(ncols=2, figsize=(14, 5.5), facecolor='white', sharey=False)

for i, method in enumerate(methods):
    df_q1 = df[df['mcb_algorithm']==method].sort_values('dataset')
    ax[0].scatter(x = df_q1['dataset'].values, y = df_q1['MCE_sigma_features'], s = 150, marker = symbols[i], label = method,
                edgecolor = 'black', c = colors[i], lw = 1)

ax[0].set_ylabel('Multicalibration Error', weight='bold', fontsize=18)
ax[0].set_xticks(range(len(df['dataset'].unique())))
ax[0].set_xticklabels(df['dataset'].unique(), rotation=90)
ax[0].set_xlim(-0.5, len(df['dataset'].unique()) -0.5)
ax[0].legend(fontsize=14, handletextpad=0, frameon=True, loc='upper left', bbox_to_anchor=(0, 1))
ax[0].grid(True)
sns.despine(ax=ax[0])
ax[0].text(
    10.2, 88,               # x, y in axes fraction coordinates (0 to 1)
    "All Features",       # text content
    ha='right',             # horizontal alignment
    va='top',             # vertical alignment
    fontsize=16,
    fontweight='bold',
    color = 'black',
    bbox=dict(
        facecolor='lightgrey',
        edgecolor='black',
        boxstyle='square,pad=0.3',
        alpha=.7
    ),)
for i, method in enumerate(methods):
    df_q1 = df[df['mcb_algorithm']==method].sort_values('dataset')
    ax[1].scatter(x = df_q1['dataset'].values, y = df_q1['MCE_sigma_groups'], s = 150, marker = symbols[i], label = method,
                edgecolor = 'black', c = colors[i], lw = 1)

ax[1].set_xlabel("Benchmark Datasets                               ", weight='bold', fontsize=18, ha = 'right', labelpad = 12)
ax[1].set_xticks(range(len(df['dataset'].unique())))
ax[1].set_xticklabels(df['dataset'].unique(), rotation=90)
ax[1].set_xlim(-0.5, len(df['dataset'].unique()) -0.5)
ax[1].grid(True)
ax[1].text(
    -0.2, 48.3,               # x, y in axes fraction coordinates (0 to 1)
    "Prespecified Groups",       # text content
    ha='left',             # horizontal alignment
    va='top',             # vertical alignment
    fontsize=16,
    fontweight='bold',
    color = 'black',
    bbox=dict(
        facecolor='lightgrey',
        edgecolor='black',
        boxstyle='square,pad=0.3',
        alpha=.7
    ),)

sns.despine()
#make white background
plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
#plt.savefig('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/MCE_sigmaScaletwoplots.pdf', dpi=300, bbox_inches='tight')
plt.show()

""":md
# Q3: Ablation Study
"""

""":py"""
dfmcboost = df[df['mcb_algorithm'].isin(['MCGrad', 'No Rescaling','Only One Round (T=1)', 'No Min Hessian Leaf'])].sort_values(by = ['dataset', 'mcb_algorithm'])
baseline_df = dfmcboost[dfmcboost['mcb_algorithm'] == 'MCGrad'].reset_index(drop=True)

df_with_improvements = dfmcboost.groupby('dataset').apply(compute_improvement).reset_index(drop=True)
df_with_improvements = df_with_improvements[df_with_improvements['mcb_algorithm']!='MCGrad'].reset_index(drop = True)

""":py '4061506623996176'"""
import matplotlib.pyplot as plt
import seaborn as sns

# List of improvement columns to plot
improvement_cols = [
    #'MCE_per_features_improvement',
    'MCE_sigma_features_improvement',
    #'global_ECCE_perc_improvement',
    #'',
    'logloss_improvement',
    'prauc_improvement',
    'global_ECCE_perc_improvement',
]

# Melt the DataFrame for easier plotting
df_melted = df_with_improvements.melt(
    id_vars=['dataset', 'mcb_algorithm'],
    value_vars=improvement_cols,
    var_name='Metric',
    value_name='Improvement'
)

# Set up the grid of subplots
n_metrics = len(improvement_cols)
n_cols = 4  # Number of columns in the grid
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5), sharex=True, sharey=False)
axes = axes.flatten()

# Determine global y-axis limits for consistent scale
y_min = df_melted['Improvement'].min()
y_max = df_melted['Improvement'].max()

for i, metric in enumerate(improvement_cols):
    ax = axes[i]
    sns.barplot(
        data=df_melted[df_melted['Metric'] == metric],
        x='dataset',
        y='Improvement',
        hue='mcb_algorithm',
        ax=ax
    )
    title = metric.split('_improvement')[0].upper()
    if title == 'MCE_SIGMA_FEATURES':
        title = 'MULTICALIBRATION ERROR'
    elif title == 'GLOBAL_ECCE_PERC':
        title = 'ECCE'
    ax.set_title(title, fontsize=12, weight = 'bold')
    ax.title.set_bbox(dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.3', alpha=.7))
    if i in [0]:
        ax.set_ylabel('Relative Deterioration (%)', fontsize = 18, weight = 'bold')
    else:
        ax.set_ylabel('')
    if i in [2]:
        ax.set_xlabel('Benchmark Datasets            ', fontsize = 18, weight = 'bold', labelpad = 12, ha = 'right')
    else:
        ax.set_xlabel('')
    #ax.set_ylim(y_min, y_max)  # Set same y-axis scale for all subplots
    ax.tick_params(axis='x', rotation=90)
    ax.legend_.remove()  # Remove legend from other subplots
    ax.grid(alpha=0.4)
# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.legend(
    handles=axes[0].get_legend_handles_labels()[0],  # Get handles from first subplot
    #labels=['No Min Hessian', 'No Rescaling', 'Only Single Round (T=1)'],  # Get labels from first subplot
    loc='upper left',
    bbox_to_anchor=(0.31, 1.07),
    ncol=3,
    frameon=True,
    fontsize = 13
)
plt.subplots_adjust(hspace=0, wspace = 0)
sns.despine()
#make white background
plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
#plt.savefig('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/ablation.pdf', dpi=300, bbox_inches='tight')
plt.show()

""":py '749392697595950'"""
df[df['mcb_algorithm'].isin(['MCGrad','No Rescaling'])][['dataset', 'mcb_algorithm','num_rounds_mcboost']]

""":py '760796766470901'"""
df[df['dataset']=='acs_employ']

""":md

"""

""":py '1879336289292186'"""
df[['dataset','prauc']].groupby('dataset')['prauc'].mean()

""":py '2076641096157834'"""
df_with_improvements[df_with_improvements['mcb_algorithm']=='No Min Hessian Leaf'][['MCE_sigma_features_improvement','global_ECCE_perc_improvement', 'logloss_improvement', 'prauc_improvement']].mean()

""":py"""


""":md
# Q4 Hyperparameter Tuning
"""

""":py"""
df_tuned = pd.read_csv('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/kdd_mcboost_results_tuned_all_datasets.csv')

dslist = []
for ds in df_tuned[~df_tuned['dataset'].isna()].dataset.unique():
    dslist.extend([ds[:10]]*9)
df_tuned['dataset'] = dslist
#df.drop('model', axis = 1, inplace = True)
df_tuned['mcb_algorithm'] = df_tuned['mcb_algorithm'].replace(
    {
        'base_model': 'BasePred',
        'DFMCBoost': 'DFMC',
        'MCGrad_no_unshrink': 'No Rescaling',
        'MCGrad_one_round': 'Only One Round (T=1)',
        'MCGrad_msh_0': 'No Min Hessian Leaf',
        'MCGrad_group_features': 'Prespecified Groups',
    }
)
symbols = ['p', 'o', 'D', 'v', 'X']
colors = ['chocolate', 'gray', 'green', 'pink', 'blue']

df_tuned = df_tuned[df_tuned['mcb_algorithm'] == 'MCGrad']
df_tuned['mcb_algorithm'] = 'MCGrad_tuned'
df = df[df['mcb_algorithm']=='MCGrad']
#df = pd.concat([df, df_tuned], ignore_index=True)
sublist_ds = df_tuned.dataset.unique()
df = df[df['dataset'].isin(sublist_ds)].reset_index(drop = True)

df_tuned.sort_values('dataset', inplace = True)
df.sort_values('dataset', inplace = True)

df_tuned = df_tuned.reset_index(drop = True)
df = df.reset_index(drop = True)

""":py '1258558605941723'"""
pd.concat([df, df_tuned], ignore_index = True).sort_values(['dataset', 'mcb_algorithm'])

""":py '2234008673726110'"""
df

""":py"""
# compute the relative improvement of MCGrad_tuned wrt MCGrad for each dataset
higherisbetter = ['prauc']
lowerisbetter = ['MCE_sigma_features', 'logloss', 'global_ECCE_perc']

for col in lowerisbetter:
    df_tuned[col+'_improvement'] = 100*(df[col] - df_tuned[col])/df[col]
for col in higherisbetter:
    df_tuned[col+'_improvement'] = 100*(df_tuned[col] - df[col])/df[col]

""":py '603802626109813'"""
df_tuned[['dataset', 'MCE_sigma_features_improvement', 'logloss_improvement', 'prauc_improvement', 'global_ECCE_perc_improvement']]

""":py '2556916744653047'"""
df_tuned.columns

""":py"""
df_tuned[col]

""":py '24532420616365852'"""
df2 = pd.read_csv('/data/sandcastle/boxes/fbsource/fbcode/multicalibration/poc_notebook_template/kdd_mcboost_results_tuned.csv')
df2[df2['mcb_algorithm'] == 'MCGrad']

""":py"""

