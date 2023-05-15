import pandas as pd
import yaml
from matplotlib import pyplot as plt
import matplotlib as mpl
import shutil

mpl.use('TkAgg')

import os
import glob

main_dir = "../saved_models_segment/5fold"

all_mean_dfs = []  # Initialize list to store mean dataframes for each set of models

# Create separate lists for each experiment
experiments = {
    '128_R': [],
    '128_F': [],
    'RF_testing_5fold': [],
}

for dirpath, dirnames, filenames in os.walk(main_dir):

    if 'segresnet_4' not in dirnames:
        # Skip results that are not 5fold cross validated or the first 5 models
        continue

    experiment_name = dirpath.split(os.sep)[-1]
    if experiment_name not in experiments:
        continue

    all_dataframes = []
    for dirname in dirnames:
        if "segresnet_" in dirname:
            accuracy_file = os.path.join(dirpath, dirname, "model", "accuracy_history.csv")
            data = pd.read_csv(accuracy_file, delimiter='\t')
            data = data.sort_values('epoch')

            data[f'Smoothed metric'] = data['metric'].ewm(alpha=0.01, adjust=False).mean()
            all_dataframes.append(data)

    mean_df = pd.concat(all_dataframes).groupby('epoch')['Smoothed metric'].mean().reset_index()
    experiments[experiment_name].append(mean_df)  # Append mean_df to the corresponding experiment list

# Concatenate the mean dataframes for each experiment separately
combined_dfs = []
for experiment_name in experiments:
    combined_df = pd.concat(experiments[experiment_name])
    combined_df = combined_df.groupby('epoch')['Smoothed metric'].mean().reset_index()
    combined_dfs.append(combined_df)

# Create a new plot
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('DCS')

# Plot the smoothed DCS values for each experiment
for i, combined_df in enumerate(combined_dfs):
    experiment_name = list(experiments.keys())[i]
    experiment_name = {
        '128_F': 'Generated',
        '128_R': 'Real',
        'RF_testing_5fold': 'Real and Generated',
    }.get(experiment_name)
    ax.plot(combined_df['epoch'], combined_df['Smoothed metric'], label=f'{experiment_name} (EMA)')

ax.legend()
plt.savefig(f'../results/segmentation/5_fold/128_all_alt.png')
plt.show()

