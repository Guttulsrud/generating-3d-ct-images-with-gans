import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.storage.Storage import Storage

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import os

import matplotlib as mpl

mpl.use('TkAgg')

import numpy as np
from tqdm import tqdm
import nibabel as nib
from utils.inference.generate_image import generate_image
from visualization.display_image import display_image

import pandas as pd
import seaborn as sns
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


show_all = False

experiment_name = 'normalized2mm'

if show_all:
    storage = Storage(experiment_name=None)
    folders = [f for f in os.listdir('../saved_models') if os.path.isdir(os.path.join('../saved_models', f))]
    folder_names = [f for f in folders]

    experiments = []
    for idx, file_name in enumerate(folder_names):
        file_name = file_name.split('saved_models\\')[-1]

        if 'saved_models' in file_name:
            continue

        print(f'[{idx}] {file_name}')
        experiments.append(file_name)

    mode = input('Choose experiment, or press d to download from Cloud')
    mode = int(mode) if mode != 'd' else mode

    if mode == 'd':
        experiments = []
        for idx, experiment in enumerate(storage.list_folders()):
            print(f'[{idx}] {experiment}')
            experiments.append(experiment)

        mode = input('Choose experiment')
        storage.download_results(experiments[int(mode)])
        experiment_name = experiments[int(mode)]
    else:
        experiment_name = experiments[int(mode)]

try:
    file_path = glob.glob(f'../saved_models/{experiment_name}/*events*')[0]
except Exception:
    try:
        file_path = glob.glob(f'../saved_models/{experiment_name}/saved_model/*events*')[0]
    except Exception:
        print('No events file found')
        exit()

# List of scalar names to extract
scalar_names = ['D', 'D_real', 'D_fake', 'G_fake', 'E', 'Sub_E']

# Create an EventAccumulator object to read the events
event_acc = EventAccumulator(file_path)

# Load the events
event_acc.Reload()

print('Generating scalar events')
# Loop over the scalar names and extract the data for each scalar
scalar_data = {}
for scalar_name in scalar_names:
    # Get the scalar events
    scalar_events = event_acc.Scalars(scalar_name)

    # Extract the raw data from the scalar events
    raw_data = [(event.step, event.value) for event in scalar_events]

    # Store the raw data in a dictionary
    scalar_data[scalar_name] = raw_data

scalar_name_mapping = {
    'D': 'Discriminator',
    'D_real': 'Discriminator (real)',
    'D_fake': 'Discriminator (fake)',
    'G_fake': 'Generator',
    'E': 'Encoder',
    'Sub_E': 'Sub-encoder'
}


def plot_results(data, experiment_name, network, smooth_only=False, save_plot=False, print_stats=False):
    if network == 'Generator':
        scalars = ['G_fake']
    elif network == 'Discriminator':
        scalars = ['D', 'D_real', 'D_fake']
    else:
        scalars = ['E', 'Sub_E']

    scalar_data = {key: data[key] for key in scalars if key in data}

    stats_labels = ['Average', 'Min', 'Median', 'Max']
    stats_data = {label: [] for label in stats_labels}
    scalar_labels = []

    for scalar, scalar_values in scalar_data.items():
        scalar = scalar_name_mapping[scalar]
        scalar_labels.append(scalar)

        alpha = 0.001
        data = pd.DataFrame(scalar_values, columns=['Iteration', 'Loss'])
        data['Smoothed Loss'] = data['Loss'].ewm(alpha=alpha).mean()

        fig, ax = plt.subplots()
        ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)

        ax.plot(data['Iteration'], data['Loss'], alpha=0.2, label='Loss')
        ax.plot(data['Iteration'], data['Smoothed Loss'], label='Loss (EMA)')
        # Add lines along the x and y axes

        plt.legend()
        plt.show()
        if print_stats:
            avg = np.mean(data['Loss'])
            minimum = np.min(data['Loss'])
            median = np.median(data['Loss'])
            maximum = np.max(data['Loss'])

            stats_data['Average'].append(avg)
            stats_data['Min'].append(minimum)
            stats_data['Median'].append(median)
            stats_data['Max'].append(maximum)

            print(
                f"{scalar} - Min: {minimum}, Max: {maximum}, Mean: {avg}, Median: {median}, STD: {np.std(data['Loss'])}")

    stats_df = pd.DataFrame(stats_data, index=scalar_labels)

    print(stats_df.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    stats_df.plot(kind='bar', ax=ax)
    plt.title(f'{network} Statistics')
    plt.xlabel('Scalars')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Statistics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.show()

    # Show the plot
    if save_plot:
        directory = f'../results/{experiment_name}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        loss_folder = f'{directory}/loss'

        if not os.path.exists(loss_folder):
            os.mkdir(loss_folder)

        plt.savefig(f'{loss_folder}/{network}.png')


def display_combined_stats(data, experiment_name, show=False, save=False, generate_images=False, n_images=10):
    print('Creating plots...')
    networks = {
        'Generator': ['G_fake'],
        'Discriminator': ['D', 'D_real', 'D_fake'],
        'Encoder': ['E', 'Sub_E']
    }

    stats_labels = ['Average', 'Min', 'Median', 'Max']
    stats_data = {label: [] for label in stats_labels}
    scalar_labels = []

    for network, scalars in networks.items():
        for scalar in scalars:
            if scalar not in data:
                continue

            scalar_values = data[scalar]
            scalar_name = scalar_name_mapping[scalar]
            scalar_labels.append(f"{scalar_name}")
            scalar_df = pd.DataFrame(scalar_values, columns=['Iteration', 'Loss'])

            avg = np.mean(scalar_df['Loss'])
            minimum = np.min(scalar_df['Loss'])
            median = np.median(scalar_df['Loss'])
            maximum = np.max(scalar_df['Loss'])

            stats_data['Average'].append(avg)
            stats_data['Min'].append(minimum)
            stats_data['Median'].append(median)
            stats_data['Max'].append(maximum)

    stats_df = pd.DataFrame(stats_data, index=scalar_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    stats_df.plot(kind='bar', ax=ax)
    plt.title(f'{experiment_name} Stats', fontsize=16)
    plt.xlabel('Scalars')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Statistics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show the plot
    if save:
        directory = f'../results/{experiment_name}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        stats_folder = f'{directory}/stats'

        if not os.path.exists(stats_folder):
            os.mkdir(stats_folder)

        plt.savefig(f'{stats_folder}/bar_plot.png')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.axis('off')
    ax2.axis('tight')
    table = ax2.table(cellText=stats_df.round(2).values, colLabels=stats_df.columns, rowLabels=stats_df.index,
                      loc='left', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1)
    plt.title(f'{experiment_name} Stats', fontsize=12)
    plt.tight_layout()

    if save:
        directory = f'../results/{experiment_name}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        stats_folder = f'{directory}/stats'

        if not os.path.exists(stats_folder):
            os.mkdir(stats_folder)

        plt.savefig(f'{stats_folder}/stats.png')

    if generate_images:
        model_path = f'../saved_models/{experiment_name}/saved_model'

        out_folder = f'../results/{experiment_name}/generated_images'

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if not os.path.exists(f'{out_folder}/generated_images/png'):
            os.makedirs(f'{out_folder}/generated_images/png')

        if not os.path.exists(f'{out_folder}/generated_images/nifti'):
            os.makedirs(f'{out_folder}/generated_images/nifti')

        print('Generating images...')
        for x in tqdm(range(0, n_images)):
            gen_image = generate_image(model_path=model_path, save_step=80000)
            # fig = display_image(gen_image, show=False, return_figure=True)
            # fig.savefig(f'{out_folder}/generated_images/png/image_{x + 1}.png')
            nifti = nib.Nifti1Image(gen_image, np.eye(4))
            nib.save(nifti, f'{out_folder}/generated_images/nifti/image_{x + 1}.nii.gz')


plot_results(scalar_data, experiment_name, 'Generator', save_plot=True, smooth_only=True, print_stats=True)
# plot_results(scalar_data, experiment_name, 'Generator', save_plot=True, smooth_only=True, print_stats=True)
# plot_results(scalar_data,experiment_name,  'Discriminator', save_plot=True, smooth_only=True, print_stats=True)
# plot_results(scalar_data,experiment_name,  'Encoder', save_plot=True, smooth_only=True, print_stats=True)
# display_combined_stats(data=scalar_data, experiment_name=experiment_name, save=True, generate_images=False, n_images=10)
