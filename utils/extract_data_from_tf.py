import os

import matplotlib as mpl
import numpy as np

mpl.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

experiment_name = '3k_dataset_shuffled160k'
file_path = f'../saved_models/{experiment_name}/checkpoint/events.out.tfevents.1678783139.DESKTOP-O2QP71I'

# List of scalar names to extract
scalar_names = ['D', 'D_real', 'D_fake', 'G_fake', 'E', 'Sub_E']

# Create an EventAccumulator object to read the events
event_acc = EventAccumulator(file_path)

# Load the events
event_acc.Reload()

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


def plot_results(data, network, smooth_only=False, save_plot=False):
    if network == 'Generator':
        scalars = ['G_fake']
    elif network == 'Discriminator':
        scalars = ['D', 'D_real', 'D_fake']
    else:
        scalars = ['E', 'Sub_E']

    scalar_data = {key: data[key] for key in scalars if key in data}
    # Create a single figure and axis outside the loop
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for scalar, scalar_values in scalar_data.items():
        scalar = scalar_name_mapping[scalar]
        data = pd.DataFrame(scalar_values, columns=['Iteration', 'Loss'])

        # Smooth the data using a rolling window
        window_size = 15  # Choose an appropriate window size based on the number of data points
        data['SmoothedValue'] = data['Loss'].rolling(window=window_size, center=True).mean()

        if smooth_only:
            sns.lineplot(x='Iteration', y='SmoothedValue', data=data, linewidth=2.5, ax=ax,
                         label=f'{scalar} Rolling average {window_size}')
        else:
            sns.lineplot(x='Iteration', y='Loss', data=data, linewidth=2.5, ax=ax, label=f'{scalar} Loss')

    # Set title and axis labels
    plt.title(f'{network} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    # Show the plot
    if save_plot:
        directory = f'../plots/{experiment_name}'
        if not os.path.exists(directory):
            os.mkdir(directory)
        plt.savefig(f'{directory }/{network}.png')


plot_results(scalar_data, 'Generator', save_plot=True, smooth_only=True)
plot_results(scalar_data, 'Discriminator', save_plot=True, smooth_only=True)
plot_results(scalar_data, 'Encoder', save_plot=True, smooth_only=True)
