import glob
import os.path

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

experiment = '2023-02-25 15-37-49 w 512'

if not os.path.exists(f'../results/gan/legacy/{experiment}'):
    os.makedirs(f'../results/gan/legacy/{experiment}')

event_file = glob.glob(f'../saved_models/legacy/{experiment}/tensorboard/*events*')[0]


# Name of the tensor to extract
tensor_name = "Discriminator Loss"
# tensor_name = "Generator Loss"
# Number of batches per epoch
batches_per_epoch = 32

# Load the Tensorboard event file
data = tf.compat.v1.train.summary_iterator(event_file)

# Extract the tensor values and calculate the average value per epoch
tensor_values = []
epoch_values = []
batch_counter = 0
batch_values = []
for summary in data:
    for value in summary.summary.value:
        print(value.tag)
        if value.tag == tensor_name:
            batch_values.append(tf.make_ndarray(value.tensor))
            batch_counter += 1
            if batch_counter == batches_per_epoch:
                epoch_values.append(np.mean(batch_values))
                tensor_values.extend(batch_values)
                batch_values = []
                batch_counter = 0
                if len(epoch_values) == 200:
                    break
    if len(epoch_values) == 200:
        break
exit()
stats_labels = ['Average', 'Min', 'Median', 'Max']
stats_data = {label: [] for label in stats_labels}

avg = np.mean(epoch_values)
minimum = np.min(epoch_values)
median = np.median(epoch_values)
maximum = np.max(epoch_values)

stats_data['Average'].append(avg)
stats_data['Min'].append(minimum)
stats_data['Median'].append(median)
stats_data['Max'].append(maximum)

df = pd.DataFrame(stats_data)

df.to_csv(f'../results/gan/legacy/{experiment}/{tensor_name}_results.csv', index=False)

alpha = 0.001
data = pd.DataFrame(epoch_values, columns=['Loss'])
data['Smoothed Loss'] = data['Loss'].ewm(alpha=alpha).mean()

fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)

ax.plot(data['Loss'], alpha=0.2, label='Loss')
ax.plot(data['Smoothed Loss'], label='Loss (EMA)')
plt.legend()
plt.savefig(f'../results/gan/legacy/{experiment}/{tensor_name}.png')
