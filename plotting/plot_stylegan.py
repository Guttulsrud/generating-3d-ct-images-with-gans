import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
# Load the event files
file_path = '../saved_models/legacy/train/tensorboard/events*'
event_files = tf.io.gfile.glob(file_path)

# epoch_disc_loss
# Create lists to store the x-axis and y-axis values from all files
gen_x_values = []
gen_y_values = []
disc_x_values = []
disc_y_values = []

# Loop through all the event files
for event_file in event_files:
    # Create a summary iterator for the current file
    summary_iterator = tf.compat.v1.train.summary_iterator(event_file)

    # Loop through the summary iterator and extract the desired data
    for event in summary_iterator:
        for value in event.summary.value:
            if value.tag == 'epoch_disc_loss':
                try:
                    disc_x_values.append(event.step)
                    disc_y_values.append(value.simple_value)
                except:
                    pass
            if value.tag == 'epoch_gen_loss':
                try:
                    gen_x_values.append(event.step)
                    gen_y_values.append(value.simple_value)
                except:
                    pass

def plot(y, name):
    alpha = 0.001
    data = pd.DataFrame(y, columns=['Loss'])
    data['Smoothed Loss'] = data['Loss'].ewm(alpha=alpha).mean()

    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)

    ax.plot(data['Loss'], alpha=0.2, label='Loss')
    ax.plot(data['Smoothed Loss'], label='Loss (EMA)')
    plt.legend()
    print(name)
    print(data['Loss'].min())
    print(data['Loss'].max())
    print(data['Loss'].mean())
    # plt.savefig(f'StyleGAN{name}.png')


plot(disc_y_values, 'Discriminator')
plot(gen_y_values, 'Generator')