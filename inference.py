from datetime import datetime

import numpy as np

from utils.config import load_config_file
from utils.init import get_architecture
import matplotlib.pyplot as plt
import os

from utils.plotting import create_plot

config = load_config_file()
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H-%M-%S")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

samples = 1

architecture = get_architecture(config=config)
network = architecture(start_datetime=dt_string, config=config)

for i in range(samples):
    generated_image = network.generate_image()
    generated_image = np.squeeze(generated_image)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Loop over the slices and plot them in the subplots
    for i in range(generated_image.shape[2]):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(generated_image[:, :, i], cmap='gray')
        axs[row, col].set_title(f"Slice {i + 1}", size=15)
        axs[row, col].axis('off')

    # Show the plot
    plt.show()
    fig.savefig(f'generated.png')

